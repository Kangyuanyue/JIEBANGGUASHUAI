from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

import torch
from funasr import AutoModel


TAG_RE = re.compile(r"<\|[^|]+?\|>")


def resolve_model_reference(value: str) -> str:
    path = Path(value).expanduser()
    return str(path.resolve()) if path.exists() else value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize(text: str | None) -> str:
    text = TAG_RE.sub("", str(text or "")).lower()
    return "".join(char for char in text if not char.isspace() and not unicodedata.category(char).startswith("P"))


def edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for index, ref_char in enumerate(reference, start=1):
        current = [index]
        for other_index, hyp_char in enumerate(hypothesis, start=1):
            current.append(min(current[-1] + 1, previous[other_index] + 1, previous[other_index - 1] + (ref_char != hyp_char)))
        previous = current
    return previous[-1]


def parse_result(result: Any) -> str:
    if isinstance(result, list):
        return "".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in result)
    if isinstance(result, dict):
        return str(result.get("text", ""))
    return str(result)


def evaluate_model(
    base_model_path: str,
    checkpoint_path: str | None,
    manifests: dict[str, Path],
    device: str,
    output_dir: Path,
) -> dict[str, Any]:
    cuda_device = torch.device(device)
    torch.cuda.set_device(cuda_device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = AutoModel(model=base_model_path, device=device, trust_remote_code=True, disable_update=True)
    if checkpoint_path:
        state_dict = torch.load(Path(checkpoint_path) / "model.pt", map_location=cuda_device, weights_only=True)
        model.model.load_state_dict(state_dict, strict=True)
        model.model.eval()
    results = {}
    reload_probe = None
    for dataset_name, manifest_path in manifests.items():
        rows = read_jsonl(manifest_path)
        predictions = []
        total_errors = 0
        total_characters = 0
        started = time.perf_counter()
        for row in rows:
            raw = parse_result(model.generate(
                input=row["audio_path"],
                cache={},
                language="zh",
                use_itn=False,
                batch_size_s=60,
            ))
            reference = normalize(row["text"])
            hypothesis = normalize(raw)
            errors = edit_distance(reference, hypothesis)
            total_errors += errors
            total_characters += len(reference)
            predictions.append({
                "uid": row["uid"],
                "audio_path": row["audio_path"],
                "reference": row["text"],
                "raw_text": raw,
                "hypothesis": hypothesis,
                "errors": errors,
                "characters": len(reference),
                "cer": errors / len(reference) if reference else 0.0,
            })
            if reload_probe is None:
                reload_probe = {"uid": row["uid"], "raw_text": raw, "nonempty": bool(hypothesis)}
        elapsed = time.perf_counter() - started
        pred_path = output_dir / f"{dataset_name}_pred.jsonl"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with pred_path.open("w", encoding="utf-8", newline="\n") as handle:
            for prediction in predictions:
                handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
        results[dataset_name] = {
            "count": len(rows),
            "errors": total_errors,
            "characters": total_characters,
            "cer": total_errors / total_characters if total_characters else 0.0,
            "elapsed_seconds": elapsed,
            "real_time_audio_seconds_per_wall_second": sum(1 for _ in rows) / elapsed if elapsed else None,
            "prediction_path": str(pred_path),
        }
    results["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated()
    results["checkpoint_reload_inference"] = reload_probe
    del model
    torch.cuda.empty_cache()
    return results


def run(args: argparse.Namespace) -> dict[str, Any]:
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output_dir = Path(args.out).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty evaluation directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifests = {
        "internal_dev": Path(args.internal_dev).resolve(),
        "real_dev": Path(args.real_dev).resolve(),
        "real_holdout": Path(args.real_holdout).resolve(),
    }
    base_model = resolve_model_reference(cfg["base_model"])
    models = {
        "baseline": None,
        "real_only": str(Path(args.real_checkpoint).resolve()),
        "real_plus_synthetic": str(Path(args.augmented_checkpoint).resolve()),
    }
    metrics = {}
    for name, checkpoint_path in models.items():
        metrics[name] = evaluate_model(base_model, checkpoint_path, manifests, cfg["device"], output_dir / name)

    improvement = metrics["real_only"]["internal_dev"]["cer"] - metrics["real_plus_synthetic"]["internal_dev"]["cer"]
    best_reference_holdout = min(metrics["baseline"]["real_holdout"]["cer"], metrics["real_only"]["real_holdout"]["cer"])
    holdout_degradation = metrics["real_plus_synthetic"]["real_holdout"]["cer"] - best_reference_holdout
    gate = {
        "synthetic_dev_improvement": improvement,
        "required_improvement": cfg["synthetic_dev_min_improvement"],
        "real_holdout_degradation": holdout_degradation,
        "maximum_holdout_degradation": cfg["real_holdout_max_degradation"],
        "synthetic_improvement_pass": improvement >= cfg["synthetic_dev_min_improvement"],
        "real_holdout_pass": holdout_degradation <= cfg["real_holdout_max_degradation"],
    }
    gate["stage5a_pass"] = gate["synthetic_improvement_pass"] and gate["real_holdout_pass"]
    result = {"base_model": base_model, "checkpoints": models, "metrics": metrics, "gate": gate}
    (output_dir / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline and two Stage 5A SenseVoice checkpoints.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--real-checkpoint", required=True)
    parser.add_argument("--augmented-checkpoint", required=True)
    parser.add_argument("--internal-dev", required=True)
    parser.add_argument("--real-dev", required=True)
    parser.add_argument("--real-holdout", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"gate": result["gate"], "cer": {m: {d: x["cer"] for d, x in v.items() if isinstance(x, dict) and "cer" in x} for m, v in result["metrics"].items()}}, ensure_ascii=False, indent=2))
