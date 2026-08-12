from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from funasr import AutoModel


TAG_RE = re.compile(r"<\|[^|]+?\|>")
CANDIDATES = ("mixture", "source0", "source1")


def read_jsonl(path: Path):
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    output = Path(args.out).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty Stage 6 ASR directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    separation = read_jsonl(Path(args.separation_dir) / "separation_results.jsonl")
    if len(separation) != int(config["expected_count"]):
        raise ValueError(f"Expected {config['expected_count']} separation rows, found {len(separation)}")

    device = torch.device(config["device"])
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = AutoModel(model=config["base_model"], device=str(device), trust_remote_code=True, disable_update=True)
    state_dict = torch.load(Path(config["checkpoint"]) / "model.pt", map_location=device, weights_only=True)
    model.model.load_state_dict(state_dict, strict=True)
    model.model.eval()
    del state_dict
    torch.cuda.empty_cache()

    all_metrics = {}
    total_audio_seconds = 0.0
    total_elapsed = 0.0
    for candidate in CANDIDATES:
        predictions = []
        errors = characters = 0
        audio_seconds = 0.0
        started = time.perf_counter()
        for index, row in enumerate(separation, start=1):
            path = row["mixture_path"] if candidate == "mixture" else row["source_paths"][int(candidate[-1])]
            info = sf.info(path)
            audio_seconds += info.frames / info.samplerate
            raw = parse_result(model.generate(
                input=path,
                cache={},
                language="zh",
                use_itn=False,
                batch_size_s=60,
            ))
            reference = normalize(row.get("label", ""))
            hypothesis = normalize(raw)
            sample_errors = edit_distance(reference, hypothesis) if row["is_positive"] else 0
            sample_chars = len(reference) if row["is_positive"] else 0
            errors += sample_errors
            characters += sample_chars
            predictions.append({
                "uid": row["uid"],
                "raw_text": raw,
                "text": hypothesis,
                "reference": row.get("label", ""),
                "is_positive": row["is_positive"],
                "audio_path": path,
                "errors": sample_errors,
                "characters": sample_chars,
                "cer": sample_errors / sample_chars if sample_chars else 0.0,
            })
            if index % 50 == 0 or index == len(separation):
                print(f"{candidate}: {index}/{len(separation)}")
        elapsed = time.perf_counter() - started
        with (output / f"{candidate}_pred.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in predictions:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        all_metrics[candidate] = {
            "count": len(predictions),
            "positive_cer": errors / characters if characters else 0.0,
            "errors": errors,
            "characters": characters,
            "elapsed_seconds": elapsed,
            "audio_seconds": audio_seconds,
            "real_time_factor": elapsed / audio_seconds if audio_seconds else None,
        }
        total_audio_seconds += audio_seconds
        total_elapsed += elapsed
    result = {
        "dataset": config["dataset_name"],
        "base_model": config["base_model"],
        "checkpoint": config["checkpoint"],
        "use_itn": False,
        "vad": False,
        "candidates": all_metrics,
        "total_elapsed_seconds": total_elapsed,
        "total_audio_seconds": total_audio_seconds,
        "combined_real_time_factor": total_elapsed / total_audio_seconds,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "evaluation_only": True,
        "official_test_data_used_for_training": False,
    }
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the locked Stage 6 SenseVoice checkpoint on three candidates.")
    parser.add_argument("--separation-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
