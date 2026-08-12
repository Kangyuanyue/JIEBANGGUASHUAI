from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
from funasr import AutoModel

from evaluate_stage5a import edit_distance, normalize, parse_result, resolve_model_reference


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audio_path(row: dict[str, Any]) -> str:
    value = row.get("audio_path") or row.get("audio_cmd") or row.get("output_audio_path")
    if not value:
        raise KeyError(f"Missing audio_path/audio_cmd/output_audio_path for {row.get('uid')}")
    return str(value)


def reference_text(row: dict[str, Any]) -> str:
    return str(row.get("text") or row.get("label") or "")


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest).resolve()
    rows = read_jsonl(manifest_path)
    if not rows:
        raise ValueError("Manifest is empty")

    output = Path(args.out).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty ASR output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    model_ref = resolve_model_reference(args.model)
    model = AutoModel(model=model_ref, device=str(device), trust_remote_code=True, disable_update=True)
    if args.checkpoint:
        state = torch.load(Path(args.checkpoint) / "model.pt", map_location=device, weights_only=True)
        model.model.load_state_dict(state, strict=True)
        model.model.eval()
        del state

    predictions = []
    errors = characters = 0
    audio_seconds = 0.0
    started = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        path = audio_path(row)
        info = sf.info(path)
        audio_seconds += info.frames / info.samplerate
        raw = parse_result(
            model.generate(
                input=path,
                cache={},
                language=args.language,
                use_itn=args.use_itn,
                batch_size_s=60,
            )
        )
        reference = normalize(reference_text(row))
        hypothesis = normalize(raw)
        is_positive = bool(row.get("is_positive", True))
        sample_errors = edit_distance(reference, hypothesis) if is_positive and reference else 0
        sample_chars = len(reference) if is_positive else 0
        errors += sample_errors
        characters += sample_chars
        predictions.append(
            {
                "uid": row.get("uid", str(index)),
                "raw_text": raw,
                "text": hypothesis,
                "reference": reference_text(row),
                "is_positive": is_positive,
                "errors": sample_errors,
                "characters": sample_chars,
                "cer": sample_errors / sample_chars if sample_chars else 0.0,
            }
        )
        if index % 50 == 0 or index == len(rows):
            print(f"asr: {index}/{len(rows)}")

    elapsed = time.perf_counter() - started
    pred_path = output / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics = {
        "count": len(predictions),
        "positive_cer": errors / characters if characters else 0.0,
        "errors": errors,
        "characters": characters,
        "elapsed_seconds": elapsed,
        "audio_seconds": audio_seconds,
        "real_time_factor": elapsed / audio_seconds if audio_seconds else None,
        "model": model_ref,
        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        "use_itn": args.use_itn,
        "vad": False,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SenseVoice on a generic JSONL manifest.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", default="iic/SenseVoiceSmall")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--use-itn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
