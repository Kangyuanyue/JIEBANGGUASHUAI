#!/usr/bin/env python
"""Run reproducible local CosyVoice zero-shot synthesis from a JSONL manifest."""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio


def _soundfile_load(path: str | Path, *args, **kwargs) -> tuple[torch.Tensor, int]:
    """Compatibility loader for torchaudio 2.11 on Windows without shared FFmpeg DLLs."""
    del args, kwargs
    samples, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    return torch.from_numpy(samples.T.copy()), sample_rate


torchaudio.load = _soundfile_load


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COSYVOICE_ROOT = PROJECT_ROOT / "third_party" / "CosyVoice"
MATCHA_ROOT = COSYVOICE_ROOT / "third_party" / "Matcha-TTS"
for dependency_path in (COSYVOICE_ROOT, MATCHA_ROOT):
    sys.path.insert(0, str(dependency_path))

def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models" / "Fun-CosyVoice3-0.5B",
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--speaker-similarity", action="store_true")
    parser.add_argument(
        "--text-frontend",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disabled by default so synthesis is fully offline and deterministic.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.text_frontend:
        offline_wetext = types.ModuleType("wetext")

        class OfflineNormalizer:
            def __init__(self, *unused_args, **unused_kwargs):
                raise RuntimeError("WeText is intentionally disabled for offline synthesis")

        offline_wetext.Normalizer = OfflineNormalizer
        sys.modules["wetext"] = offline_wetext

    from cosyvoice.cli.cosyvoice import AutoModel

    rows = read_jsonl(args.manifest.resolve())
    if not rows:
        raise ValueError("The synthesis manifest is empty")
    report_path = args.report.resolve()
    if report_path.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite existing report: {report_path}")
    completed_rows = read_jsonl(report_path) if report_path.exists() else []
    completed_uids = {row["uid"] for row in completed_rows}
    if len(completed_uids) != len(completed_rows):
        raise ValueError(f"Duplicate UIDs in existing report: {report_path}")

    required = {"uid", "text", "prompt_text", "prompt_audio_path", "output_audio_path"}
    for index, row in enumerate(rows, start=1):
        missing = required.difference(row)
        if missing:
            raise ValueError(f"Row {index} is missing fields: {sorted(missing)}")
        prompt = Path(row["prompt_audio_path"])
        output = Path(row["output_audio_path"])
        if not prompt.is_file():
            raise FileNotFoundError(f"Prompt audio does not exist: {prompt}")
        if row["uid"] in completed_uids and not output.is_file():
            raise FileNotFoundError(f"Completed report row has no audio: {output}")
        if output.exists() and row["uid"] not in completed_uids:
            raise FileExistsError(f"Refusing to overwrite generated audio: {output}")

    pending_rows = [row for row in rows if row["uid"] not in completed_uids]
    if not pending_rows:
        print(json.dumps({"generated": 0, "resumed": len(completed_rows), "total": len(rows)}, ensure_ascii=False))
        return

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    model = AutoModel(model_dir=str(args.model.resolve()), fp16=args.fp16)
    model_load_seconds = time.perf_counter() - load_started

    prompt_embeddings = {}
    generated_count = 0
    for row in pending_rows:
        output = Path(row["output_audio_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        chunks = list(
            model.inference_zero_shot(
                row["text"],
                row["prompt_text"],
                row["prompt_audio_path"],
                stream=False,
                speed=float(row.get("speed", 1.0)),
                text_frontend=args.text_frontend,
            )
        )
        if not chunks:
            raise RuntimeError(f"CosyVoice returned no audio for {row['uid']}")
        speech = torch.cat([chunk["tts_speech"] for chunk in chunks], dim=1)
        samples = speech.squeeze(0).detach().float().cpu().numpy()
        sf.write(output, samples, model.sample_rate, subtype="PCM_16")
        speaker_similarity = None
        if args.speaker_similarity:
            prompt_path = row["prompt_audio_path"]
            if prompt_path not in prompt_embeddings:
                prompt_embeddings[prompt_path] = model.frontend._extract_spk_embedding(prompt_path)
            generated_embedding = model.frontend._extract_spk_embedding(str(output))
            speaker_similarity = float(
                torch.nn.functional.cosine_similarity(
                    prompt_embeddings[prompt_path], generated_embedding, dim=1
                ).item()
            )
        peak_gpu_mb = (
            round(torch.cuda.max_memory_allocated() / 1024**2, 2)
            if torch.cuda.is_available()
            else 0.0
        )
        result = {
            **row,
            "tts_engine": "Fun-CosyVoice3-0.5B-2512",
            "sample_rate": model.sample_rate,
            "channels": 1,
            "duration_seconds": round(len(samples) / model.sample_rate, 4),
            "inference_seconds": round(time.perf_counter() - started, 4),
            "peak_amplitude": round(float(np.max(np.abs(samples))), 6),
            "speaker_similarity": round(speaker_similarity, 6) if speaker_similarity is not None else None,
            "model_load_seconds": round(model_load_seconds, 4),
            "peak_gpu_memory_mb": peak_gpu_mb,
        }
        append_jsonl(report_path, result)
        generated_count += 1
        completed = len(completed_rows) + generated_count
        if generated_count == 1 or completed % 10 == 0 or completed == len(rows):
            print(json.dumps({"completed": completed, "total": len(rows), "uid": row["uid"]}, ensure_ascii=False), flush=True)

    print(
        json.dumps(
            {
                "generated": generated_count,
                "resumed": len(completed_rows),
                "total": len(rows),
                "report": str(report_path),
                "model_load_seconds": round(model_load_seconds, 2),
                "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2)
                if torch.cuda.is_available()
                else 0.0,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
