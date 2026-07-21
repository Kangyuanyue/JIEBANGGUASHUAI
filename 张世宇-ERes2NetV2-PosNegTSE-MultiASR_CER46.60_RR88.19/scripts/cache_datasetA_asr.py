#!/usr/bin/env python3
"""Cache ASR transcripts for datasetA command audio.

The speaker gate score is already cached separately. This script runs ASR once
for every command audio, so we can later sweep speaker thresholds and text
post-processing without repeatedly loading/running the ASR model.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from asr import create_asr_backend  # noqa: E402
from audio_quality import preprocess_waveform  # noqa: E402
from audio_utils import load_audio_file  # noqa: E402
from command_postprocess import normalize_command_text  # noqa: E402
from config import apply_env_overrides, load_config  # noqa: E402
from dataset_loader import load_meta  # noqa: E402


def safe_console_text(text: str, limit: int = 40) -> str:
    preview = (text or "")[:limit]
    return preview.encode("gbk", errors="replace").decode("gbk")


def save_cache(path: Path, args: argparse.Namespace, rows: list[dict[str, Any]], total_sec: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "meta": args.meta,
                "audio_root": args.audio_root,
                "config": args.config,
                "n_total": len(rows),
                "total_duration_sec": total_sec,
                "avg_sec_per_sample": total_sec / max(1, len(rows)),
                "samples": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache FunASR transcripts for datasetA.")
    parser.add_argument("--meta", default="output/datasetA_all.jsonl", help="Merged datasetA jsonl.")
    parser.add_argument("--audio-root", default="datasetA", help="Audio root.")
    parser.add_argument("--config", default="configs/datasetA_speaker_tuned.json", help="Pipeline config.")
    parser.add_argument("--output", default="output/datasetA_asr_cache_paraformer.json", help="ASR cache JSON.")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N samples.")
    parser.add_argument("--checkpoint-every", type=int, default=50, help="Save partial cache every N samples.")
    args = parser.parse_args()

    cfg = apply_env_overrides(load_config(args.config))
    samples = load_meta(args.meta, audio_root=args.audio_root)
    if args.limit > 0:
        samples = samples[: args.limit]

    asr = create_asr_backend(cfg.asr)
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    print(f"Loaded {len(samples)} samples from {args.meta}")
    print(f"ASR backend={cfg.asr.backend}, model_dir={cfg.asr.model_dir or cfg.asr.model_name}")
    for i, sample in enumerate(samples, 1):
        start = time.perf_counter()
        try:
            wav, sr = load_audio_file(sample.cmd_audio)
            wav, sr, quality = preprocess_waveform(
                wav,
                sr,
                target_sr=cfg.target_sr,
                enable_vad=cfg.preprocess.enable_vad,
                vad_threshold_ratio=cfg.preprocess.vad_threshold_ratio,
            )
            text = asr.transcribe(wav, sr)
            text = normalize_command_text(text)
            error = ""
        except Exception as e:
            text = ""
            quality = None
            error = str(e)

        elapsed = time.perf_counter() - start
        row = {
            "id": sample.id,
            "cmd_audio": sample.cmd_audio,
            "label": sample.label,
            "is_positive": bool((sample.label or "").strip()),
            "asr_text": text,
            "elapsed_sec": elapsed,
            "error": error,
        }
        if quality is not None:
            row["quality"] = {
                "score": quality.score,
                "snr_db": quality.snr_db,
                "duration_sec": quality.duration_sec,
                "speech_ratio": quality.speech_ratio,
                "no_speech": quality.no_speech,
            }
        rows.append(row)
        if args.checkpoint_every > 0 and i % args.checkpoint_every == 0:
            save_cache(Path(args.output), args, rows, time.perf_counter() - t0)
        print(f"[{i:04d}/{len(samples)}] id={sample.id} text={safe_console_text(text)!r} elapsed={elapsed:.3f}s")

    total = time.perf_counter() - t0
    out = Path(args.output)
    save_cache(out, args, rows, total)
    print(f"\nSaved ASR cache: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
