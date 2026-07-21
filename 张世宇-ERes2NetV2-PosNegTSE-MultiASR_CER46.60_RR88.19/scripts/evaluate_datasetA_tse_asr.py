#!/usr/bin/env python3
"""Compare raw and TSE-enhanced ASR on an evenly sampled DatasetA subset.

DatasetA labels are used only for development-set reporting. No optimizer or
model parameter is updated in this script.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from asr import create_asr_backend  # noqa: E402
from audio_utils import load_audio_file  # noqa: E402
from command_postprocess import normalize_command_text  # noqa: E402
from config import load_config  # noqa: E402
from dataset_loader import load_meta  # noqa: E402
from metrics_cer import _levenshtein_chars, normalize_text  # noqa: E402
from tse_model import PositiveNegativeTSE, build_pseudo_enrollments  # noqa: E402


def cache_map(path: str) -> dict[str, dict]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(row["id"]): row for row in payload["samples"]}


def evenly_spaced(items: list, count: int) -> list:
    if count <= 0 or count >= len(items):
        return items
    indices = np.linspace(0, len(items) - 1, count, dtype=int)
    return [items[int(index)] for index in indices]


def corpus_cer(rows: list[dict], field: str) -> float:
    refs = sum(len(normalize_text(row["label"])) for row in rows)
    errors = sum(_levenshtein_chars(row["label"], row.get(field, "")) for row in rows)
    return 100.0 * errors / max(1, refs)


def save(path: Path, metadata: dict, rows: list[dict], elapsed: float) -> None:
    summary = {
        **metadata,
        "n_samples": len(rows),
        "elapsed_sec": elapsed,
        "raw_vad_cer": corpus_cer(rows, "raw_vad_text"),
        "raw_full_cer": corpus_cer(rows, "raw_full_text"),
        "tse_cer": corpus_cer(rows, "tse_text"),
        "tse_better_than_raw_vad": sum(
            _levenshtein_chars(row["label"], row["tse_text"])
            < _levenshtein_chars(row["label"], row["raw_vad_text"])
            for row in rows
        ),
        "tse_worse_than_raw_vad": sum(
            _levenshtein_chars(row["label"], row["tse_text"])
            > _levenshtein_chars(row["label"], row["raw_vad_text"])
            for row in rows
        ),
        "samples": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", default="output/datasetA_all.jsonl")
    parser.add_argument("--audio-root", default="datasetA")
    parser.add_argument("--config", default="configs/speaker_frontend_no_vad_15s.json")
    parser.add_argument("--checkpoint", default="pretrained/tse_posneg_cnceleb_stage2.pt")
    parser.add_argument("--raw-vad-cache", default="output/datasetA_asr_cache_paraformer.json")
    parser.add_argument("--raw-full-cache", default="output/datasetA_asr_cache_paraformer_no_vad.json")
    parser.add_argument("--positive-count", type=int, default=200)
    parser.add_argument("--save-audio", type=int, default=4)
    parser.add_argument("--audio-dir", default="output/datasetA_tse_examples")
    parser.add_argument("--output", default="output/datasetA_tse_asr_eval_200.json")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    samples = [sample for sample in load_meta(args.meta, args.audio_root) if sample.label.strip()]
    samples = evenly_spaced(samples, args.positive_count)
    raw_vad = cache_map(args.raw_vad_cache)
    raw_full = cache_map(args.raw_full_cache)
    cfg = load_config(args.config)
    asr = create_asr_backend(cfg.asr)
    tse = PositiveNegativeTSE(checkpoint=args.checkpoint, device=args.device)
    audio_dir = Path(args.audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    output = Path(args.output)
    metadata = {
        "meta": args.meta,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "datasetA_used_for_training": False,
        "selection": "evenly_spaced_positive_development_samples",
    }
    rows = []
    started = time.perf_counter()

    for index, sample in enumerate(samples, 1):
        wake, wake_sr = load_audio_file(sample.wake_audio)
        command, command_sr = load_audio_file(sample.cmd_audio)
        enrollments = build_pseudo_enrollments(wake, wake_sr)
        extraction = tse.extract(
            command,
            command_sr,
            enrollments.positive,
            enrollments.sample_rate,
            enrollments.negative,
            enrollments.sample_rate,
        )
        tse_text = normalize_command_text(
            asr.transcribe(extraction.waveform, extraction.sample_rate)
        )
        sample_id = str(sample.id)
        item = {
            "id": sample_id,
            "label": normalize_command_text(sample.label),
            "raw_vad_text": normalize_command_text(raw_vad.get(sample_id, {}).get("asr_text", "")),
            "raw_full_text": normalize_command_text(raw_full.get(sample_id, {}).get("asr_text", "")),
            "tse_text": tse_text,
            "negative_enrollment_source": enrollments.negative_source,
            "tse_elapsed_sec": extraction.elapsed_sec,
            "input_rms": extraction.input_rms,
            "output_rms": extraction.output_rms,
        }
        rows.append(item)
        if index <= args.save_audio:
            sf.write(audio_dir / f"{sample_id}_raw.wav", command, command_sr)
            sf.write(audio_dir / f"{sample_id}_tse.wav", extraction.waveform, extraction.sample_rate)
        if args.checkpoint_every > 0 and index % args.checkpoint_every == 0:
            save(output, metadata, rows, time.perf_counter() - started)
        print(
            f"[{index:04d}/{len(samples)}] {sample_id} "
            f"raw={item['raw_vad_text'][:24]!r} tse={tse_text[:24]!r}"
        )

    save(output, metadata, rows, time.perf_counter() - started)
    payload = json.loads(output.read_text(encoding="utf-8"))
    print(json.dumps({k: v for k, v in payload.items() if k != "samples"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
