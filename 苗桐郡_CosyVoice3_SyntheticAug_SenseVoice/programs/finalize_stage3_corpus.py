#!/usr/bin/env python
"""Filter stage-3 TTS results and create preserved 16 kHz training copies."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def failure_reasons(row: dict, config: dict) -> list[str]:
    reasons = []
    if not row["decoded"]:
        reasons.append("decode_failed")
    if row["channels"] != 1:
        reasons.append("not_mono")
    if not (config["min_duration_seconds"] <= row["duration_seconds"] <= config["max_duration_seconds"]):
        reasons.append("duration_out_of_range")
    if row["peak_amplitude"] < config["min_peak_amplitude"]:
        reasons.append("peak_too_low")
    if row["rms"] < config["min_rms"]:
        reasons.append("rms_too_low")
    if row["silence_ratio"] > config["max_silence_ratio"]:
        reasons.append("too_much_silence")
    if row["clipping_ratio"] > config["max_clipping_ratio"]:
        reasons.append("clipping")
    if row["cer"] is None or row["cer"] > config["max_sample_cer"]:
        reasons.append("asr_cer_too_high")
    if row.get("speaker_similarity") is None or row["speaker_similarity"] < config["min_speaker_similarity"]:
        reasons.append("speaker_similarity_too_low")
    return reasons


def resample_to_training_format(source: Path, destination: Path, target_rate: int) -> None:
    samples, sample_rate = sf.read(source, always_2d=True, dtype="float32")
    mono = np.mean(samples, axis=1)
    if sample_rate != target_rate:
        divisor = gcd(sample_rate, target_rate)
        mono = resample_poly(mono, target_rate // divisor, sample_rate // divisor).astype(np.float32)
    sf.write(destination, mono, target_rate, subtype="PCM_16")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--details", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--pass-manifest", type=Path, required=True)
    parser.add_argument("--fail-manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.pass_manifest, args.fail_manifest, args.summary):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite: {output}")
    if args.audio_dir.exists():
        raise FileExistsError(f"Refusing to reuse training audio directory: {args.audio_dir}")

    rows = read_jsonl(args.details)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if len(rows) != 720:
        raise ValueError(f"Expected 720 quality rows, found {len(rows)}")
    args.audio_dir.mkdir(parents=True)

    passed = []
    failed = []
    reason_counts = Counter()
    split_stats: dict[str, Counter] = defaultdict(Counter)
    speaker_stats: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        reasons = failure_reasons(row, config)
        split_stats[row["split"]]["total"] += 1
        speaker_stats[row["speaker"]]["total"] += 1
        if reasons:
            failed.append({**row, "quality_pass": False, "failure_reasons": reasons})
            split_stats[row["split"]]["failed"] += 1
            speaker_stats[row["speaker"]]["failed"] += 1
            reason_counts.update(reasons)
            continue
        destination = (args.audio_dir / f"{row['uid']}.wav").resolve()
        resample_to_training_format(
            Path(row["output_audio_path"]), destination, int(config["training_sample_rate"])
        )
        info = sf.info(destination)
        if info.samplerate != config["training_sample_rate"] or info.channels != config["training_channels"]:
            raise RuntimeError(f"Training copy format check failed: {destination}")
        passed.append(
            {
                "uid": row["uid"],
                "text_uid": row["text_uid"],
                "text": row["text"],
                "audio_path": str(destination),
                "source_audio_path": row["output_audio_path"],
                "speaker": row["speaker"],
                "speaker_display_name": row["speaker_display_name"],
                "source": row["source"],
                "source_id": row.get("source_id"),
                "intent": row["intent"],
                "template_id": row["template_id"],
                "group_id": row["group_id"],
                "split": row["split"],
                "speed": row["speed"],
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "duration_seconds": round(info.duration, 4),
                "asr_text": row["asr_text"],
                "cer": row["cer"],
                "speaker_similarity": row["speaker_similarity"],
                "quality_pass": True,
                "failure_reasons": [],
                "allowed_for_training": True,
                "contains_official_test_data": False,
            }
        )
        split_stats[row["split"]]["passed"] += 1
        speaker_stats[row["speaker"]]["passed"] += 1

    write_jsonl(args.pass_manifest, passed)
    write_jsonl(args.fail_manifest, failed)
    summary = {
        "total": len(rows),
        "passed": len(passed),
        "failed": len(failed),
        "pass_rate": round(len(passed) / len(rows), 6),
        "quality_config": config,
        "failure_reason_counts": dict(sorted(reason_counts.items())),
        "split_stats": {key: dict(value) for key, value in sorted(split_stats.items())},
        "speaker_stats": {key: dict(value) for key, value in sorted(speaker_stats.items())},
        "training_audio_files": len(list(args.audio_dir.glob("*.wav"))),
        "original_audio_preserved": True,
        "official_test_data_used": False,
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
