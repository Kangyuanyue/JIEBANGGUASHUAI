#!/usr/bin/env python
"""Validate TTS audio quality and optionally compute SenseVoice back-transcription CER."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


SENSEVOICE_TAG = re.compile(r"<\|[^|>]+\|>")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or "")).lower()
    text = SENSEVOICE_TAG.sub("", text)
    return "".join(
        char
        for char in text
        if not char.isspace() and not unicodedata.category(char).startswith("P")
    )


def edit_distance(reference: str, hypothesis: str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for ref_index, ref_char in enumerate(reference, start=1):
        current = [ref_index]
        for hyp_index, hyp_char in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[hyp_index] + 1,
                    previous[hyp_index - 1] + (ref_char != hyp_char),
                )
            )
        previous = current
    return previous[-1]


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthesis-report", type=Path, required=True)
    parser.add_argument("--asr-pred", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--details", type=Path, required=True)
    args = parser.parse_args()
    for output in (args.out, args.details):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite validation output: {output}")

    synthesis = read_jsonl(args.synthesis_report)
    predictions = {}
    if args.asr_pred:
        predictions = {row["uid"]: row for row in read_jsonl(args.asr_pred)}

    details = []
    total_errors = 0
    total_chars = 0
    speaker_stats: dict[str, Counter] = defaultdict(Counter)
    for row in synthesis:
        audio_path = Path(row["output_audio_path"])
        samples, sample_rate = sf.read(audio_path, always_2d=True, dtype="float32")
        mono = samples.shape[1] == 1
        peak = float(np.max(np.abs(samples))) if samples.size else 0.0
        rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
        silence_ratio = float(np.mean(np.abs(samples) < 1e-3)) if samples.size else 1.0
        clipping_ratio = float(np.mean(np.abs(samples) >= 0.999)) if samples.size else 0.0
        duration = len(samples) / sample_rate if sample_rate else 0.0
        pred = predictions.get(row["uid"], {})
        reference = normalize_text(row["text"])
        hypothesis = normalize_text(pred.get("text", "")) if predictions else ""
        errors = edit_distance(reference, hypothesis) if predictions else None
        cer = errors / len(reference) if predictions and reference else None
        if errors is not None:
            total_errors += errors
            total_chars += len(reference)
            speaker_stats[row["speaker"]]["errors"] += errors
            speaker_stats[row["speaker"]]["chars"] += len(reference)
        automatic_pass = bool(
            samples.size
            and mono
            and sample_rate == 24000
            and duration > 0.3
            and peak > 0.01
            and clipping_ratio <= 0.001
            and silence_ratio < 0.95
        )
        speaker_stats[row["speaker"]]["items"] += 1
        speaker_stats[row["speaker"]]["automatic_pass"] += int(automatic_pass)
        details.append(
            {
                **row,
                "asr_raw_text": pred.get("raw_text"),
                "asr_text": pred.get("text"),
                "normalized_reference": reference,
                "normalized_hypothesis": hypothesis if predictions else None,
                "character_errors": errors,
                "cer": round(cer, 6) if cer is not None else None,
                "decoded": bool(samples.size),
                "sample_rate": sample_rate,
                "channels": samples.shape[1],
                "duration_seconds": round(duration, 4),
                "peak_amplitude": round(peak, 6),
                "rms": round(rms, 6),
                "silence_ratio": round(silence_ratio, 6),
                "clipping_ratio": round(clipping_ratio, 8),
                "automatic_quality_pass": automatic_pass,
            }
        )

    per_speaker = {}
    for speaker, stats in sorted(speaker_stats.items()):
        per_speaker[speaker] = {
            "items": stats["items"],
            "automatic_pass": stats["automatic_pass"],
            "cer": round(stats["errors"] / stats["chars"], 6)
            if stats["chars"]
            else None,
        }
    overall_cer = total_errors / total_chars if total_chars else None
    metrics = {
        "items": len(details),
        "decoded": sum(item["decoded"] for item in details),
        "automatic_quality_pass": sum(item["automatic_quality_pass"] for item in details),
        "asr_evaluated": bool(predictions),
        "total_character_errors": total_errors if predictions else None,
        "total_reference_characters": total_chars if predictions else None,
        "cer": round(overall_cer, 6) if overall_cer is not None else None,
        "cer_target": 0.10,
        "cer_target_pass": overall_cer is not None and overall_cer <= 0.10,
        "human_clarity_target": "at_least_11_of_12",
        "human_clarity_status": "pending_user_listening",
        "per_speaker": per_speaker,
    }
    args.details.parent.mkdir(parents=True, exist_ok=True)
    with args.details.open("w", encoding="utf-8", newline="\n") as handle:
        for row in details:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
