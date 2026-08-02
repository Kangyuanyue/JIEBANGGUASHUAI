#!/usr/bin/env python3
"""Summarize and validate V3 same-start provenance metadata."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


SPEECH_ROLES = {
    "target_speech",
    "interferer_speech",
    "non_target_speech_1",
    "non_target_speech_2",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def describe(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "mean": sum(values) / len(values),
        "max": max(values),
    }


def validate_speaker_rule(row: dict) -> str | None:
    speech = [stream for stream in row["streams"] if stream.get("role") in SPEECH_ROLES]
    if len(speech) != 2:
        return f"expected 2 speech streams, found {len(speech)}"
    target = row["target_member"]
    members = [stream.get("member_index") for stream in speech]
    if row["split"] == "pos" or row["audio_role"] == "wake":
        if speech[0].get("role") != "target_speech" or members[0] != target:
            return "first speech stream is not the target speaker"
        if members[1] == target:
            return "interferer is the target speaker"
    elif target in members or len(set(members)) != 2:
        return "NEG command does not contain two distinct non-wake speakers"
    return None


def analyze(rows: list[dict]) -> dict:
    violations = []
    for row in rows:
        problem = validate_speaker_rule(row)
        if problem:
            violations.append({"output_path": row["output_path"], "problem": problem})

    groups = {}
    for split in ("pos", "neg"):
        for role in ("wake", "command"):
            selected = [row for row in rows if row["split"] == split and row["audio_role"] == role]
            key = f"{split}_{role}"
            groups[key] = {
                "duration_seconds": describe([float(row["duration_sec"]) for row in selected]),
                "tir_db": describe([float(row["tir_db"]) for row in selected]),
                "extreme_tir_below_minus10_count": sum(float(row["tir_db"]) < -10.0 for row in selected),
                "duration_above_5_seconds_count": sum(float(row["duration_sec"]) > 5.0 for row in selected),
                "duration_above_8_seconds_count": sum(float(row["duration_sec"]) > 8.0 for row in selected),
            }

    return {
        "provenance_rows": len(rows),
        "speaker_rule_violation_count": len(violations),
        "speaker_rule_violation_examples": violations[:10],
        "alignment_modes": dict(Counter(row["alignment_mode"] for row in rows)),
        "actual_overlap_ratios": sorted({float(row["actual_overlap_ratio"]) for row in rows}),
        "limiter_applied_count": sum(float(row["limiter_gain"]) < 1.0 for row in rows),
        "target_member_usage": dict(sorted(Counter(str(row["target_member"]) for row in rows).items())),
        "stream_role_counts": dict(Counter(stream["role"] for row in rows for stream in row["streams"])),
        "groups": groups,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Output report already exists; choose a new path")
    report = analyze(read_jsonl(args.provenance))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
