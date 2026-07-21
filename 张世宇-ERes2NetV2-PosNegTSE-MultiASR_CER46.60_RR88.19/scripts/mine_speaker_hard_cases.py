#!/usr/bin/env python3
"""Mine hard negative and false-reject speaker trials from score dumps."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def genre(path: str) -> str:
    name = Path(path).name
    return name.split("-")[0] if "-" in name else "unknown"


def write_trials(rows: list[dict], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["enroll_audio", "test_audio", "label"])
        for r in rows:
            writer.writerow([r["enroll_audio"], r["test_audio"], r["label"]])


def summarize(rows: list[dict]) -> dict:
    enroll_genres = Counter(genre(r["enroll_audio"]) for r in rows)
    test_genres = Counter(genre(r["test_audio"]) for r in rows)
    pair_genres = Counter(f"{genre(r['enroll_audio'])}->{genre(r['test_audio'])}" for r in rows)
    return {
        "count": len(rows),
        "top_enroll_genres": enroll_genres.most_common(12),
        "top_test_genres": test_genres.most_common(12),
        "top_pair_genres": pair_genres.most_common(12),
        "score_min": min((float(r["score"]) for r in rows), default=None),
        "score_max": max((float(r["score"]) for r in rows), default=None),
        "score_mean": sum(float(r["score"]) for r in rows) / len(rows) if rows else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Mine hard speaker verification cases")
    parser.add_argument("--score-dump", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--reject-low", type=float, default=0.0)
    parser.add_argument("--accept-high", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=300)
    parser.add_argument("--output-json", default="output/speaker_hard_cases.json")
    parser.add_argument("--hard-trials", default="output/speaker_hard_trials.csv")
    args = parser.parse_args()

    records = json.loads(Path(args.score_dump).read_text(encoding="utf-8"))
    valid = [r for r in records if "score" in r and "label" in r]

    false_accepts = [r for r in valid if int(r["label"]) == 0 and float(r["score"]) >= args.threshold]
    false_rejects = [r for r in valid if int(r["label"]) == 1 and float(r["score"]) < args.threshold]
    uncertain = [
        r
        for r in valid
        if args.reject_low <= float(r["score"]) <= args.accept_high
    ]

    hard_negative_ranked = sorted(
        [r for r in valid if int(r["label"]) == 0],
        key=lambda r: float(r["score"]),
        reverse=True,
    )
    hard_positive_ranked = sorted(
        [r for r in valid if int(r["label"]) == 1],
        key=lambda r: float(r["score"]),
    )

    selected_hard = hard_negative_ranked[: args.top_k] + hard_positive_ranked[: args.top_k]
    selected_hard = sorted(selected_hard, key=lambda r: (int(r["label"]), -float(r["score"])))

    out = {
        "thresholds": {
            "threshold": args.threshold,
            "reject_low": args.reject_low,
            "accept_high": args.accept_high,
        },
        "total": len(valid),
        "false_accepts": summarize(false_accepts),
        "false_rejects": summarize(false_rejects),
        "uncertain_band": summarize(uncertain),
        "hard_negative_top": summarize(hard_negative_ranked[: args.top_k]),
        "hard_positive_low": summarize(hard_positive_ranked[: args.top_k]),
        "examples": {
            "false_accepts": false_accepts[:20],
            "false_rejects": false_rejects[:20],
            "hard_negative_top": hard_negative_ranked[:20],
            "hard_positive_low": hard_positive_ranked[:20],
        },
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    write_trials(selected_hard, Path(args.hard_trials))
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"Hard trials saved to {Path(args.hard_trials).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
