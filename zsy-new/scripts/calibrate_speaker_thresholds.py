#!/usr/bin/env python3
"""Calibrate speaker verification thresholds from a speaker score dump."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rates(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    pred = scores >= threshold
    pos = labels == 1
    neg = labels == 0
    tp = int((pred & pos).sum())
    fn = int((~pred & pos).sum())
    tn = int((~pred & neg).sum())
    fp = int((pred & neg).sum())
    pos_total = max(1, int(pos.sum()))
    neg_total = max(1, int(neg.sum()))
    accept_recall = tp / pos_total
    reject_rate = tn / neg_total
    far = fp / neg_total
    frr = fn / pos_total
    competition_score = 0.4 * accept_recall + 0.4 * reject_rate
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "accept_recall": accept_recall,
        "reject_rate": reject_rate,
        "far": far,
        "frr": frr,
        "competition_score": competition_score,
    }


def pick_min_abs_gap(rows: list[dict], a: str, b: str) -> dict:
    return min(rows, key=lambda r: abs(r[a] - r[b]))


def pick_best_competition(rows: list[dict]) -> dict:
    return max(rows, key=lambda r: (r["competition_score"], r["reject_rate"], r["accept_recall"]))


def pick_for_reject_rate(rows: list[dict], target_rr: float) -> dict:
    feasible = [r for r in rows if r["reject_rate"] >= target_rr]
    if feasible:
        return max(feasible, key=lambda r: (r["accept_recall"], r["competition_score"]))
    return max(rows, key=lambda r: r["reject_rate"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate speaker thresholds")
    parser.add_argument("--score-dump", required=True)
    parser.add_argument("--output", default="output/speaker_threshold_calibration.json")
    parser.add_argument("--target-rr", type=float, nargs="*", default=[0.90, 0.95, 0.98, 0.99])
    args = parser.parse_args()

    records = json.loads(Path(args.score_dump).read_text(encoding="utf-8"))
    valid = [r for r in records if "score" in r and "label" in r]
    if not valid:
        raise SystemExit("No valid score records found.")

    scores = np.asarray([float(r["score"]) for r in valid], dtype=np.float64)
    labels = np.asarray([int(r["label"]) for r in valid], dtype=np.int32)
    thresholds = np.unique(
        np.concatenate(
            [
                [scores.min() - 1e-6],
                scores,
                [scores.max() + 1e-6],
            ]
        )
    )
    rows = [rates(scores, labels, float(th)) for th in thresholds]

    eer = pick_min_abs_gap(rows, "far", "frr")
    best_comp = pick_best_competition(rows)
    rr_points = {f"rr_{int(target * 100)}": pick_for_reject_rate(rows, target) for target in args.target_rr}

    # Useful two-threshold gate: reject below low, accept above high, send middle
    # to heavier downstream checks or ASR confidence fusion.
    # Low threshold for safe early rejection: below this threshold we catch many
    # negatives while keeping about 98% of positives above it.
    reject_low_candidates = [r for r in rows if r["accept_recall"] >= 0.98]
    reject_low = max(reject_low_candidates, key=lambda r: r["reject_rate"]) if reject_low_candidates else best_comp

    # High threshold for high-confidence acceptance: above this threshold false
    # accepts are around 1% on the calibration set.
    accept_high = pick_for_reject_rate(rows, 0.99)

    out = {
        "source": str(Path(args.score_dump).resolve()),
        "n_trials": int(scores.size),
        "n_positive": int((labels == 1).sum()),
        "n_negative": int((labels == 0).sum()),
        "score_stats": {
            "positive_mean": float(scores[labels == 1].mean()),
            "positive_std": float(scores[labels == 1].std()),
            "negative_mean": float(scores[labels == 0].mean()),
            "negative_std": float(scores[labels == 0].std()),
            "positive_p05": float(np.percentile(scores[labels == 1], 5)),
            "negative_p95": float(np.percentile(scores[labels == 0], 95)),
        },
        "recommended": {
            "gate_threshold": best_comp["threshold"],
            "speaker_reject_low": reject_low["threshold"],
            "speaker_accept_high": accept_high["threshold"],
            "reason": "best competition-weighted balance on this calibration set",
        },
        "operating_points": {
            "eer": eer,
            "best_competition": best_comp,
            **rr_points,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
