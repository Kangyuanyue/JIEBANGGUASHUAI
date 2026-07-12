#!/usr/bin/env python3
"""Sweep simple score-level fusion for cached speaker-eval dumps."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from speaker_eval import compute_metrics  # noqa: E402


def _load_scores(path: str | Path) -> dict[str, dict[str, Any]]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for i, row in enumerate(rows):
        if "score" not in row:
            continue
        key = str(row.get("id") or f"trial_{i}")
        out[key] = {"score": float(row["score"]), "label": int(row["label"])}
    return out


def _aligned(paths: list[str]) -> tuple[list[str], np.ndarray, list[np.ndarray]]:
    loaded = [_load_scores(path) for path in paths]
    keys = sorted(set.intersection(*(set(x) for x in loaded)))
    if not keys:
        raise ValueError("No common scored trials across dumps.")

    labels = np.asarray([loaded[0][k]["label"] for k in keys], dtype=np.int32)
    scores = []
    for data in loaded:
        other_labels = np.asarray([data[k]["label"] for k in keys], dtype=np.int32)
        if not np.array_equal(labels, other_labels):
            raise ValueError("Aligned trial labels do not match.")
        scores.append(np.asarray([data[k]["score"] for k in keys], dtype=np.float64))
    return keys, labels, scores


def _normalize(scores: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return scores
    if mode == "zscore":
        std = float(np.std(scores))
        return (scores - float(np.mean(scores))) / max(std, 1e-8)
    if mode == "minmax":
        lo = float(np.min(scores))
        hi = float(np.max(scores))
        return (scores - lo) / max(hi - lo, 1e-8)
    raise ValueError(f"Unknown normalization: {mode}")


def _sweep_two(
    labels: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    step: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n_steps = int(round(1.0 / step))
    for i in range(n_steps + 1):
        weight_a = min(1.0, max(0.0, i * step))
        fused = weight_a * score_a + (1.0 - weight_a) * score_b
        metrics = compute_metrics(fused, labels)
        rows.append(
            {
                "weight_first": weight_a,
                "weight_second": 1.0 - weight_a,
                "metrics": asdict(metrics),
                "recommended_gate_threshold": metrics.best_competition_threshold,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep speaker score fusion")
    parser.add_argument("--score-dumps", nargs="+", required=True, help="JSON score dumps to fuse")
    parser.add_argument("--names", nargs="+", default=[])
    parser.add_argument("--normalize", choices=["none", "zscore", "minmax"], default="none")
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if len(args.score_dumps) != 2:
        raise ValueError("This utility currently supports exactly two score dumps.")
    if args.names and len(args.names) != 2:
        raise ValueError("--names must match --score-dumps length.")

    keys, labels, scores = _aligned(args.score_dumps)
    norm_scores = [_normalize(s, args.normalize) for s in scores]

    single = []
    names = args.names or [Path(p).stem for p in args.score_dumps]
    for name, score in zip(names, norm_scores):
        metrics = compute_metrics(score, labels)
        single.append(
            {
                "name": name,
                "metrics": asdict(metrics),
                "recommended_gate_threshold": metrics.best_competition_threshold,
            }
        )

    sweeps = _sweep_two(labels, norm_scores[0], norm_scores[1], args.step)
    best = max(sweeps, key=lambda row: row["metrics"]["best_competition_score"])
    best_eer = min(sweeps, key=lambda row: row["metrics"]["eer"])
    out = {
        "names": names,
        "normalize": args.normalize,
        "n_common_trials": len(keys),
        "n_positive": int((labels == 1).sum()),
        "n_negative": int((labels == 0).sum()),
        "single_models": single,
        "best_by_competition_score": best,
        "best_by_eer": best_eer,
        "sweep": sweeps,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
