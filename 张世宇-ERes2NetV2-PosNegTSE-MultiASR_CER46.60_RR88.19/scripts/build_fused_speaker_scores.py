#!/usr/bin/env python3
"""Build a DatasetA-style fused speaker score dump from multiple score dumps."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.evaluate_datasetA_fusion_cv import sample_key  # noqa: E402


def _load_dump(path: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw["samples"] if isinstance(raw, dict) and "samples" in raw else raw
    return {sample_key(r): r for r in rows if "score" in r}


def main() -> int:
    parser = argparse.ArgumentParser(description="Fuse speaker score dumps.")
    parser.add_argument("--score-dumps", nargs="+", required=True)
    parser.add_argument("--weights", nargs="+", type=float, required=True)
    parser.add_argument("--names", nargs="+", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if len(args.score_dumps) != len(args.weights):
        raise ValueError("--score-dumps and --weights must have the same length.")
    if args.names and len(args.names) != len(args.score_dumps):
        raise ValueError("--names must match --score-dumps length.")

    dumps = [_load_dump(Path(p)) for p in args.score_dumps]
    keys = sorted(set.intersection(*(set(d) for d in dumps)))
    if not keys:
        raise ValueError("No common scored samples across dumps.")

    weight_sum = sum(args.weights)
    if weight_sum <= 0:
        raise ValueError("Sum of weights must be positive.")
    weights = [w / weight_sum for w in args.weights]
    names = args.names or [Path(p).stem for p in args.score_dumps]

    fused_rows = []
    for key in keys:
        base = deepcopy(dumps[0][key])
        parts = {name: float(d[key].get("score", 0.0)) for name, d in zip(names, dumps)}
        fused = sum(w * float(d[key].get("score", 0.0)) for w, d in zip(weights, dumps))
        base["score"] = float(fused)
        base["accepted"] = False
        base["reason"] = "fused_score_unthresholded"
        base["backend_scores"] = parts
        base["fusion"] = {
            "score_dumps": [str(Path(p)) for p in args.score_dumps],
            "names": names,
            "weights": weights,
        }
        fused_rows.append(base)

    out = {
        "fusion": {
            "score_dumps": [str(Path(p)) for p in args.score_dumps],
            "names": names,
            "weights": weights,
            "n_common_samples": len(fused_rows),
        },
        "samples": fused_rows,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out["fusion"], ensure_ascii=False, indent=2))
    print(f"Saved: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
