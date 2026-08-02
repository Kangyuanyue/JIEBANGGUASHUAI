#!/usr/bin/env python3
"""Analyze paired V3 ASR results by target-to-interferer ratio."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


TIR_BINS = [
    (-99.0, -5.0, "tir_below_minus5"),
    (-5.0, 0.0, "tir_minus5_to_0"),
    (0.0, 5.0, "tir_0_to_5"),
    (5.0, 99.0, "tir_5_or_above"),
]


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def load_variant(root: Path, pred_path: Path, cer_function) -> dict:
    manifest = read_jsonl(root / "infer_manifest.jsonl")
    predictions = {row["uid"]: row for row in read_jsonl(pred_path)}
    command_provenance = {
        (row["split"], int(row["sample_id"])): row
        for row in read_jsonl(root / "provenance.jsonl")
        if row["audio_role"] == "command"
    }
    positives = []
    for row in manifest:
        if not row["is_positive"]:
            continue
        prediction = predictions[row["uid"]]
        edits, chars = cer_function(row["label"], prediction["text"])
        provenance = command_provenance[("pos", int(row["id"]))]
        positives.append(
            {
                "id": int(row["id"]),
                "edits": edits,
                "chars": chars,
                "tir_db": float(provenance["tir_db"]),
            }
        )
    groups = {}
    for lower, upper, name in TIR_BINS:
        selected = [row for row in positives if lower <= row["tir_db"] < upper]
        total_chars = sum(row["chars"] for row in selected)
        groups[name] = {
            "count": len(selected),
            "cer": sum(row["edits"] for row in selected) / total_chars if total_chars else 0.0,
        }
    return {"positives": {row["id"]: row for row in positives}, "by_tir": groups}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minus10-root", required=True, type=Path)
    parser.add_argument("--minus10-pred", required=True, type=Path)
    parser.add_argument("--minus20-root", required=True, type=Path)
    parser.add_argument("--minus20-pred", required=True, type=Path)
    parser.add_argument("--common-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Output report already exists; choose a new path")

    sys.path.insert(0, str(args.common_dir.resolve()))
    from common import cer  # pylint: disable=import-outside-toplevel

    minus10 = load_variant(args.minus10_root, args.minus10_pred, cer)
    minus20 = load_variant(args.minus20_root, args.minus20_pred, cer)
    paired = {"minus20_better": 0, "same": 0, "minus20_worse": 0}
    for sample_id, first in minus10["positives"].items():
        second = minus20["positives"][sample_id]
        first_cer = first["edits"] / first["chars"] if first["chars"] else 0.0
        second_cer = second["edits"] / second["chars"] if second["chars"] else 0.0
        if second_cer < first_cer:
            paired["minus20_better"] += 1
        elif second_cer > first_cer:
            paired["minus20_worse"] += 1
        else:
            paired["same"] += 1

    report = {
        "minus10_by_tir": minus10["by_tir"],
        "minus20_by_tir": minus20["by_tir"],
        "paired_positive_outcome": paired,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
