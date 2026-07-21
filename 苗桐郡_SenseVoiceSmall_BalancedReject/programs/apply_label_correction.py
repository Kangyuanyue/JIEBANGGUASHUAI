#!/usr/bin/env python3
"""Snap ASR hypotheses to known train-set command labels when the match is safe."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from common import (
    extract_prediction_text,
    levenshtein,
    load_simple_yaml,
    normalize_text,
    read_jsonl,
    write_jsonl,
)


DEFAULT_CONFIG = {
    "min_edit_similarity": 0.50,
    "min_margin": 0.10,
    "min_edit_distance": 0,
    "max_edit_distance": 999,
    "max_length_delta": 999,
}


def load_labels(train_manifest: Path) -> List[Dict[str, Any]]:
    labels: List[Dict[str, Any]] = []
    seen = set()
    for row in read_jsonl(train_manifest):
        if not row.get("is_positive"):
            continue
        label = row.get("label", "")
        normalized = normalize_text(label)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        labels.append({"label": label, "normalized": normalized})
    if not labels:
        raise ValueError(f"No positive labels found in {train_manifest}")
    return labels


def edit_similarity(a: str, b: str) -> Tuple[float, int]:
    if not a or not b:
        return 0.0, max(len(a), len(b))
    distance = levenshtein(list(a), list(b))
    denom = max(len(a), len(b), 1)
    return max(0.0, 1.0 - distance / denom), distance


def best_label(text: str, labels: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    normalized = normalize_text(text)
    best: Dict[str, Any] = {
        "label": "",
        "normalized": "",
        "source_length": len(normalized),
        "label_length": 0,
        "length_delta": 999,
        "edit_similarity": 0.0,
        "edit_distance": 999,
        "margin": 0.0,
    }
    second_score = 0.0
    for item in labels:
        score, distance = edit_similarity(normalized, item["normalized"])
        if score > best["edit_similarity"]:
            second_score = best["edit_similarity"]
            best = {
                "label": item["label"],
                "normalized": item["normalized"],
                "source_length": len(normalized),
                "label_length": len(item["normalized"]),
                "length_delta": abs(len(normalized) - len(item["normalized"])),
                "edit_similarity": score,
                "edit_distance": distance,
                "margin": score - second_score,
            }
        elif score > second_score:
            second_score = score
            best["margin"] = best["edit_similarity"] - second_score
    return best


def should_correct(match: Dict[str, Any], config: Dict[str, Any]) -> bool:
    return (
        float(match["edit_similarity"]) >= float(config.get("min_edit_similarity", 0.0))
        and float(match["margin"]) >= float(config.get("min_margin", 0.0))
        and int(match["edit_distance"]) >= int(config.get("min_edit_distance", 0))
        and int(match["edit_distance"]) <= int(config.get("max_edit_distance", 999))
        and int(match.get("length_delta", 999)) <= int(config.get("max_length_delta", 999))
    )


def apply_correction(
    pred_rows: Sequence[Dict[str, Any]],
    labels: Sequence[Dict[str, Any]],
    config: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    corrected_rows: List[Dict[str, Any]] = []
    corrected_count = 0
    for row in pred_rows:
        text = extract_prediction_text(row)
        match = best_label(text, labels)
        corrected = should_correct(match, config)
        out = dict(row)
        out["source_text"] = text
        out["label_correction"] = {
            "applied": corrected,
            "label": match["label"],
            "edit_similarity": match["edit_similarity"],
            "edit_distance": match["edit_distance"],
            "length_delta": match["length_delta"],
            "margin": match["margin"],
            "config": config,
        }
        if corrected:
            out["text"] = match["label"]
            out["raw_text"] = match["label"]
            corrected_count += 1
        corrected_rows.append(out)
    return corrected_rows, {"corrected_count": corrected_count, "count": len(corrected_rows)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--train-manifest", default=Path("data/splits/train.jsonl"), type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--min-edit-similarity", type=float)
    parser.add_argument("--min-margin", type=float)
    parser.add_argument("--min-edit-distance", type=int)
    parser.add_argument("--max-edit-distance", type=int)
    parser.add_argument("--max-length-delta", type=int)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    train_manifest = args.train_manifest if args.train_manifest.is_absolute() else project / args.train_manifest

    config = dict(DEFAULT_CONFIG)
    if args.config:
        config.update(load_simple_yaml(args.config))
    if args.min_edit_similarity is not None:
        config["min_edit_similarity"] = args.min_edit_similarity
    if args.min_margin is not None:
        config["min_margin"] = args.min_margin
    if args.min_edit_distance is not None:
        config["min_edit_distance"] = args.min_edit_distance
    if args.max_edit_distance is not None:
        config["max_edit_distance"] = args.max_edit_distance
    if args.max_length_delta is not None:
        config["max_length_delta"] = args.max_length_delta

    labels = load_labels(train_manifest)
    started = time.time()
    corrected_rows, stats = apply_correction(read_jsonl(args.pred), labels, config)
    write_jsonl(args.out, corrected_rows)

    meta = {
        "pred": str(args.pred.resolve()),
        "train_manifest": str(train_manifest.resolve()),
        "config": config,
        "label_count": len(labels),
        "corrected_count": stats["corrected_count"],
        "count": stats["count"],
        "elapsed_seconds": time.time() - started,
    }
    args.out.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.out}")
    print(f"Corrected {stats['corrected_count']}/{stats['count']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
