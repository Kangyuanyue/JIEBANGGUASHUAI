#!/usr/bin/env python3
"""Fuse two prediction files by scoring each candidate with train-label similarity."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from common import (
    extract_prediction_text,
    label_edit_similarity,
    label_similarity,
    load_simple_yaml,
    normalize_text,
    read_jsonl,
    text_domain_score,
    write_jsonl,
)


DEFAULT_CONFIG = {
    "weight_edit_similarity": 0.5,
    "weight_ngram_similarity": 2.0,
    "weight_domain_score": 0.0,
    "weight_length": 0.02,
    "weight_over_length": 0.05,
    "length_cap": 8,
    "margin": 0.03,
    "tie_breaker": "vad",
}


def load_labels(train_manifest: Path) -> List[str]:
    return [row.get("label", "") for row in read_jsonl(train_manifest) if row.get("is_positive")]


def candidate_features(name: str, row: Dict[str, Any], labels: List[str]) -> Dict[str, Any]:
    raw_text = extract_prediction_text(row)
    text = normalize_text(raw_text)
    return {
        "name": name,
        "uid": row.get("uid"),
        "raw_text": raw_text,
        "text": text,
        "length": len(text),
        "edit_similarity": label_edit_similarity(text, labels),
        "ngram_similarity": label_similarity(text, labels),
        "domain_score": text_domain_score(text),
        "source_error": row.get("error"),
    }


def score_candidate(candidate: Dict[str, Any], config: Dict[str, Any]) -> float:
    length_cap = int(config.get("length_cap", 0))
    over_length = max(0, candidate["length"] - length_cap) if length_cap > 0 else 0
    return (
        float(config.get("weight_edit_similarity", 0.0)) * candidate["edit_similarity"]
        + float(config.get("weight_ngram_similarity", 0.0)) * candidate["ngram_similarity"]
        + float(config.get("weight_domain_score", 0.0)) * candidate["domain_score"]
        - float(config.get("weight_length", 0.0)) * candidate["length"]
        - float(config.get("weight_over_length", 0.0)) * over_length
    )


def choose_candidate(a: Dict[str, Any], b: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    score_a = score_candidate(a, config)
    score_b = score_candidate(b, config)
    a["fusion_score"] = score_a
    b["fusion_score"] = score_b
    margin = float(config.get("margin", 0.0))
    if abs(score_a - score_b) <= margin:
        tie_breaker = str(config.get("tie_breaker", "vad"))
        return a if a["name"] == tie_breaker else b
    return a if score_a > score_b else b


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--pred-a", required=True, type=Path)
    parser.add_argument("--pred-b", required=True, type=Path)
    parser.add_argument("--name-a", default="vad")
    parser.add_argument("--name-b", default="novad")
    parser.add_argument("--train-manifest", default=Path("data/splits/train.jsonl"), type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    train_manifest = args.train_manifest if args.train_manifest.is_absolute() else project / args.train_manifest
    config = dict(DEFAULT_CONFIG)
    if args.config:
        config.update(load_simple_yaml(args.config))

    labels = load_labels(train_manifest)
    pred_a = {row["uid"]: row for row in read_jsonl(args.pred_a)}
    pred_b = {row["uid"]: row for row in read_jsonl(args.pred_b)}
    manifest_rows = read_jsonl(args.manifest)

    fused = []
    counts = {args.name_a: 0, args.name_b: 0}
    started = time.time()
    for item in manifest_rows:
        uid = item["uid"]
        features_a = candidate_features(args.name_a, pred_a.get(uid, {"uid": uid, "text": ""}), labels)
        features_b = candidate_features(args.name_b, pred_b.get(uid, {"uid": uid, "text": ""}), labels)
        selected = choose_candidate(features_a, features_b, config)
        counts[selected["name"]] = counts.get(selected["name"], 0) + 1
        fused.append(
            {
                "uid": uid,
                "audio_cmd": item.get("audio_cmd"),
                "raw_text": selected["raw_text"],
                "text": selected["text"],
                "model": "fusion",
                "selected_source": selected["name"],
                "fusion_config": config,
                "fusion_candidates": [
                    {
                        "name": features_a["name"],
                        "text": features_a["text"],
                        "score": features_a["fusion_score"],
                        "length": features_a["length"],
                        "edit_similarity": features_a["edit_similarity"],
                        "ngram_similarity": features_a["ngram_similarity"],
                    },
                    {
                        "name": features_b["name"],
                        "text": features_b["text"],
                        "score": features_b["fusion_score"],
                        "length": features_b["length"],
                        "edit_similarity": features_b["edit_similarity"],
                        "ngram_similarity": features_b["ngram_similarity"],
                    },
                ],
                "error": selected.get("source_error"),
            }
        )

    write_jsonl(args.out, fused)
    meta_path = args.out.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "manifest": str(args.manifest.resolve()),
                "pred_a": str(args.pred_a.resolve()),
                "pred_b": str(args.pred_b.resolve()),
                "name_a": args.name_a,
                "name_b": args.name_b,
                "config": config,
                "selected_counts": counts,
                "count": len(fused),
                "elapsed_seconds": time.time() - started,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.out}")
    print(f"Selected counts: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
