#!/usr/bin/env python3
"""Tune simple rejection thresholds on the dev split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from common import (
    audio_info,
    cer,
    extract_prediction_text,
    label_edit_similarity,
    label_similarity,
    normalize_text,
    read_jsonl,
    text_domain_score,
    write_json,
    write_simple_yaml,
)


def score(metrics: Dict, rr_weight: float, cer_weight: float, false_reject_weight: float) -> float:
    return (
        rr_weight * metrics["negative"]["rr"]
        - cer_weight * metrics["positive"]["cer"]
        - false_reject_weight * metrics["positive"]["false_reject_rate"]
    )


def build_features(manifest_rows: List[Dict], pred_rows: List[Dict], known_labels: List[str]) -> List[Dict]:
    predictions = {row["uid"]: row for row in pred_rows}
    features: List[Dict] = []
    for item in manifest_rows:
        pred = predictions.get(item["uid"], {})
        text = extract_prediction_text(pred)
        normalized = normalize_text(text)
        edits, chars = cer(item.get("label", ""), normalized)
        features.append(
            {
                "uid": item["uid"],
                "is_positive": item["is_positive"],
                "label": item.get("label", ""),
                "text": normalized,
                "chars": chars,
                "edits_if_kept": edits,
                "edits_if_rejected": chars,
                "text_len": len(normalized),
                "duration": audio_info(item["audio_cmd"])["duration"],
                "domain_score": text_domain_score(normalized),
                "label_similarity": label_similarity(normalized, known_labels),
                "label_edit_similarity": label_edit_similarity(normalized, known_labels),
            }
        )
    return features


def evaluate_candidate(features: List[Dict], config: Dict) -> Dict:
    total_edits = 0
    total_chars = 0
    pos_total = 0
    neg_total = 0
    neg_rejected = 0
    pos_rejected = 0
    examples = []
    for item in features:
        rejected = (
            (config.get("reject_empty", True) and not item["text"])
            or item["text_len"] < int(config["min_chars"])
            or (int(config.get("max_chars", 0)) > 0 and item["text_len"] > int(config["max_chars"]))
            or (
                float(config.get("max_duration_seconds", 0.0)) > 0
                and item["duration"] > float(config["max_duration_seconds"])
            )
            or item["domain_score"] < float(config["min_domain_score"])
            or item["label_similarity"] < float(config["min_label_similarity"])
            or item["label_edit_similarity"] < float(config.get("min_label_edit_similarity", 0.0))
        )
        if item["is_positive"]:
            pos_total += 1
            if rejected:
                pos_rejected += 1
                edits = item["edits_if_rejected"]
            else:
                edits = item["edits_if_kept"]
            total_edits += edits
            total_chars += item["chars"]
            if len(examples) < 20 and edits:
                examples.append(
                    {
                        "uid": item["uid"],
                        "label": item["label"],
                        "prediction": item["text"],
                        "rejected": rejected,
                        "edits": edits,
                        "chars": item["chars"],
                    }
                )
        else:
            neg_total += 1
            if rejected:
                neg_rejected += 1
    return {
        "positive": {
            "count": pos_total,
            "cer": total_edits / total_chars if total_chars else 0.0,
            "edits": total_edits,
            "chars": total_chars,
            "false_reject_rate": pos_rejected / pos_total if pos_total else 0.0,
            "false_reject_count": pos_rejected,
        },
        "negative": {
            "count": neg_total,
            "rr": neg_rejected / neg_total if neg_total else 0.0,
            "rejected": neg_rejected,
            "false_accept_rate": (neg_total - neg_rejected) / neg_total if neg_total else 0.0,
        },
        "overall": {
            "missing_predictions": 0,
        },
        "error_examples": examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=Path("data/splits/dev.jsonl"), type=Path)
    parser.add_argument("--train-manifest", default=Path("data/splits/train.jsonl"), type=Path)
    parser.add_argument("--dev-pred", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--metrics-out", type=Path)
    parser.add_argument("--rr-weight", type=float, default=1.0)
    parser.add_argument("--cer-weight", type=float, default=1.0)
    parser.add_argument("--false-reject-weight", type=float, default=1.5)
    parser.add_argument("--max-false-reject-rate", type=float)
    parser.add_argument("--min-rr", type=float)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    manifest_path = args.manifest if args.manifest.is_absolute() else project / args.manifest
    train_manifest_path = args.train_manifest if args.train_manifest.is_absolute() else project / args.train_manifest
    manifest_rows = read_jsonl(manifest_path)
    known_labels = [row.get("label", "") for row in read_jsonl(train_manifest_path) if row.get("is_positive")]
    pred_rows = read_jsonl(args.dev_pred)
    features = build_features(manifest_rows, pred_rows, known_labels)

    candidates: List[Dict] = []
    for min_chars in range(1, 5):
        for max_chars in [0, 10, 11, 12, 13, 14, 15, 16, 18, 20, 25, 30]:
            for max_duration_seconds in [0.0, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0]:
                for min_domain_score in [0.0, 0.25, 0.5]:
                    for min_label_similarity in [0.0, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.25, 0.30]:
                        for min_label_edit_similarity in [
                            0.0,
                            0.05,
                            0.10,
                            0.15,
                            0.20,
                            0.25,
                            0.30,
                            0.35,
                            0.40,
                        ]:
                            config = {
                                "reject_empty": True,
                                "min_chars": min_chars,
                                "max_chars": max_chars,
                                "max_duration_seconds": max_duration_seconds,
                                "min_domain_score": min_domain_score,
                                "min_label_similarity": min_label_similarity,
                                "min_label_edit_similarity": min_label_edit_similarity,
                                "known_labels_path": str(train_manifest_path.relative_to(project)),
                            }
                            metrics = evaluate_candidate(features, config)
                            if (
                                args.max_false_reject_rate is not None
                                and metrics["positive"]["false_reject_rate"] > args.max_false_reject_rate
                            ):
                                continue
                            if args.min_rr is not None and metrics["negative"]["rr"] < args.min_rr:
                                continue
                            candidates.append(
                                {
                                    "config": config,
                                    "metrics": metrics,
                                    "score": score(
                                        metrics,
                                        args.rr_weight,
                                        args.cer_weight,
                                        args.false_reject_weight,
                                    ),
                                }
                            )

    if not candidates:
        raise RuntimeError("No reject candidates matched the requested constraints.")
    best = max(candidates, key=lambda item: item["score"])
    header = "Generated by optimize_reject.py on the dev split. Apply unchanged to test."
    write_simple_yaml(args.out, best["config"], header=header)
    metrics_out = args.metrics_out or args.out.with_suffix(".metrics.json")
    write_json(metrics_out, {"best": best, "top_candidates": sorted(candidates, key=lambda item: item["score"], reverse=True)[:10]})
    print(f"Best config: {best['config']}")
    print(f"Dev CER={best['metrics']['positive']['cer']:.4f}, RR={best['metrics']['negative']['rr']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
