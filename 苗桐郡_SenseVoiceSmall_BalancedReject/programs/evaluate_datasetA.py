#!/usr/bin/env python3
"""Evaluate datasetA predictions with CER for pos and RR for neg."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

from common import audio_info, cer, extract_prediction_text, load_simple_yaml, project_root, read_jsonl, should_reject, write_json


def load_known_labels(reject_config: Dict | None) -> List[str] | None:
    if not reject_config or not reject_config.get("known_labels_path"):
        return None
    path = Path(str(reject_config["known_labels_path"]))
    if not path.is_absolute():
        path = project_root() / path
    return [row.get("label", "") for row in read_jsonl(path) if row.get("is_positive")]


def should_reject_from_prediction_features(pred: Dict, reject_config: Dict | None) -> bool:
    if not reject_config:
        return False
    candidates = pred.get("fusion_candidates") or []
    if not isinstance(candidates, list) or not candidates:
        return False
    lengths = [int(item.get("length", 0)) for item in candidates if isinstance(item, dict)]
    scores = [float(item.get("score", 0.0)) for item in candidates if isinstance(item, dict)]
    edit_sims = [float(item.get("edit_similarity", 0.0)) for item in candidates if isinstance(item, dict)]
    if not lengths:
        return False

    max_candidate_chars = int(reject_config.get("max_candidate_chars", 0))
    max_both_candidate_chars = int(reject_config.get("max_both_candidate_chars", 0))
    min_max_candidate_edit_similarity = float(reject_config.get("min_max_candidate_edit_similarity", 0.0))

    if max_candidate_chars > 0 and max(lengths) > max_candidate_chars:
        return True
    if max_both_candidate_chars > 0 and min(lengths) > max_both_candidate_chars:
        return True
    if edit_sims and min_max_candidate_edit_similarity > 0:
        if max(edit_sims) < min_max_candidate_edit_similarity:
            return True
    if scores and "min_max_fusion_score" in reject_config:
        if max(scores) < float(reject_config["min_max_fusion_score"]):
            return True
    return False


def evaluate(manifest_rows: List[Dict], pred_rows: List[Dict], reject_config: Dict | None = None) -> Dict:
    predictions = {row["uid"]: row for row in pred_rows}
    known_labels = load_known_labels(reject_config)
    total_edits = 0
    total_chars = 0
    pos_total = 0
    neg_total = 0
    neg_rejected = 0
    pos_rejected = 0
    missing_predictions = 0
    examples = []

    for item in manifest_rows:
        uid = item["uid"]
        pred = predictions.get(uid)
        if pred is None:
            missing_predictions += 1
            pred = {"uid": uid, "text": "", "missing": True}
        text = extract_prediction_text(pred)
        duration = None
        if reject_config and float(reject_config.get("max_duration_seconds", 0.0)) > 0:
            duration = audio_info(item["audio_cmd"])["duration"]
        rejected = should_reject(text, reject_config, known_labels, audio_duration=duration)
        rejected = rejected or should_reject_from_prediction_features(pred, reject_config)
        if item["is_positive"]:
            pos_total += 1
            if rejected:
                pos_rejected += 1
            edits, chars = cer(item.get("label", ""), "" if rejected else text)
            total_edits += edits
            total_chars += chars
            if len(examples) < 20 and edits:
                examples.append(
                    {
                        "uid": uid,
                        "label": item.get("label", ""),
                        "prediction": text,
                        "rejected": rejected,
                        "edits": edits,
                        "chars": chars,
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
            "missing_predictions": missing_predictions,
            "prediction_count": len(pred_rows),
        },
        "error_examples": examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--pred", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--reject-config", type=Path)
    args = parser.parse_args()

    reject_config = load_simple_yaml(args.reject_config) if args.reject_config else None
    metrics = evaluate(read_jsonl(args.manifest), read_jsonl(args.pred), reject_config)
    metrics["inputs"] = {
        "manifest": str(args.manifest.resolve()),
        "pred": str(args.pred.resolve()),
        "reject_config": str(args.reject_config.resolve()) if args.reject_config else None,
    }
    write_json(args.out, metrics)
    print(f"CER={metrics['positive']['cer']:.4f}")
    print(f"RR={metrics['negative']['rr']:.4f}")
    print(f"False reject rate={metrics['positive']['false_reject_rate']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
