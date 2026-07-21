#!/usr/bin/env python3
"""Export a datasetA run to the official submission JSON shape."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from common import (
    audio_info,
    cer,
    extract_prediction_text,
    load_simple_yaml,
    project_root,
    read_json,
    read_jsonl,
    repair_mojibake,
    should_reject,
    write_json,
)
from evaluate_datasetA import load_known_labels, should_reject_from_prediction_features


def resolve_path(path: Optional[str | Path], base: Path) -> Optional[Path]:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return base / candidate


def load_duration_ms(pred_path: Path, explicit_duration_ms: Optional[float] = None) -> float:
    if explicit_duration_ms is not None:
        return float(explicit_duration_ms)
    meta_path = pred_path.with_suffix(".meta.json")
    if not meta_path.exists():
        return 0.0
    meta = read_json(meta_path, default={})
    elapsed = meta.get("elapsed_seconds")
    try:
        return float(elapsed) * 1000.0
    except (TypeError, ValueError):
        return 0.0


def choose_submission_ids(rows: Iterable[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[str]]:
    """Preserve official ids when present; use uid only when no original id exists."""
    rows = list(rows)
    warnings: List[str] = []
    id_map: Dict[str, Any] = {}
    fallback_count = 0
    for row in rows:
        original_id = row.get("id", row.get("source_id"))
        if original_id is None:
            original_id = row["uid"]
            fallback_count += 1
        id_map[row["uid"]] = original_id
    if fallback_count:
        warnings.append(f"{fallback_count} rows have no original id/source_id; internal uid is used for those rows.")
    return id_map, warnings


def is_rejected(item: Dict[str, Any], pred: Dict[str, Any], reject_config: Optional[Dict[str, Any]]) -> bool:
    text = extract_prediction_text(pred)
    duration = None
    if reject_config and float(reject_config.get("max_duration_seconds", 0.0)) > 0:
        duration = audio_info(item["audio_cmd"])["duration"]
    known_labels = load_known_labels(reject_config)
    rejected = should_reject(text, reject_config, known_labels, audio_duration=duration)
    return rejected or should_reject_from_prediction_features(pred, reject_config)


def build_submission(
    manifest_rows: List[Dict[str, Any]],
    pred_rows: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    pred_path: Path,
    reject_config: Optional[Dict[str, Any]] = None,
    duration_ms: Optional[float] = None,
) -> Tuple[Dict[str, Any], List[str]]:
    predictions = {row["uid"]: row for row in pred_rows}
    id_map, warnings = choose_submission_ids(manifest_rows)
    result_rows: List[Dict[str, Any]] = []
    positive_edits = 0
    positive_chars = 0
    neg_total = 0
    neg_rejected = 0

    for item in manifest_rows:
        uid = item["uid"]
        pred = predictions.get(uid, {"uid": uid, "text": "", "missing": True})
        raw_text = repair_mojibake(extract_prediction_text(pred))
        rejected = is_rejected(item, pred, reject_config)

        if item.get("is_positive"):
            content = "" if rejected else raw_text
            label = repair_mojibake(str(item.get("label", "")))
            edits, chars = cer(label, content)
            positive_edits += edits
            positive_chars += chars
            row_cer = edits / chars if chars else (0.0 if not content else 1.0)
        else:
            content = "" if rejected else raw_text
            label = ""
            row_cer = 0.0
            neg_total += 1
            if not content:
                neg_rejected += 1

        result_rows.append(
            {
                "id": id_map[uid],
                "content": content,
                "label": label,
                "cer": row_cer,
            }
        )

    computed_avg_cer = positive_edits / positive_chars if positive_chars else 0.0
    computed_avg_rr = neg_rejected / neg_total if neg_total else 0.0
    expected_avg_cer = float(metrics.get("positive", {}).get("cer", computed_avg_cer))
    expected_avg_rr = float(metrics.get("negative", {}).get("rr", computed_avg_rr))
    if not math.isclose(computed_avg_cer, expected_avg_cer, rel_tol=0.0, abs_tol=1e-12):
        warnings.append(
            f"computed avg_cer={computed_avg_cer:.12f} differs from metrics avg_cer={expected_avg_cer:.12f}."
        )
    if not math.isclose(computed_avg_rr, expected_avg_rr, rel_tol=0.0, abs_tol=1e-12):
        warnings.append(
            f"computed avg_rr={computed_avg_rr:.12f} differs from metrics avg_rr={expected_avg_rr:.12f}."
        )

    duration_value = load_duration_ms(pred_path, duration_ms)
    if duration_value <= 0:
        warnings.append("duration is 0 because this composed run has no pred.meta.json timing file.")

    submission = {
        "result": {
            "results": result_rows,
            "avg_cer": expected_avg_cer,
            "avg_rr": expected_avg_rr,
            "duration": duration_value,
        }
    }
    return submission, warnings


def validate_submission(submission: Dict[str, Any], expected_count: int) -> None:
    result = submission.get("result")
    if not isinstance(result, dict):
        raise ValueError("submission.result must be an object")
    rows = result.get("results")
    if not isinstance(rows, list):
        raise ValueError("submission.result.results must be a list")
    if len(rows) != expected_count:
        raise ValueError(f"results count mismatch: {len(rows)} != {expected_count}")
    for index, row in enumerate(rows):
        for key in ["id", "content", "label", "cer"]:
            if key not in row:
                raise ValueError(f"missing key {key!r} in results[{index}]")
    for key in ["avg_cer", "avg_rr", "duration"]:
        if key not in result:
            raise ValueError(f"missing key result.{key}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="datasetA manifest for this submission")
    parser.add_argument("--pred", type=Path, help="Prediction JSONL")
    parser.add_argument("--metrics", required=True, type=Path, help="Run metrics.json")
    parser.add_argument("--reject-config", type=Path, help="Optional reject YAML; defaults to metrics inputs")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--duration-ms", type=float)
    args = parser.parse_args()

    project = project_root()
    metrics = read_json(args.metrics)
    metrics_inputs = metrics.get("inputs", {})
    manifest_path = args.manifest or resolve_path(metrics_inputs.get("manifest"), project)
    pred_path = args.pred or resolve_path(metrics_inputs.get("pred"), project)
    reject_config_path = args.reject_config or resolve_path(metrics_inputs.get("reject_config"), project)
    if manifest_path is None:
        raise ValueError("--manifest is required when metrics.inputs.manifest is unavailable")
    if pred_path is None:
        raise ValueError("--pred is required when metrics.inputs.pred is unavailable")

    reject_config = load_simple_yaml(reject_config_path) if reject_config_path else None
    manifest_rows = read_jsonl(manifest_path)
    pred_rows = read_jsonl(pred_path)
    submission, warnings = build_submission(
        manifest_rows=manifest_rows,
        pred_rows=pred_rows,
        metrics=metrics,
        pred_path=pred_path,
        reject_config=reject_config,
        duration_ms=args.duration_ms,
    )
    validate_submission(submission, expected_count=len(manifest_rows))
    write_json(args.out, submission)
    print(f"Wrote {args.out}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
