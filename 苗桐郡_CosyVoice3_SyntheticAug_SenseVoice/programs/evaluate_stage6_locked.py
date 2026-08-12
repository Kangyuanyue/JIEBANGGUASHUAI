from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from evaluate_stage5a import edit_distance, normalize
from train_stage5a5_selector import CANDIDATES, audio_features, choose_from_predictions, feature_vector, nearest_lexicon


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_flat_yaml(path: Path) -> dict[str, Any]:
    values = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        if value.lower() in {"true", "false"}:
            values[key] = value.lower() == "true"
        else:
            try:
                values[key] = float(value) if any(char in value for char in ".eE") else int(value)
            except ValueError:
                values[key] = value.strip("\"'")
    return values


def label_edit_similarity(text: str, labels: list[str]) -> float:
    text = normalize(text)
    if not text:
        return 0.0
    return max((1.0 - edit_distance(text, label) / max(len(text), len(label), 1) for label in labels), default=0.0)


def rejected(text: str, config: dict[str, Any], known_labels: list[str]) -> bool:
    value = normalize(text)
    if config.get("reject_empty", True) and not value:
        return True
    if len(value) < int(config.get("min_chars", 1)):
        return True
    max_chars = int(config.get("max_chars", 0))
    if max_chars and len(value) > max_chars:
        return True
    threshold = float(config.get("min_label_edit_similarity", 0.0))
    return bool(threshold and label_edit_similarity(value, known_labels) < threshold)


def candidate_row(prediction: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw_text": prediction["raw_text"],
        "hypothesis": prediction["text"],
        "errors": prediction["errors"],
        "characters": prediction["characters"],
        "cer": prediction["cer"],
    }


def summarize(rows: list[dict[str, Any]], field: str, apply_reject: bool) -> dict[str, Any]:
    errors = characters = pos_count = neg_count = neg_rejected = pos_rejected = 0
    for row in rows:
        value = row[field]
        is_rejected = bool(row[f"{field}_rejected"]) if apply_reject else not normalize(value["hypothesis"])
        if row["is_positive"]:
            pos_count += 1
            pos_rejected += int(is_rejected)
            reference = normalize(row["reference"])
            hypothesis = "" if is_rejected else normalize(value["hypothesis"])
            errors += edit_distance(reference, hypothesis)
            characters += len(reference)
        else:
            neg_count += 1
            neg_rejected += int(is_rejected)
    return {
        "positive": {
            "count": pos_count,
            "cer": errors / characters if characters else 0.0,
            "errors": errors,
            "characters": characters,
            "false_reject_rate": pos_rejected / pos_count if pos_count else 0.0,
            "false_reject_count": pos_rejected,
        },
        "negative": {
            "count": neg_count,
            "rr": neg_rejected / neg_count if neg_count else 0.0,
            "rejected": neg_rejected,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.out).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty Stage 6 evaluation directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    separation = {row["uid"]: row for row in read_jsonl(Path(args.separation_dir) / "separation_results.jsonl")}
    predictions = {
        name: {row["uid"]: row for row in read_jsonl(Path(args.asr_dir) / f"{name}_pred.jsonl")}
        for name in CANDIDATES
    }
    if any(set(rows) != set(separation) for rows in predictions.values()):
        raise ValueError("Stage 6 UID mismatch between separation and ASR")

    selector_config = json.loads(Path(args.selector_config).read_text(encoding="utf-8"))
    lexicon_rows = read_jsonl(Path(args.lexicon_manifest))
    lexicon = sorted({normalize(row["text"]) for row in lexicon_rows if row.get("split") == "train"} - {""})
    reject_config = read_flat_yaml(Path(args.reject_config))
    known_rows = read_jsonl(Path(args.known_labels_manifest))
    known_labels = sorted({normalize(row.get("label", "")) for row in known_rows if row.get("is_positive")} - {""})

    samples = []
    for uid in sorted(separation):
        sep = separation[uid]
        audio_paths = {"mixture": sep["mixture_path"], "source0": sep["source_paths"][0], "source1": sep["source_paths"][1]}
        candidate_rows = {name: candidate_row(predictions[name][uid]) for name in CANDIDATES}
        sample = {
            "uid": uid,
            "reference": sep["label"],
            "is_positive": sep["is_positive"],
            "speaker_scores": sep["speaker_scores"],
            "mixture_speaker_score": sep["mixture_speaker_score"],
            "ecapa_selected_source": sep["selected_source"],
            "audio_paths": audio_paths,
            "audio_stats": {name: audio_features(path) for name, path in audio_paths.items()},
            "lexical": {name: nearest_lexicon(candidate_rows[name]["hypothesis"], lexicon) for name in CANDIDATES},
            "candidates": candidate_rows,
            "reference_path": sep["reference_path"],
        }
        samples.append(sample)

    matrix_rows = []
    feature_names = None
    for sample in samples:
        for candidate in CANDIDATES:
            values, names = feature_vector(sample, candidate)
            feature_names = feature_names or names
            matrix_rows.append(values)
    if feature_names != selector_config["feature_names"]:
        raise ValueError("Stage 6 selector feature schema mismatch")
    model = joblib.load(args.selector_model)
    prediction_values = model.predict(np.asarray(matrix_rows, dtype=np.float32))
    best = selector_config["best_configuration"]
    choices = choose_from_predictions(samples, prediction_values, float(best["mixture_penalty"]), float(best["alternate_source_penalty"]))

    rendered = []
    for sample, choice in zip(samples, choices):
        ecapa_choice = f"source{sample['ecapa_selected_source']}"
        fields = {
            "direct": sample["candidates"]["mixture"],
            "ecapa": sample["candidates"][ecapa_choice],
            "selector": sample["candidates"][choice],
        }
        rendered.append({
            **sample,
            "ecapa_candidate": ecapa_choice,
            "selector_candidate": choice,
            **fields,
            **{f"{name}_rejected": rejected(value["hypothesis"], reject_config, known_labels) for name, value in fields.items()},
        })
    with (output / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rendered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = {
        "dataset": args.dataset,
        "count": len(rendered),
        "fixed_configuration": {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "selector_model": str(Path(args.selector_model).resolve()),
            "reject_config": str(Path(args.reject_config).resolve()),
            "vad": False,
            "use_itn": False,
        },
        "strategies": {
            "direct_no_reject": summarize(rendered, "direct", False),
            "direct_balanced_reject": summarize(rendered, "direct", True),
            "ecapa_balanced_reject": summarize(rendered, "ecapa", True),
            "selector_balanced_reject": summarize(rendered, "selector", True),
        },
        "selector_candidate_counts": dict(Counter(choices)),
        "evaluation_only": True,
        "test_data_used_for_training_or_tuning": False,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the fully locked Stage 6 selector and BalancedReject.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--separation-dir", required=True)
    parser.add_argument("--asr-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--selector-model", required=True)
    parser.add_argument("--selector-config", required=True)
    parser.add_argument("--lexicon-manifest", required=True)
    parser.add_argument("--reject-config", required=True)
    parser.add_argument("--known-labels-manifest", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
