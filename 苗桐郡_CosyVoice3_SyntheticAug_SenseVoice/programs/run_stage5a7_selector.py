from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib

from evaluate_stage5a import normalize
from train_stage5a5_selector import (
    CANDIDATES,
    choose_from_predictions,
    load_partition,
    matrix,
    metric,
    oracle_choices,
    read_jsonl,
    write_jsonl,
)


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.out).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty Stage 5A.7 selector directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    selector_config = json.loads(Path(args.selector_config).read_text(encoding="utf-8"))
    lexicon_rows = read_jsonl(Path(args.lexicon_manifest))
    if any(row.get("contains_official_test_data") for row in lexicon_rows):
        raise ValueError("Official test data detected in selector lexicon")
    lexicon = sorted({
        normalize(row["text"])
        for row in lexicon_rows
        if row.get("split") == selector_config["training_lexicon_split"]
        and row.get("allowed_for_training", True)
    } - {""})

    samples = load_partition(args.partition, Path(args.separation_dir), Path(args.asr_dir), lexicon)
    if len(samples) != int(args.expected_count):
        raise ValueError(f"Expected {args.expected_count} samples, found {len(samples)}")
    x, _, _, feature_names = matrix(samples)
    if feature_names != selector_config["feature_names"]:
        raise ValueError("Frozen selector feature schema mismatch")
    model = joblib.load(args.selector_model)
    predicted = model.predict(x)
    best = selector_config["best_configuration"]
    choices = choose_from_predictions(
        samples,
        predicted,
        float(best["mixture_penalty"]),
        float(best["alternate_source_penalty"]),
    )

    rendered = []
    for sample_index, (sample, choice) in enumerate(zip(samples, choices)):
        predicted_cer = {
            candidate: float(predicted[sample_index * len(CANDIDATES) + candidate_index])
            for candidate_index, candidate in enumerate(CANDIDATES)
        }
        oracle = oracle_choices([sample])[0]
        rendered.append({
            **sample,
            "selector_candidate": choice,
            "deployed_candidate": choice,
            "predicted_candidate_cer": predicted_cer,
            "selector": sample["candidates"][choice],
            "oracle_candidate": oracle,
            "oracle": sample["candidates"][oracle],
            "selector_frozen_from": str(Path(args.selector_model).resolve()),
        })
    write_jsonl(output / f"{args.partition}_predictions.jsonl", rendered)

    groups = defaultdict(list)
    for row in rendered:
        groups[str(int(row["tir_db"]))].append(row)
    result = {
        "partition": args.partition,
        "count": len(rendered),
        "metrics": {
            "mixture": metric(samples, fixed="mixture"),
            "ecapa_selected": metric(samples, fixed="ecapa_selected"),
            "adaptive_selector": metric(samples, choices=choices),
            "oracle_three_candidate": metric(samples, choices=oracle_choices(samples)),
            "target_clean": metric(samples, fixed="target_clean"),
        },
        "candidate_counts": dict(Counter(choices)),
        "by_tir": {
            tir: {
                "count": len(rows),
                "ecapa_cer": sum(row["ecapa_selected"]["errors"] for row in rows) / sum(row["ecapa_selected"]["characters"] for row in rows),
                "adaptive_cer": sum(row["selector"]["errors"] for row in rows) / sum(row["selector"]["characters"] for row in rows),
                "mixture_count": sum(row["selector_candidate"] == "mixture" for row in rows),
            }
            for tir, rows in sorted(groups.items(), key=lambda item: float(item[0]))
        },
        "official_test_data_violations": 0,
    }
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen Stage 5A.5 selector on a new acoustic partition.")
    parser.add_argument("--separation-dir", required=True)
    parser.add_argument("--asr-dir", required=True)
    parser.add_argument("--lexicon-manifest", required=True)
    parser.add_argument("--selector-model", required=True)
    parser.add_argument("--selector-config", required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--expected-count", required=True, type=int)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
