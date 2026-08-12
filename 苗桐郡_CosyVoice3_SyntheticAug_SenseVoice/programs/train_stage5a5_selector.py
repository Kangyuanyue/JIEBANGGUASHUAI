from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import soundfile as sf
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor

from evaluate_stage5a import edit_distance, normalize


CANDIDATES = ("mixture", "source0", "source1")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def index_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {row["uid"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate UID detected")
    return result


def nearest_lexicon(text: str, lexicon: list[str]) -> tuple[float, float]:
    candidate = normalize(text)
    if not candidate:
        return 1.0, 1.0
    best_distance = None
    best_length_delta = None
    for reference in lexicon:
        distance = edit_distance(reference, candidate) / max(len(reference), len(candidate), 1)
        length_delta = abs(len(reference) - len(candidate)) / max(len(reference), len(candidate), 1)
        if best_distance is None or (distance, length_delta) < (best_distance, best_length_delta):
            best_distance, best_length_delta = distance, length_delta
    return float(best_distance), float(best_length_delta)


def audio_features(path: str) -> tuple[float, float, float]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    rms = float(np.sqrt(np.mean(np.square(mono)) + 1e-12))
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    duration = len(mono) / sample_rate if sample_rate else 0.0
    return rms, peak, duration


def load_partition(name: str, separation_dir: Path, asr_dir: Path, lexicon: list[str]) -> list[dict[str, Any]]:
    separation = index_rows(read_jsonl(separation_dir / "separation_results.jsonl"))
    predictions = {
        candidate: index_rows(read_jsonl(asr_dir / f"{candidate}_pred.jsonl"))
        for candidate in (*CANDIDATES, "selected", "target_clean")
    }
    uid_set = set(separation)
    if any(set(rows) != uid_set for rows in predictions.values()):
        raise ValueError(f"UID mismatch in partition {name}")
    if any(row.get("contains_official_test_data") for row in separation.values()):
        raise ValueError(f"Official test data detected in {name}")

    samples = []
    for uid in sorted(uid_set):
        sep = separation[uid]
        audio_paths = {
            "mixture": sep["mixture_path"],
            "source0": sep["source_paths"][0],
            "source1": sep["source_paths"][1],
        }
        audio_stats = {candidate: audio_features(path) for candidate, path in audio_paths.items()}
        candidate_rows = {}
        lexical = {}
        for candidate in CANDIDATES:
            pred = predictions[candidate][uid]
            lexical[candidate] = nearest_lexicon(pred["hypothesis"], lexicon)
            candidate_rows[candidate] = {
                "raw_text": pred["raw_text"], "hypothesis": pred["hypothesis"],
                "errors": pred["errors"], "characters": pred["characters"], "cer": pred["cer"],
            }
        selected = predictions["selected"][uid]
        clean = predictions["target_clean"][uid]
        samples.append({
            "uid": uid, "partition": name, "reference": predictions["mixture"][uid]["reference"],
            "tir_db": sep["tir_db"], "snr_db": sep["snr_db"], "speaker": sep["speaker"],
            "speaker_scores": sep["speaker_scores"], "mixture_speaker_score": sep["mixture_speaker_score"],
            "ecapa_selected_source": sep["selected_source"], "ecapa_selection_correct": sep["selection_correct"],
            "lexical": lexical, "audio_stats": audio_stats, "audio_paths": audio_paths,
            "reference_path": sep["reference_path"], "target_clean_path": sep["target_clean_path"],
            "candidates": candidate_rows,
            "ecapa_selected": {"raw_text": selected["raw_text"], "hypothesis": selected["hypothesis"],
                               "errors": selected["errors"], "characters": selected["characters"], "cer": selected["cer"]},
            "target_clean": {"raw_text": clean["raw_text"], "hypothesis": clean["hypothesis"],
                             "errors": clean["errors"], "characters": clean["characters"], "cer": clean["cer"]},
        })
    return samples


def feature_vector(sample: dict[str, Any], candidate: str) -> tuple[list[float], list[str]]:
    scores = sample["speaker_scores"]
    source_max = max(scores)
    source_margin = abs(scores[0] - scores[1])
    if candidate == "mixture":
        candidate_score = sample["mixture_speaker_score"]
        other_score = source_max
        is_ecapa = 0.0
    else:
        index = int(candidate[-1])
        candidate_score = scores[index]
        other_score = scores[1 - index]
        is_ecapa = float(index == sample["ecapa_selected_source"])
    rms, peak, duration = sample["audio_stats"][candidate]
    mixture_rms = sample["audio_stats"]["mixture"][0]
    hypothesis = sample["candidates"][candidate]["hypothesis"]
    lexical_distance, lexical_length_delta = sample["lexical"][candidate]
    source_lexical = [sample["lexical"][f"source{i}"][0] for i in (0, 1)]
    values = [
        float(candidate == "mixture"), float(candidate == "source0"), float(candidate == "source1"),
        candidate_score, other_score, source_max, min(scores), source_margin,
        sample["mixture_speaker_score"], candidate_score - other_score, is_ecapa,
        lexical_distance, lexical_length_delta, min(source_lexical), abs(source_lexical[0] - source_lexical[1]),
        float(len(hypothesis)), float(not hypothesis), float(any(char.isdigit() for char in hypothesis)),
        float(any("a" <= char.lower() <= "z" for char in hypothesis)),
        rms, peak, duration, rms / max(mixture_rms, 1e-8),
    ]
    names = [
        "is_mixture", "is_source0", "is_source1", "candidate_speaker_score", "other_source_score",
        "source_score_max", "source_score_min", "source_score_margin", "mixture_speaker_score",
        "candidate_score_advantage", "is_ecapa_selected", "lexical_distance", "lexical_length_delta",
        "best_source_lexical_distance", "source_lexical_margin", "hypothesis_length", "hypothesis_empty",
        "contains_digit", "contains_latin", "audio_rms", "audio_peak", "audio_duration", "rms_over_mixture",
    ]
    return values, names


def matrix(samples: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, list[tuple[int, str]], list[str]]:
    x, y, keys = [], [], []
    feature_names = None
    for sample_index, sample in enumerate(samples):
        for candidate in CANDIDATES:
            features, names = feature_vector(sample, candidate)
            feature_names = feature_names or names
            x.append(features)
            y.append(sample["candidates"][candidate]["cer"])
            keys.append((sample_index, candidate))
    return np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32), keys, feature_names or []


def choose_from_predictions(samples: list[dict[str, Any]], predictions: np.ndarray,
                            mixture_penalty: float, alternate_penalty: float) -> list[str]:
    choices = []
    for sample_index, sample in enumerate(samples):
        values = {}
        for candidate_index, candidate in enumerate(CANDIDATES):
            value = float(predictions[sample_index * len(CANDIDATES) + candidate_index])
            if candidate == "mixture":
                value += mixture_penalty
            elif candidate != f"source{sample['ecapa_selected_source']}":
                value += alternate_penalty
            values[candidate] = value
        choices.append(min(CANDIDATES, key=lambda candidate: (
            values[candidate], candidate != f"source{sample['ecapa_selected_source']}", candidate,
        )))
    return choices


def metric(samples: list[dict[str, Any]], choices: list[str] | None = None,
           fixed: str | None = None) -> dict[str, Any]:
    errors = characters = 0
    for index, sample in enumerate(samples):
        if choices is not None:
            value = sample["candidates"][choices[index]]
        elif fixed in CANDIDATES:
            value = sample["candidates"][fixed]
        else:
            value = sample[fixed]
        errors += value["errors"]
        characters += value["characters"]
    return {"count": len(samples), "errors": errors, "characters": characters,
            "cer": errors / characters if characters else 0.0}


def oracle_choices(samples: list[dict[str, Any]]) -> list[str]:
    return [min(CANDIDATES, key=lambda name: (sample["candidates"][name]["errors"], name)) for sample in samples]


def model_specs(seed: int):
    for family in ("random_forest", "extra_trees"):
        for max_depth, min_leaf, max_features in itertools.product((4, 8, None), (3, 8, 16), ("sqrt", 0.75, 1.0)):
            cls = RandomForestRegressor if family == "random_forest" else ExtraTreesRegressor
            yield family, {"max_depth": max_depth, "min_samples_leaf": min_leaf, "max_features": max_features}, cls(
                n_estimators=240, max_depth=max_depth, min_samples_leaf=min_leaf,
                max_features=max_features, random_state=seed, n_jobs=-1,
            )
    for max_leaf_nodes, l2 in itertools.product((7, 15, 31), (0.0, 0.1, 1.0)):
        params = {"max_leaf_nodes": max_leaf_nodes, "l2_regularization": l2}
        yield "hist_gradient_boosting", params, HistGradientBoostingRegressor(
            max_iter=180, learning_rate=0.05, max_leaf_nodes=max_leaf_nodes,
            l2_regularization=l2, random_state=seed,
        )


def partition_metrics(samples: list[dict[str, Any]], choices: list[str]) -> dict[str, Any]:
    return {
        "mixture": metric(samples, fixed="mixture"),
        "ecapa_selected": metric(samples, fixed="ecapa_selected"),
        "adaptive_selector": metric(samples, choices=choices),
        "oracle_three_candidate": metric(samples, choices=oracle_choices(samples)),
        "target_clean": metric(samples, fixed="target_clean"),
        "candidate_counts": dict(Counter(choices)),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    project = Path(args.project_root).resolve()
    output = Path(args.out).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty Stage 5A.5 selector directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    lexicon_rows = read_jsonl(project / config["training_lexicon_manifest"])
    if any(row.get("contains_official_test_data") for row in lexicon_rows):
        raise ValueError("Official test data detected in lexicon source")
    lexicon = sorted({normalize(row["text"]) for row in lexicon_rows
                      if row.get("split") == config["training_lexicon_split"] and row.get("allowed_for_training", True)} - {""})
    partitions = {
        "train": load_partition("train", Path(args.train_separation), Path(args.train_asr), lexicon),
        "dev": load_partition("dev", Path(args.dev_separation), Path(args.dev_asr), lexicon),
        "internal_test": load_partition("internal_test", Path(args.test_separation), Path(args.test_asr), lexicon),
    }
    expected = {"train": 484, "dev": 108, "internal_test": 108}
    if {name: len(rows) for name, rows in partitions.items()} != expected:
        raise ValueError(f"Unexpected partition sizes: { {name: len(rows) for name, rows in partitions.items()} }")

    matrices = {name: matrix(rows) for name, rows in partitions.items()}
    x_train, y_train, _, feature_names = matrices["train"]
    x_dev, _, _, _ = matrices["dev"]
    x_test, _, _, _ = matrices["internal_test"]
    leaderboard = []
    fitted_models = []
    for family, model_params, model in model_specs(int(config["seed"])):
        model.fit(x_train, y_train)
        dev_prediction = model.predict(x_dev)
        for mixture_penalty, alternate_penalty in itertools.product(
            config["mixture_penalties"], config["alternate_source_penalties"]
        ):
            choices = choose_from_predictions(partitions["dev"], dev_prediction, float(mixture_penalty), float(alternate_penalty))
            dev_metric = metric(partitions["dev"], choices=choices)
            changes = sum(choice != f"source{sample['ecapa_selected_source']}"
                          for choice, sample in zip(choices, partitions["dev"]))
            leaderboard.append({
                "family": family, "model_params": model_params,
                "mixture_penalty": float(mixture_penalty), "alternate_source_penalty": float(alternate_penalty),
                "dev_cer": dev_metric["cer"], "dev_errors": dev_metric["errors"], "dev_changes": changes,
            })
        fitted_models.append((family, model_params, model))
    best = min(leaderboard, key=lambda row: (
        row["dev_cer"], row["dev_changes"], row["family"],
        json.dumps(row["model_params"], sort_keys=True), row["mixture_penalty"], row["alternate_source_penalty"],
    ))
    best_model = next(model for family, params, model in fitted_models
                      if family == best["family"] and params == best["model_params"])

    model_predictions = {
        "train": best_model.predict(x_train), "dev": best_model.predict(x_dev),
        "internal_test": best_model.predict(x_test),
    }
    choices = {
        name: choose_from_predictions(rows, model_predictions[name], best["mixture_penalty"], best["alternate_source_penalty"])
        for name, rows in partitions.items()
    }
    metrics = {name: partition_metrics(rows, choices[name]) for name, rows in partitions.items()}
    test_improvement = metrics["internal_test"]["ecapa_selected"]["cer"] - metrics["internal_test"]["adaptive_selector"]["cer"]
    adopted = test_improvement >= float(config["minimum_internal_test_improvement"])

    output_rows = {}
    for name, rows in partitions.items():
        rendered = []
        prediction_values = model_predictions[name]
        for sample_index, (sample, choice) in enumerate(zip(rows, choices[name])):
            predicted_cer = {
                candidate: float(prediction_values[sample_index * len(CANDIDATES) + candidate_index])
                for candidate_index, candidate in enumerate(CANDIDATES)
            }
            oracle = oracle_choices([sample])[0]
            rendered.append({**sample, "selector_candidate": choice,
                             "deployed_candidate": choice if adopted else f"source{sample['ecapa_selected_source']}",
                             "predicted_candidate_cer": predicted_cer,
                             "selector": sample["candidates"][choice], "oracle_candidate": oracle,
                             "oracle": sample["candidates"][oracle]})
        output_rows[name] = rendered
        write_jsonl(output / f"{name}_predictions.jsonl", rendered)

    by_tir = defaultdict(list)
    for index, sample in enumerate(partitions["internal_test"]):
        by_tir[str(int(sample["tir_db"]))].append((sample, choices["internal_test"][index]))
    tir_metrics = {}
    for tir, pairs in sorted(by_tir.items(), key=lambda item: float(item[0])):
        rows = [pair[0] for pair in pairs]
        selected = [pair[1] for pair in pairs]
        tir_metrics[tir] = {
            "count": len(rows), "mixture_cer": metric(rows, fixed="mixture")["cer"],
            "ecapa_selected_cer": metric(rows, fixed="ecapa_selected")["cer"],
            "adaptive_selector_cer": metric(rows, choices=selected)["cer"],
            "oracle_cer": metric(rows, choices=oracle_choices(rows))["cer"],
            "target_clean_cer": metric(rows, fixed="target_clean")["cer"],
            "candidate_counts": dict(Counter(selected)),
        }

    importance = None
    if hasattr(best_model, "feature_importances_"):
        importance = sorted(({"feature": name, "importance": float(value)}
                             for name, value in zip(feature_names, best_model.feature_importances_)),
                            key=lambda row: row["importance"], reverse=True)
    result = {
        "method": "supervised candidate CER regressor with adaptive mixture bypass",
        "partition_policy": "train 484 fits model; dev 108 selects hyperparameters; internal-test 108 is final evaluation",
        "partition_counts": expected, "training_lexicon_count": len(lexicon),
        "feature_names": feature_names, "best_configuration": best,
        "leaderboard_count": len(leaderboard), "metrics": metrics, "internal_test_by_tir": tir_metrics,
        "internal_test_improvement": test_improvement,
        "required_improvement": float(config["minimum_internal_test_improvement"]),
        "selector_adopted": adopted,
        "selected_strategy": "adaptive_selector" if adopted else "ecapa_selected",
        "feature_importance": importance,
        "official_test_data_violations": 0,
    }
    joblib.dump(best_model, output / "selector_model.joblib")
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "leaderboard.json").write_text(json.dumps(sorted(leaderboard, key=lambda row: row["dev_cer"]), ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "selector_config.json").write_text(json.dumps({
        "best_configuration": best, "feature_names": feature_names,
        "training_lexicon_manifest": config["training_lexicon_manifest"],
        "training_lexicon_split": config["training_lexicon_split"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate Stage 5A.5 adaptive ASR candidate selector.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-separation", required=True)
    parser.add_argument("--train-asr", required=True)
    parser.add_argument("--dev-separation", required=True)
    parser.add_argument("--dev-asr", required=True)
    parser.add_argument("--test-separation", required=True)
    parser.add_argument("--test-asr", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--project-root", default=".")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"best_configuration": result["best_configuration"],
                      "metrics": result["metrics"], "internal_test_improvement": result["internal_test_improvement"],
                      "selector_adopted": result["selector_adopted"]}, ensure_ascii=False, indent=2))
