#!/usr/bin/env python3
"""Build one reproducible JSON summary for the optimized competition versions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from asr_consensus import select_consensus_text, should_select_tse_text  # noqa: E402
from metrics_cer import _levenshtein_chars, normalize_text  # noqa: E402


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def index_rows(payload: dict) -> dict[str, dict]:
    return {str(row["id"]): row for row in payload["samples"]}


def fixed_gate_metrics(score_path: str, asr_path: str, threshold: float) -> dict:
    scores = load(score_path)["samples"]
    asr = index_rows(load(asr_path))
    positive_chars = positive_errors = positive_accepted = 0
    negative = false_accept = 0
    for row in scores:
        text = normalize_text(asr.get(str(row["id"]), {}).get("asr_text", ""))
        accepted = float(row["score"]) >= threshold and bool(text)
        if row["is_positive"]:
            label = normalize_text(row["label"])
            positive_chars += len(label)
            positive_errors += _levenshtein_chars(label, text if accepted else "")
            positive_accepted += int(accepted)
        else:
            negative += 1
            false_accept += int(accepted)
    n_positive = sum(bool(row["is_positive"]) for row in scores)
    cer = 100.0 * positive_errors / max(1, positive_chars)
    rr = 100.0 * (negative - false_accept) / max(1, negative)
    return {
        "threshold": threshold,
        "n_total": len(scores),
        "n_positive": n_positive,
        "n_rejection": negative,
        "cer": cer,
        "rejection_rate": rr,
        "positive_accept_rate": 100.0 * positive_accepted / max(1, n_positive),
        "proxy_score_40_40": 0.4 * (100.0 - cer) + 0.4 * rr,
        "false_accepts": false_accept,
    }


def corpus_asr_cer(path: str) -> float:
    rows = [row for row in load(path)["samples"] if row["is_positive"]]
    refs = sum(len(normalize_text(row["label"])) for row in rows)
    errors = sum(_levenshtein_chars(row["label"], row["asr_text"]) for row in rows)
    return 100.0 * errors / max(1, refs)


def selective_tse_metrics(routing_path: str, sensevoice_path: str) -> dict:
    rows = load(routing_path)["samples"]
    sensevoice = index_rows(load(sensevoice_path))
    refs = baseline_errors = selected_errors = all_tse_errors = 0
    selected_count = improved = degraded = 0
    for row in rows:
        label = normalize_text(row["label"])
        raw_candidates = [
            row["raw_vad_text"],
            row["raw_full_text"],
            sensevoice[row["id"]]["asr_text"],
        ]
        baseline, _, _ = select_consensus_text(raw_candidates)
        use_tse = (
            row["tse_similarity"] >= 0.15
            and should_select_tse_text(
                baseline,
                row["tse_text"],
                raw_candidates,
                row["similarity_gain"],
                min_similarity_gain=-0.05,
                max_text_distance_ratio=0.25,
                require_command_prior_not_worse=True,
            )
        )
        selected = row["tse_text"] if use_tse else baseline
        base_error = _levenshtein_chars(label, baseline)
        selected_error = _levenshtein_chars(label, selected)
        refs += len(label)
        baseline_errors += base_error
        selected_errors += selected_error
        all_tse_errors += _levenshtein_chars(label, row["tse_text"])
        selected_count += int(use_tse)
        improved += int(selected_error < base_error)
        degraded += int(selected_error > base_error)
    return {
        "n_positive_subset": len(rows),
        "baseline_consensus_cer": 100.0 * baseline_errors / max(1, refs),
        "selective_tse_cer": 100.0 * selected_errors / max(1, refs),
        "all_tse_cer": 100.0 * all_tse_errors / max(1, refs),
        "tse_selected": selected_count,
        "selected_improved": improved,
        "selected_degraded": degraded,
    }


def main() -> int:
    old_robust = load("output/datasetA_final_robust_cascade_no_prior_full_stats.json")
    stage2_train = load("output/tse_cnceleb_training_history.json")
    external_stage2 = load("output/tse_external_stage2_unseen32.json")
    external_stage3 = load("output/tse_external_stage3_unseen32.json")
    robust_train = load("output/tse_cnceleb_robust_training_history.json")
    external_eres_calibration = load("output/cnceleb2_300_eres_no_vad_15s_eval.json")
    external_fusion_calibration = load("output/cnceleb2_300_fusion_no_vad_15s_eval.json")
    v4_full_stats = load("output/datasetA_competition_v4_selective_tse_full_stats.json")
    result = {
        "evaluation_policy": {
            "datasetA_model_training": False,
            "datasetA_use": "development evaluation and operating-point comparison only",
            "tse_training": "CN-Celeb2 speaker-disjoint synthetic mixtures",
        },
        "previous_robust_pipeline": old_robust,
        "asr_only_cer": {
            "paraformer_energy_vad": corpus_asr_cer("output/datasetA_asr_cache_paraformer.json"),
            "paraformer_full_audio": corpus_asr_cer("output/datasetA_asr_cache_paraformer_no_vad.json"),
            "sensevoice_energy_vad": corpus_asr_cer("output/datasetA_asr_cache_sensevoice.json"),
            "label_free_three_path_consensus": corpus_asr_cer("output/datasetA_asr_cache_consensus3.json"),
        },
        "full_datasetA_versions": {
            "v0_external_calibrated_eres2netv2": fixed_gate_metrics(
                "output/frontend_full_eres_no_vad_15s_scores.json",
                "output/datasetA_asr_cache_consensus3.json",
                0.316535085439682,
            ),
            "release_external_calibrated_fusion": fixed_gate_metrics(
                "output/frontend_full_fusion_no_vad_15s_scores.json",
                "output/datasetA_asr_cache_consensus3.json",
                0.2642105449799689,
            ),
            "v1_cer_oriented": fixed_gate_metrics(
                "output/frontend_full_eres_no_vad_15s_scores.json",
                "output/datasetA_asr_cache_consensus3.json",
                0.22,
            ),
            "v2_balanced_eres2netv2": fixed_gate_metrics(
                "output/frontend_full_eres_no_vad_15s_scores.json",
                "output/datasetA_asr_cache_consensus3.json",
                0.2879392278498303,
            ),
            "v3_high_rr_eres_campplus": fixed_gate_metrics(
                "output/frontend_full_fusion_no_vad_15s_scores.json",
                "output/datasetA_asr_cache_consensus3.json",
                0.28653230974959004,
            ),
        },
        "v4_selective_tse_subset": selective_tse_metrics(
            "output/datasetA_tse_routing_eval_200.json",
            "output/datasetA_asr_cache_sensevoice.json",
        ),
        "v4_selective_tse_full": v4_full_stats,
        "tse_training": {
            "stage2_best_step": stage2_train["best_step"],
            "stage2_best_validation_si_snri_db": stage2_train["best_validation_si_snri_db"],
            "stage2_unseen": {
                key: external_stage2[key]
                for key in (
                    "mean_si_snri_db",
                    "median_si_snri_db",
                    "improved_ratio",
                    "identity_accuracy",
                    "mean_target_similarity_gain",
                )
            },
            "stage3_unseen": {
                key: external_stage3[key]
                for key in (
                    "mean_si_snri_db",
                    "median_si_snri_db",
                    "improved_ratio",
                    "identity_accuracy",
                    "mean_target_similarity_gain",
                )
            },
            "robust_validation": {
                "baseline_si_snri_db": robust_train["history"][0]["si_snri_db"],
                "best_si_snri_db": robust_train["best_validation_si_snri_db"],
                "best_step": robust_train["best_step"],
                "noise_files": robust_train["augmentation"]["noise_files"],
                "rir_files": robust_train["augmentation"]["rir_files"],
            },
        },
        "external_speaker_calibration": {
            "eres2netv2": external_eres_calibration["metrics"],
            "eres2netv2_campplus_fusion": external_fusion_calibration["metrics"],
        },
    }
    output = Path("output/competition_optimization_results.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
