"""Fusion decision logic for accept/reject.

The thresholds in this file are placeholders. They must be calibrated on a
speaker-disjoint validation set before final submission.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass
class Evidence:
    speaker_similarity: float = 0.0
    target_frame_ratio: float = 0.0
    non_target_frame_ratio: float = 0.0
    overlap_probability: float = 0.0
    asr_confidence: float = 0.0
    command_prior_score: float = 0.0
    enrollment_bad_quality: float = 0.0
    query_noise_penalty: float = 0.0
    no_speech: bool = False
    query_snr_db: float = 0.0
    wake_quality_score: float = 1.0


def should_reject_early(e: Evidence, thresholds: Dict[str, Any]) -> bool:
    """Reject obvious non-target or no-speech samples before ASR."""
    if e.no_speech:
        return True

    reject_low = float(thresholds.get("speaker_reject_low", 0.45))
    max_target_ratio = float(thresholds.get("max_target_ratio_for_reject", 0.08))

    return e.speaker_similarity <= reject_low and e.target_frame_ratio <= max_target_ratio


def should_accept_fast(e: Evidence, thresholds: Dict[str, Any]) -> bool:
    """Identify clear target speech samples that can skip TSE."""
    accept_high = float(thresholds.get("speaker_accept_high", 0.72))
    min_target_ratio = float(thresholds.get("min_target_ratio", 0.18))
    max_non_target = float(thresholds.get("max_non_target_ratio_for_accept", 0.70))

    return (
        e.speaker_similarity >= accept_high
        and e.target_frame_ratio >= min_target_ratio
        and e.non_target_frame_ratio <= max_non_target
    )


def should_use_tse(e: Evidence, thresholds: Dict[str, Any]) -> bool:
    """Decide whether the hard-case TSE path should be triggered."""
    overlap_th = float(thresholds.get("overlap_trigger_threshold", 0.35))
    low_snr_th = float(thresholds.get("low_snr_threshold_db", -2.0))
    target_low_snr_ratio = float(thresholds.get("target_low_snr_ratio", 0.12))
    accept_high = float(thresholds.get("speaker_accept_high", 0.72))
    reject_low = float(thresholds.get("speaker_reject_low", 0.45))

    middle_similarity = reject_low < e.speaker_similarity < accept_high
    low_snr_target = e.query_snr_db < low_snr_th and e.target_frame_ratio > target_low_snr_ratio
    has_overlap = e.overlap_probability > overlap_th
    has_target_and_non_target = e.target_frame_ratio > target_low_snr_ratio and e.non_target_frame_ratio > 0.25

    return bool(middle_similarity or low_snr_target or has_overlap or has_target_and_non_target)


def fusion_score(e: Evidence, thresholds: Dict[str, Any]) -> float:
    """Compute a linear fusion score."""
    weights = thresholds.get("fusion_weights", {}) or {}
    return (
        float(weights.get("speaker_similarity", 1.20)) * e.speaker_similarity
        + float(weights.get("target_frame_ratio", 0.90)) * e.target_frame_ratio
        + float(weights.get("non_target_frame_ratio", -0.70)) * e.non_target_frame_ratio
        + float(weights.get("asr_confidence", 0.40)) * e.asr_confidence
        + float(weights.get("command_prior_score", 0.25)) * e.command_prior_score
        + float(weights.get("enrollment_bad_quality", -0.30)) * e.enrollment_bad_quality
        + float(weights.get("query_noise_penalty", -0.20)) * e.query_noise_penalty
    )


def final_decision(e: Evidence, thresholds: Dict[str, Any]) -> Tuple[bool, float, str]:
    """Return (accept, score, reason)."""
    if should_reject_early(e, thresholds):
        return False, 0.0, "early_reject"

    score = fusion_score(e, thresholds)
    accept_threshold = float(thresholds.get("fusion_accept_threshold", 0.50))
    min_asr_conf = float(thresholds.get("min_asr_confidence", 0.35))

    # If speaker evidence is weak, do not let a fluent ASR result dominate RR.
    if e.speaker_similarity < float(thresholds.get("speaker_reject_low", 0.45)) and e.target_frame_ratio < float(thresholds.get("min_target_ratio", 0.18)):
        return False, score, "weak_speaker_and_target_evidence"

    # If ASR is very uncertain but target evidence is also weak, reject.
    if e.asr_confidence < min_asr_conf and e.target_frame_ratio < float(thresholds.get("min_target_ratio", 0.18)):
        return False, score, "low_asr_and_target_evidence"

    if score >= accept_threshold:
        return True, score, "accept_by_fusion"
    return False, score, "reject_by_fusion"
