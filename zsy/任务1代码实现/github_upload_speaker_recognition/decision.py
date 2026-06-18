"""Accept/reject fusion logic for target-speaker command recognition."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DecisionEvidence:
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


@dataclass
class DecisionResult:
    accepted: bool
    score: float
    reason: str
    use_tse: bool = False


def should_reject_early(e: DecisionEvidence, cfg) -> bool:
    if e.no_speech:
        return True
    return (
        e.speaker_similarity <= cfg.speaker_reject_low
        and e.target_frame_ratio <= cfg.max_target_ratio_for_reject
    )


def should_use_tse(e: DecisionEvidence, cfg) -> bool:
    middle_similarity = cfg.speaker_reject_low < e.speaker_similarity < cfg.speaker_accept_high
    low_snr_target = e.query_snr_db < cfg.low_snr_threshold_db and e.target_frame_ratio > cfg.target_low_snr_ratio
    has_overlap = e.overlap_probability > cfg.overlap_trigger_threshold
    target_and_other = e.target_frame_ratio > cfg.target_low_snr_ratio and e.non_target_frame_ratio > 0.25
    return bool(middle_similarity or low_snr_target or has_overlap or target_and_other)


def fusion_score(e: DecisionEvidence, cfg) -> float:
    return (
        cfg.weight_speaker_similarity * e.speaker_similarity
        + cfg.weight_target_frame_ratio * e.target_frame_ratio
        + cfg.weight_non_target_frame_ratio * e.non_target_frame_ratio
        + cfg.weight_asr_confidence * e.asr_confidence
        + cfg.weight_command_prior_score * e.command_prior_score
        + cfg.weight_enrollment_bad_quality * e.enrollment_bad_quality
        + cfg.weight_query_noise_penalty * e.query_noise_penalty
    )


def final_decision(e: DecisionEvidence, cfg) -> DecisionResult:
    if should_reject_early(e, cfg):
        return DecisionResult(False, 0.0, "early_reject", use_tse=False)

    use_tse = should_use_tse(e, cfg)
    score = fusion_score(e, cfg)

    if (
        e.speaker_similarity < cfg.speaker_reject_low
        and e.target_frame_ratio < cfg.min_target_ratio
    ):
        return DecisionResult(False, score, "weak_target_evidence", use_tse=use_tse)

    if (
        e.asr_confidence < cfg.min_asr_confidence
        and e.target_frame_ratio < cfg.min_target_ratio
    ):
        return DecisionResult(False, score, "low_asr_and_target_evidence", use_tse=use_tse)

    if score >= cfg.fusion_accept_threshold:
        return DecisionResult(True, score, "accept_by_fusion", use_tse=use_tse)
    return DecisionResult(False, score, "reject_by_fusion", use_tse=use_tse)


def proxy_target_ratios(segment_similarities: list[float], cfg) -> tuple[float, float, float]:
    """Approximate pVAD evidence from segment-level speaker similarities.

    This is not a replacement for a trained target-speaker pVAD. It gives the
    current framework a calibrated placeholder until the dataset is available.
    """
    if not segment_similarities:
        return 0.0, 1.0, 0.0

    sims = [float(s) for s in segment_similarities]
    target_ratio = sum(s >= cfg.speaker_reject_low for s in sims) / len(sims)
    non_target_ratio = sum(s < cfg.speaker_reject_low for s in sims) / len(sims)
    spread = max(sims) - min(sims)
    overlap_probability = min(1.0, max(0.0, spread / max(1e-6, cfg.speaker_accept_high - cfg.speaker_reject_low)))
    return float(target_ratio), float(non_target_ratio), float(overlap_probability)
