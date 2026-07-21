"""Label-free consensus and guarded TSE text routing."""

from __future__ import annotations

from command_grammar import grammar_features
from command_postprocess import command_prior_score, normalize_command_text
from metrics_cer import _levenshtein_chars


def select_consensus_text(
    texts: list[str],
    weird_weight: float = 0.35,
    command_weight: float = 0.10,
) -> tuple[str, int, list[float]]:
    normalized = [normalize_command_text(text) for text in texts]
    scores = []
    for text in normalized:
        consensus_cost = sum(_levenshtein_chars(text, other) for other in normalized)
        features = grammar_features(text)
        scores.append(
            float(consensus_cost)
            + weird_weight * features.weird_score
            - command_weight * features.command_score
        )
    selected = min(range(len(scores)), key=lambda index: (scores[index], index))
    return normalized[selected], selected, scores


def should_select_tse_text(
    baseline_text: str,
    tse_text: str,
    raw_candidates: list[str],
    similarity_gain: float,
    min_similarity_gain: float = -0.05,
    max_text_distance_ratio: float = 0.25,
    require_command_prior_not_worse: bool = True,
) -> bool:
    baseline = normalize_command_text(baseline_text)
    tse = normalize_command_text(tse_text)
    candidates = [normalize_command_text(text) for text in raw_candidates]
    if not tse or similarity_gain < min_similarity_gain:
        return False
    if require_command_prior_not_worse and command_prior_score(tse) < command_prior_score(baseline):
        return False
    if not candidates:
        return False
    min_distance = min(_levenshtein_chars(tse, text) for text in candidates)
    length = max(1, max([len(tse), *(len(text) for text in candidates)]))
    return min_distance <= max(1, int(max_text_distance_ratio * length))
