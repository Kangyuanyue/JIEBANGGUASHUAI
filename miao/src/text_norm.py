"""Text normalization and CER utilities.

These functions are deliberately lightweight and dependency-free so that the
local scorer can run in a clean environment.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

_PUNCT_RE = re.compile(r"[\s\t\n\r，。！？、；：,.!?;:\"'“”‘’（）()【】\[\]《》<>]+")

_DIGIT_MAP = {
    "零": "0",
    "〇": "0",
    "一": "1",
    "二": "2",
    "两": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "七": "7",
    "八": "8",
    "九": "9",
}


def normalize_text(text: str | None, normalize_digits: bool = False) -> str:
    """Normalize Chinese command text for local CER calculation.

    The official evaluation may use a different normalization policy. Keep this
    function conservative: remove punctuation/space and optionally normalize
    single Chinese digits. Avoid semantic synonym replacement here.
    """
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = _PUNCT_RE.sub("", text)
    if normalize_digits:
        text = "".join(_DIGIT_MAP.get(ch, ch) for ch in text)
    return text


def levenshtein_distance(a: Sequence[str], b: Sequence[str]) -> int:
    """Compute edit distance between two sequences."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ca = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ca == b[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,      # deletion
                cur[j - 1] + 1,   # insertion
                prev[j - 1] + cost,
            )
        prev = cur
    return prev[m]


def char_error_rate(hyp: str | None, ref: str | None, normalize_digits: bool = False) -> float:
    """Return character error rate.

    For empty reference, CER is not meaningful. This function returns 0.0 if both
    are empty and 1.0 if the reference is empty but hypothesis is non-empty.
    RR should be used for negative samples.
    """
    hyp_norm = normalize_text(hyp, normalize_digits=normalize_digits)
    ref_norm = normalize_text(ref, normalize_digits=normalize_digits)

    if len(ref_norm) == 0:
        return 0.0 if len(hyp_norm) == 0 else 1.0

    dist = levenshtein_distance(list(hyp_norm), list(ref_norm))
    return dist / max(1, len(ref_norm))


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0
