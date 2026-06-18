"""
Character Error Rate (CER) and Rejection Rate (RR) for competition evaluation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _levenshtein_chars(ref: str, hyp: str) -> int:
    """Edit distance at character level (for Chinese)."""
    ref = ref or ""
    hyp = hyp or ""
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)

    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, start=1):
        cur = [i]
        for j, hc in enumerate(hyp, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if rc == hc else 1)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def cer_single(reference: str, hypothesis: str) -> float:
    ref = (reference or "").strip()
    hyp = (hypothesis or "").strip()
    if not ref and not hyp:
        return 0.0
    if not ref:
        return 100.0 if hyp else 0.0
    dist = _levenshtein_chars(ref, hyp)
    return 100.0 * dist / len(ref)


def is_rejection_sample(label: str) -> bool:
    """拒识样本：标签为空。"""
    return label is None or str(label).strip() == ""


def is_correct_rejection(label: str, content: str) -> bool:
    return is_rejection_sample(label) and (content or "").strip() == ""


def is_correct_acceptance(label: str, content: str) -> bool:
    return not is_rejection_sample(label) and (content or "").strip() != ""


@dataclass
class SampleResult:
    id: str
    content: str
    label: str
    cer: float
    accepted: bool
    similarity: float = 0.0
    elapsed_sec: float = 0.0
    error: Optional[str] = None

    def to_submission_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "content": self.content,
            "label": self.label,
            "cer": f"{self.cer:.2f}",
        }
        return d


@dataclass
class EvalSummary:
    results: List[SampleResult] = field(default_factory=list)
    total_duration_sec: float = 0.0

    @property
    def positive_samples(self) -> List[SampleResult]:
        return [r for r in self.results if not is_rejection_sample(r.label)]

    @property
    def rejection_samples(self) -> List[SampleResult]:
        return [r for r in self.results if is_rejection_sample(r.label)]

    def final_cer(self) -> float:
        pos = self.positive_samples
        if not pos:
            return 0.0
        refs = sum(len((r.label or "").strip()) for r in pos)
        if refs == 0:
            return float(sum(r.cer for r in pos) / len(pos))
        errs = sum(_levenshtein_chars(r.label, r.content) for r in pos)
        return 100.0 * errs / refs

    def rejection_rate(self) -> float:
        rej = self.rejection_samples
        if not rej:
            return 0.0
        correct = sum(1 for r in rej if is_correct_rejection(r.label, r.content))
        return 100.0 * correct / len(rej)

    def to_submission_json(self) -> Dict[str, Any]:
        return {
            "result": {
                "results": [r.to_submission_dict() for r in self.results],
                "final_cer": f"{self.final_cer():.2f}",
                "duration": f"{self.total_duration_sec:.4f}",
            }
        }

    def extra_stats(self) -> Dict[str, Any]:
        pos = self.positive_samples
        rej = self.rejection_samples
        n = len(self.results)
        avg_time = self.total_duration_sec / max(1, n)
        return {
            "n_total": n,
            "n_positive": len(pos),
            "n_rejection": len(rej),
            "final_cer": round(self.final_cer(), 2),
            "rejection_rate": round(self.rejection_rate(), 2),
            "total_duration_sec": round(self.total_duration_sec, 4),
            "avg_sec_per_sample": round(avg_time, 4),
        }


def save_submission(summary: EvalSummary, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary.to_submission_json(), f, ensure_ascii=False, indent=2)


class InferenceTimer:
    """Context manager for batch=1 timing (competition requirement)."""

    def __init__(self):
        self._start: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self._start
