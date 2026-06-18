#!/usr/bin/env python3
"""Evaluate and calibrate the speaker verification subsystem.

Supported inputs:
  1) Competition meta: --meta path/to/meta.jsonl --audio-root data
     Positive samples are rows with non-empty labels; rejection rows are negatives.
  2) Speaker trials: --trials path/to/trials.csv --audio-root data
     CSV/JSONL fields: enroll_audio, test_audio, label (1=same speaker, 0=different)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import apply_env_overrides, load_config  # noqa: E402
from dataset_loader import load_meta  # noqa: E402
from metrics_cer import is_rejection_sample  # noqa: E402


@dataclass
class SpeakerTrial:
    enroll_audio: str
    test_audio: str
    label: int
    trial_id: str = ""


@dataclass
class SpeakerEvalResult:
    eer: float
    eer_threshold: float
    min_dcf: float
    min_dcf_threshold: float
    best_competition_threshold: float
    best_competition_score: float
    n_trials: int
    n_positive: int
    n_negative: int
    positive_mean: float | None
    negative_mean: float | None


def _resolve_audio(path: str, root: Path) -> str:
    p = Path(path)
    if p.is_file():
        return str(p.resolve())
    return str((root / path).resolve())


def _pick(row: dict[str, Any], names: Iterable[str], default: str = "") -> str:
    lowered = {str(k).lower(): v for k, v in row.items()}
    for name in names:
        if name in row and str(row[name]).strip():
            return str(row[name]).strip()
        val = lowered.get(name.lower())
        if val is not None and str(val).strip():
            return str(val).strip()
    return default


def load_trials(path: str | Path, audio_root: str | Path = "") -> list[SpeakerTrial]:
    path = Path(path)
    root = Path(audio_root) if audio_root else path.parent
    rows: list[dict[str, Any]] = []
    if path.suffix.lower() == ".csv":
        with open(path, encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
    elif path.suffix.lower() == ".jsonl":
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        raise ValueError("Trials must be .csv or .jsonl")

    trials: list[SpeakerTrial] = []
    for i, row in enumerate(rows):
        enroll = _pick(row, ("enroll_audio", "enroll", "wake_audio", "audio1", "path1"))
        test = _pick(row, ("test_audio", "test", "cmd_audio", "query_audio", "audio2", "path2"))
        label_raw = _pick(row, ("label", "target", "same_speaker", "is_same"))
        if not enroll or not test or label_raw == "":
            raise ValueError(f"Missing trial fields at row {i}: {row}")
        label = 1 if str(label_raw).strip().lower() in {"1", "true", "same", "target", "yes"} else 0
        trials.append(
            SpeakerTrial(
                enroll_audio=_resolve_audio(enroll, root),
                test_audio=_resolve_audio(test, root),
                label=label,
                trial_id=_pick(row, ("id", "trial_id"), default=f"trial_{i}"),
            )
        )
    return trials


def competition_meta_to_trials(meta: str | Path, audio_root: str | Path = "") -> list[SpeakerTrial]:
    samples = load_meta(meta, audio_root=audio_root or None)
    trials: list[SpeakerTrial] = []
    for i, sample in enumerate(samples):
        trials.append(
            SpeakerTrial(
                enroll_audio=sample.wake_audio,
                test_audio=sample.cmd_audio,
                label=0 if is_rejection_sample(sample.label) else 1,
                trial_id=sample.id or f"sample_{i}",
            )
        )
    return trials


def collect_scores(trials: list[SpeakerTrial], config_path: str = "") -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    from audio_utils import load_audio_file
    from pipeline import RecognitionPipeline

    cfg = apply_env_overrides(load_config(config_path or None))
    cfg.asr.backend = "mock"
    pipe = RecognitionPipeline(cfg)

    scores: list[float] = []
    labels: list[int] = []
    details: list[dict[str, Any]] = []
    for trial in trials:
        if not Path(trial.enroll_audio).is_file() or not Path(trial.test_audio).is_file():
            continue
        try:
            pipe.enroll_wake(trial.enroll_audio)
            wav, sr = load_audio_file(trial.test_audio)
            out = pipe.infer_command(wav, sr)
        except Exception as e:
            details.append(
                {
                    "id": trial.trial_id,
                    "enroll_audio": trial.enroll_audio,
                    "test_audio": trial.test_audio,
                    "label": int(trial.label),
                    "error": repr(e),
                }
            )
            continue
        scores.append(out.similarity)
        labels.append(int(trial.label))
        details.append(
            {
                "id": trial.trial_id,
                "enroll_audio": trial.enroll_audio,
                "test_audio": trial.test_audio,
                "label": int(trial.label),
                "score": out.similarity,
                "threshold_used": out.gate.threshold_used,
                "accepted": out.accepted,
                "reason": out.gate.reason,
                "backend_scores": out.gate.backend_scores or {},
                "segment_similarities": out.gate.segment_similarities,
                "decision_reason": out.decision.reason,
            }
        )
    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int32), details


def _rates_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> tuple[float, float]:
    pred = scores >= threshold
    pos = labels == 1
    neg = labels == 0
    fnr = float((~pred & pos).sum() / max(1, pos.sum()))
    fpr = float((pred & neg).sum() / max(1, neg.sum()))
    return fnr, fpr


def compute_metrics(scores: np.ndarray, labels: np.ndarray, p_target: float = 0.01) -> SpeakerEvalResult:
    if scores.size == 0:
        raise ValueError("No valid speaker scores collected.")

    thresholds = np.unique(np.concatenate(([scores.min() - 1e-6], scores, [scores.max() + 1e-6])))
    eer = 1.0
    eer_threshold = float(thresholds[0])
    min_gap = float("inf")
    min_dcf = float("inf")
    min_dcf_threshold = eer_threshold
    best_score = -1.0
    best_competition_threshold = eer_threshold

    for th in thresholds:
        fnr, fpr = _rates_at_threshold(scores, labels, float(th))
        gap = abs(fnr - fpr)
        if gap < min_gap:
            min_gap = gap
            eer = (fnr + fpr) / 2.0
            eer_threshold = float(th)
        dcf = p_target * fnr + (1.0 - p_target) * fpr
        if dcf < min_dcf:
            min_dcf = dcf
            min_dcf_threshold = float(th)
        accept_recall = 1.0 - fnr
        reject_rate = 1.0 - fpr
        comp_score = 0.4 * accept_recall + 0.4 * reject_rate
        if comp_score > best_score:
            best_score = comp_score
            best_competition_threshold = float(th)

    pos_scores = scores[labels == 1]
    neg_scores = scores[labels == 0]
    return SpeakerEvalResult(
        eer=100.0 * eer,
        eer_threshold=eer_threshold,
        min_dcf=min_dcf,
        min_dcf_threshold=min_dcf_threshold,
        best_competition_threshold=best_competition_threshold,
        best_competition_score=100.0 * best_score,
        n_trials=int(scores.size),
        n_positive=int((labels == 1).sum()),
        n_negative=int((labels == 0).sum()),
        positive_mean=float(pos_scores.mean()) if pos_scores.size else None,
        negative_mean=float(neg_scores.mean()) if neg_scores.size else None,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Speaker verification evaluation/calibration")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--meta", type=str, help="Competition meta csv/json/jsonl")
    src.add_argument("--trials", type=str, help="Speaker verification trials csv/jsonl")
    parser.add_argument("--audio-root", type=str, default="")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--output", type=str, default="output/speaker_eval.json")
    parser.add_argument("--score-dump", type=str, default="")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate at most N trials")
    args = parser.parse_args()

    if args.meta:
        trials = competition_meta_to_trials(args.meta, args.audio_root)
    else:
        trials = load_trials(args.trials, args.audio_root)
    if args.limit > 0:
        trials = trials[: args.limit]

    scores, labels, details = collect_scores(trials, config_path=args.config)
    metrics = compute_metrics(scores, labels)
    out = {"metrics": asdict(metrics), "recommended_gate_threshold": metrics.best_competition_threshold}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))

    if args.score_dump:
        dump_path = Path(args.score_dump)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
