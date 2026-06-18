#!/usr/bin/env python3
"""
Calibrate speaker-gate threshold on a labeled development set.

Mirrors the idea of ../calibrate.py but for cosine similarity (not spoof logits).

Layout:
    <dev_dir>/meta.csv   — same format as competition infer.py
    OR pass --meta directly

Outputs:
    output/gate_calibration.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import GateConfig, load_config  # noqa: E402
from dataset_loader import load_meta  # noqa: E402
from metrics_cer import is_rejection_sample  # noqa: E402
from pipeline import RecognitionPipeline  # noqa: E402


def _collect_similarities(pipeline: RecognitionPipeline, samples) -> tuple[list[float], list[int]]:
    """Return (similarities, binary labels) where label 1 = should accept (positive)."""
    sims: list[float] = []
    labels: list[int] = []
    for s in samples:
        if not Path(s.wake_audio).is_file() or not Path(s.cmd_audio).is_file():
            continue
        pipeline.enroll_wake(s.wake_audio)
        import numpy as np
        from audio_utils import load_audio_file

        wav, sr = load_audio_file(s.cmd_audio)
        gate = pipeline.gate.should_accept(wav, sr)
        sims.append(gate.similarity)
        labels.append(0 if is_rejection_sample(s.label) else 1)
    return sims, labels


def _sweep_threshold(sims: np.ndarray, labels: np.ndarray) -> list[dict]:
    rows = []
    for th in np.linspace(0.35, 0.95, 61):
        pred_accept = sims >= th
        pos = labels == 1
        neg = labels == 0
        tp = int((pred_accept & pos).sum())
        fn = int((~pred_accept & pos).sum())
        tn = int((~pred_accept & neg).sum())
        fp = int((pred_accept & neg).sum())
        pos_total = max(1, int(pos.sum()))
        neg_total = max(1, int(neg.sum()))
        accept_recall = 100.0 * tp / pos_total  # proxy for keeping CER samples
        reject_rate = 100.0 * tn / neg_total
        score = 0.4 * accept_recall + 0.4 * reject_rate  # match competition weights
        rows.append(
            {
                "threshold": float(th),
                "accept_recall": round(accept_recall, 2),
                "reject_rate": round(reject_rate, 2),
                "weighted_score": round(score, 2),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            }
        )
    rows.sort(key=lambda r: -r["weighted_score"])
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate speaker gate threshold")
    parser.add_argument("--meta", type=str, required=True)
    parser.add_argument("--audio-root", type=str, default="")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--output", type=str, default="output/gate_calibration.json")
    parser.add_argument("--mock-asr", action="store_true", help="Skip ASR loading (gate only)")
    args = parser.parse_args()

    cfg = load_config(args.config or None)
    if args.mock_asr:
        cfg.asr.backend = "mock"

    meta_path = Path(args.meta)
    audio_root = args.audio_root or str(meta_path.parent)
    samples = load_meta(meta_path, audio_root=audio_root)
    if not samples:
        print("No samples loaded.", file=sys.stderr)
        return 1

    pipeline = RecognitionPipeline(cfg)
    sims_list, labels_list = _collect_similarities(pipeline, samples)
    if not sims_list:
        print("No valid audio pairs found.", file=sys.stderr)
        return 1

    sims = np.array(sims_list, dtype=np.float64)
    labels = np.array(labels_list, dtype=np.int32)
    sweep = _sweep_threshold(sims, labels)
    best = sweep[0]

    out = {
        "n_samples": len(sims_list),
        "n_positive": int((labels == 1).sum()),
        "n_rejection": int((labels == 0).sum()),
        "best_threshold": best["threshold"],
        "best_metrics": best,
        "sweep_top10": sweep[:10],
        "similarity_stats": {
            "positive_mean": float(sims[labels == 1].mean()) if (labels == 1).any() else None,
            "positive_min": float(sims[labels == 1].min()) if (labels == 1).any() else None,
            "rejection_mean": float(sims[labels == 0].mean()) if (labels == 0).any() else None,
            "rejection_max": float(sims[labels == 0].max()) if (labels == 0).any() else None,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nRecommended threshold: {best['threshold']:.4f}")
    print(f"Update configs/default.json → gate.threshold")
    print(f"Saved → {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
