#!/usr/bin/env python3
"""Score wake/query speaker pairs from any competition metadata file.

Unlike evaluate_datasetA_speaker_gate.py, this script does not assume a
pos.jsonl/neg.jsonl directory layout. It accepts the same flexible metadata
formats as infer.py and writes score dumps compatible with the offline fusion
scripts.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from audio_quality import preprocess_waveform  # noqa: E402
from audio_utils import load_audio_file  # noqa: E402
from config import apply_env_overrides, load_config  # noqa: E402
from dataset_loader import CompetitionSample, load_meta  # noqa: E402
from speaker_gate import SpeakerGate  # noqa: E402


def _score_sample(sample: CompetitionSample, gate: SpeakerGate, target_sr: int, enable_vad: bool, vad_ratio: float) -> dict[str, Any]:
    start = time.perf_counter()

    wake_wav, wake_sr = load_audio_file(sample.wake_audio)
    wake_wav, wake_sr, wake_quality = preprocess_waveform(
        wake_wav,
        wake_sr,
        target_sr=target_sr,
        enable_vad=enable_vad,
        vad_threshold_ratio=vad_ratio,
    )
    gate.enroll_from_waveform(wake_wav, wake_sr)

    cmd_wav, cmd_sr = load_audio_file(sample.cmd_audio)
    cmd_wav, cmd_sr, query_quality = preprocess_waveform(
        cmd_wav,
        cmd_sr,
        target_sr=target_sr,
        enable_vad=enable_vad,
        vad_threshold_ratio=vad_ratio,
    )
    result = gate.should_accept(cmd_wav, cmd_sr, wake_quality=wake_quality, query_quality=query_quality)

    return {
        "id": sample.id,
        "wake_audio": sample.wake_audio,
        "cmd_audio": sample.cmd_audio,
        "label": sample.label,
        "is_positive": bool((sample.label or "").strip()),
        "score": result.similarity,
        "accepted": result.accepted,
        "reason": result.reason,
        "threshold_used": result.threshold_used,
        "backend_scores": result.backend_scores or {},
        "segment_scores": result.segment_similarities,
        "wake_quality": {
            "score": wake_quality.score,
            "snr_db": wake_quality.snr_db,
            "duration_sec": wake_quality.duration_sec,
            "speech_ratio": wake_quality.speech_ratio,
            "no_speech": wake_quality.no_speech,
        },
        "query_quality": {
            "score": query_quality.score,
            "snr_db": query_quality.snr_db,
            "duration_sec": query_quality.duration_sec,
            "speech_ratio": query_quality.speech_ratio,
            "no_speech": query_quality.no_speech,
        },
        "elapsed_sec": time.perf_counter() - start,
    }


def _metrics_at(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    pos = [r for r in rows if r["is_positive"]]
    neg = [r for r in rows if not r["is_positive"]]
    tp = sum(1 for r in pos if r["score"] >= threshold)
    fn = len(pos) - tp
    tn = sum(1 for r in neg if r["score"] < threshold)
    fp = len(neg) - tn
    accept_rate = tp / len(pos) if pos else 0.0
    reject_rate = tn / len(neg) if neg else 0.0
    return {
        "threshold": threshold,
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "positive_accept_rate": accept_rate,
        "negative_reject_rate": reject_rate,
        "far": fp / len(neg) if neg else 0.0,
        "frr": fn / len(pos) if pos else 0.0,
        "balanced_score": 0.5 * accept_rate + 0.5 * reject_rate,
    }


def _calibrate(rows: list[dict[str, Any]], base_threshold: float) -> dict[str, Any]:
    if not rows or not any(r["is_positive"] for r in rows) or not any(not r["is_positive"] for r in rows):
        return {"current": _metrics_at(rows, base_threshold), "note": "Calibration needs both positive and rejection labels."}

    scores = sorted({float(r["score"]) for r in rows})
    candidates = [scores[0] - 1e-6]
    candidates.extend((a + b) / 2.0 for a, b in zip(scores, scores[1:]))
    candidates.append(scores[-1] + 1e-6)
    metrics = [_metrics_at(rows, t) for t in candidates]
    return {
        "current": _metrics_at(rows, base_threshold),
        "best_balanced": max(metrics, key=lambda m: (m["balanced_score"], m["negative_reject_rate"])),
    }


def _save_outputs(summary_path: Path, dump_path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    dump_path.write_text(json.dumps({"samples": rows}, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Score generic wake/query speaker pairs.")
    parser.add_argument("--meta", required=True, help="Metadata CSV/JSON/JSONL.")
    parser.add_argument("--audio-root", default="", help="Audio root for relative paths.")
    parser.add_argument("--config", default="", help="Pipeline config JSON.")
    parser.add_argument("--output", default="output/speaker_pair_eval.json", help="Summary JSON path.")
    parser.add_argument("--score-dump", default="output/speaker_pair_scores.json", help="Per-sample score dump.")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N samples.")
    parser.add_argument("--checkpoint-every", type=int, default=50, help="Save partial output every N samples.")
    args = parser.parse_args()

    cfg = apply_env_overrides(load_config(args.config or None))
    samples = load_meta(args.meta, audio_root=args.audio_root or None)
    if args.limit > 0:
        samples = samples[: args.limit]

    gate = SpeakerGate(cfg.gate)
    rows: list[dict[str, Any]] = []
    start = time.perf_counter()

    print(f"Loaded {len(samples)} samples from {args.meta}")
    print(f"Gate backend={cfg.gate.embedding_backends}, threshold={cfg.gate.threshold:.4f}")
    for i, sample in enumerate(samples, 1):
        try:
            row = _score_sample(
                sample,
                gate,
                target_sr=cfg.target_sr,
                enable_vad=cfg.preprocess.enable_vad,
                vad_ratio=cfg.preprocess.vad_threshold_ratio,
            )
        except Exception as e:
            row = {
                "id": sample.id,
                "wake_audio": sample.wake_audio,
                "cmd_audio": sample.cmd_audio,
                "label": sample.label,
                "is_positive": bool((sample.label or "").strip()),
                "score": 0.0,
                "accepted": False,
                "reason": f"error:{e}",
                "threshold_used": cfg.gate.threshold,
                "backend_scores": {},
                "segment_scores": [],
                "error": str(e),
                "elapsed_sec": 0.0,
            }
        rows.append(row)

        if args.checkpoint_every > 0 and i % args.checkpoint_every == 0:
            elapsed = time.perf_counter() - start
            partial_summary = {
                "meta": str(Path(args.meta).resolve()),
                "n_total": len(rows),
                "elapsed_sec": elapsed,
                "avg_sec_per_sample": elapsed / max(1, len(rows)),
                "calibration": _calibrate(rows, cfg.gate.threshold),
                "partial": True,
            }
            _save_outputs(Path(args.output), Path(args.score_dump), partial_summary, rows)

        status = "ACCEPT" if row["accepted"] else "REJECT"
        print(f"[{i:04d}/{len(samples)}] {status} score={row['score']:.4f} id={row['id']}")

    elapsed = time.perf_counter() - start
    summary = {
        "meta": str(Path(args.meta).resolve()),
        "audio_root": str(Path(args.audio_root).resolve()) if args.audio_root else "",
        "n_total": len(rows),
        "n_positive": sum(1 for r in rows if r["is_positive"]),
        "n_rejection": sum(1 for r in rows if not r["is_positive"]),
        "elapsed_sec": elapsed,
        "avg_sec_per_sample": elapsed / max(1, len(rows)),
        "calibration": _calibrate(rows, cfg.gate.threshold),
    }
    _save_outputs(Path(args.output), Path(args.score_dump), summary, rows)

    print("\n=== Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved summary: {Path(args.output).resolve()}")
    print(f"Saved scores : {Path(args.score_dump).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
