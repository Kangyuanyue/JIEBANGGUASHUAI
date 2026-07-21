#!/usr/bin/env python3
"""
Competition inference entry point (Midea XH-202615).

Usage (after dataset arrives):
    python infer.py \\
        --meta sample_data/meta.example.jsonl \\
        --audio-root /path/to/test_set_A \\
        --output output/result.json

Quick self-test (mock ASR, no FunASR download):
    python infer.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is importable when invoked from another directory
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import apply_env_overrides, load_config  # noqa: E402
from dataset_loader import load_meta  # noqa: E402
from metrics_cer import EvalSummary, SampleResult, save_submission  # noqa: E402
from pipeline import RecognitionPipeline  # noqa: E402


def _self_test() -> int:
    """Validate imports, CER math, and meta loader without requiring dataset."""
    from tests.test_metrics import run_all

    run_all()

    from dataset_loader import load_meta

    example = Path(__file__).parent / "sample_data" / "meta.example.jsonl"
    samples = load_meta(example, audio_root=example.parent)
    assert len(samples) == 2, f"expected 2 example rows, got {len(samples)}"
    assert samples[1].is_rejection
    print(f"[self-test] meta loader OK ({len(samples)} example rows)")

    # Optional: ECAPA gate (requires speechbrain + pretrained weights)
    try:
        import numpy as np
        from config import AsrConfig, GateConfig, PipelineConfig
        from pipeline import RecognitionPipeline

        sr = 16000
        wake = np.random.randn(sr * 2).astype(np.float32) * 0.01
        cfg = PipelineConfig(
            gate=GateConfig(threshold=0.5, num_segments=1, segment_sec=2.0),
            asr=AsrConfig(backend="mock"),
        )
        pipe = RecognitionPipeline(cfg)
        pipe.enroll_wake_waveform(wake, sr)
        out = pipe.infer_command(wake.copy(), sr)
        print(f"[self-test] ECAPA gate OK (sim={out.similarity:.3f}, accepted={out.accepted})")
    except Exception as e:
        print(f"[self-test] ECAPA gate skipped ({e})")

    print("[self-test] All checks passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Midea XH-202615 competition inference")
    parser.add_argument("--meta", type=str, default="", help="Path to meta.csv / .json / .jsonl")
    parser.add_argument("--audio-root", type=str, default="", help="Root dir for relative audio paths")
    parser.add_argument("--config", type=str, default="", help="Pipeline JSON config (default: configs/default.json)")
    parser.add_argument("--output", type=str, default="output/result.json", help="Submission JSON path")
    parser.add_argument("--stats", type=str, default="", help="Optional path for extra stats JSON")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N samples (0=all)")
    parser.add_argument("--mock-asr", action="store_true", help="Use mock ASR (no FunASR download)")
    parser.add_argument("--force-asr", action="store_true", help="Run ASR even when gate rejects")
    parser.add_argument("--self-test", action="store_true", help="Run import/gate self-test and exit")
    args = parser.parse_args()

    if args.self_test:
        return _self_test()

    if not args.meta:
        parser.error("--meta is required unless --self-test is set.")

    cfg = load_config(args.config or None)
    cfg = apply_env_overrides(cfg)
    if args.mock_asr:
        cfg.asr.backend = "mock"
    if args.force_asr:
        cfg.force_asr = True

    meta_path = Path(args.meta)
    if not meta_path.is_file():
        print(f"ERROR: meta file not found: {meta_path}", file=sys.stderr)
        return 1

    audio_root = args.audio_root or str(meta_path.parent)
    samples = load_meta(meta_path, audio_root=audio_root)
    if args.limit > 0:
        samples = samples[: args.limit]

    print(f"Loaded {len(samples)} samples from {meta_path}")
    print(f"Config: gate_threshold={cfg.gate.threshold}, asr={cfg.asr.backend}")

    pipeline = RecognitionPipeline(cfg)
    summary = EvalSummary()

    t_batch_start = time.perf_counter()
    for i, sample in enumerate(samples):
        wake = sample.wake_audio
        cmd = sample.cmd_audio
        if not Path(wake).is_file():
            print(f"  [{i+1}] SKIP missing wake: {wake}")
            summary.results.append(
                SampleResult(
                    id=sample.id,
                    content="",
                    label=sample.label,
                    cer=100.0 if not sample.is_rejection else 0.0,
                    accepted=False,
                    error=f"missing_wake:{wake}",
                )
            )
            continue
        if not Path(cmd).is_file():
            print(f"  [{i+1}] SKIP missing cmd: {cmd}")
            summary.results.append(
                SampleResult(
                    id=sample.id,
                    content="",
                    label=sample.label,
                    cer=100.0 if not sample.is_rejection else 0.0,
                    accepted=False,
                    error=f"missing_cmd:{cmd}",
                )
            )
            continue

        result = pipeline.run_sample(wake, cmd, label=sample.label, sample_id=sample.id)
        summary.results.append(result)
        status = "ACCEPT" if result.accepted else "REJECT"
        print(
            f"  [{i+1}/{len(samples)}] {sample.id} {status} "
            f"sim={result.similarity:.3f} cer={result.cer:.1f} "
            f"content={result.content[:40]!r}"
        )

    summary.total_duration_sec = time.perf_counter() - t_batch_start

    out_path = Path(args.output)
    save_submission(summary, out_path)
    stats = summary.extra_stats()
    print("\n=== Summary ===")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\nSubmission saved → {out_path.resolve()}")

    if args.stats:
        stats_path = Path(args.stats)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
