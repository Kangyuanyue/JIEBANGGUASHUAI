#!/usr/bin/env python3
"""Add raw/TSE speaker-similarity evidence to an existing TSE ASR evaluation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from audio_utils import load_audio_file  # noqa: E402
from config import load_config  # noqa: E402
from dataset_loader import load_meta  # noqa: E402
from speaker_gate import SpeakerGate  # noqa: E402
from tse_model import PositiveNegativeTSE, build_pseudo_enrollments  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="output/datasetA_tse_asr_eval_200.json")
    parser.add_argument("--meta", default="output/datasetA_all.jsonl")
    parser.add_argument("--audio-root", default="datasetA")
    parser.add_argument("--config", default="configs/speaker_frontend_no_vad_15s.json")
    parser.add_argument("--checkpoint", default="pretrained/tse_posneg_cnceleb_stage2.pt")
    parser.add_argument("--output", default="output/datasetA_tse_routing_eval_200.json")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    source_rows = {str(row["id"]): row for row in payload["samples"]}
    samples = {str(sample.id): sample for sample in load_meta(args.meta, args.audio_root)}
    cfg = load_config(args.config)
    gate = SpeakerGate(cfg.gate)
    tse = PositiveNegativeTSE(checkpoint=args.checkpoint, device=args.device)
    rows = []
    started = time.perf_counter()

    for index, (sample_id, row) in enumerate(source_rows.items(), 1):
        sample = samples[sample_id]
        wake, wake_sr = load_audio_file(sample.wake_audio)
        command, command_sr = load_audio_file(sample.cmd_audio)
        enrollments = build_pseudo_enrollments(wake, wake_sr)
        extraction = tse.extract(
            command,
            command_sr,
            enrollments.positive,
            enrollments.sample_rate,
            enrollments.negative,
            enrollments.sample_rate,
        )
        gate.enroll_from_waveform(wake, wake_sr)
        raw_gate = gate.score_waveform(command, command_sr)
        tse_gate = gate.score_waveform(extraction.waveform, extraction.sample_rate)
        enriched = dict(row)
        enriched.update(
            {
                "raw_similarity": raw_gate.similarity,
                "tse_similarity": tse_gate.similarity,
                "similarity_gain": tse_gate.similarity - raw_gate.similarity,
                "raw_segment_similarities": raw_gate.segment_similarities,
                "tse_segment_similarities": tse_gate.segment_similarities,
            }
        )
        rows.append(enriched)
        print(
            f"[{index:04d}/{len(source_rows)}] {sample_id} "
            f"raw={raw_gate.similarity:.3f} tse={tse_gate.similarity:.3f} "
            f"gain={enriched['similarity_gain']:+.3f}"
        )

    result = {k: v for k, v in payload.items() if k != "samples"}
    result.update(
        {
            "routing_config": args.config,
            "routing_elapsed_sec": time.perf_counter() - started,
            "samples": rows,
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
