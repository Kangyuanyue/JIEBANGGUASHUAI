#!/usr/bin/env python3
"""Build target/non-target speaker trials from external metadata.

This combines AISHELL-WakeUp short enrollment utterances with local 16k wavs:
- positive trials: AISHELL wake utterance vs another AISHELL wake utterance
- negative trials: AISHELL wake utterance vs local_data utterance

The result is not a full command-recognition dataset, but it is useful for
calibrating short-enrollment target speaker rejection without using DatasetA.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def rel_or_abs(path: str, root: Path) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(p)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build external target speaker trials.")
    parser.add_argument("--wakeup-meta", default="output/external_training_metadata/aishell_wakeup_160.jsonl")
    parser.add_argument("--local-meta", default="output/external_training_metadata/local_data_16k_wav_inventory.jsonl")
    parser.add_argument("--audio-root", default=".")
    parser.add_argument("--output", default="output/external_training_metadata/external_target_speaker_trials.csv")
    parser.add_argument("--num-positive", type=int, default=1000)
    parser.add_argument("--num-negative", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260710)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    root = Path(args.audio_root)
    wake_rows = [r for r in read_jsonl(Path(args.wakeup_meta)) if r.get("has_audio") and r.get("audio_path")]
    local_rows = [r for r in read_jsonl(Path(args.local_meta)) if r.get("audio_path")]

    if len(wake_rows) < 2:
        raise SystemExit("Need at least two AISHELL-WakeUp rows with audio.")
    if not local_rows:
        raise SystemExit("Need local_data rows with audio.")

    rows: list[tuple[str, str, int, str]] = []
    for i in range(args.num_positive):
        a, b = rng.sample(wake_rows, 2)
        rows.append((rel_or_abs(a["audio_path"], root), rel_or_abs(b["audio_path"], root), 1, f"pos_{i:06d}"))
    for i in range(args.num_negative):
        a = rng.choice(wake_rows)
        b = rng.choice(local_rows)
        rows.append((rel_or_abs(a["audio_path"], root), rel_or_abs(b["audio_path"], root), 0, f"neg_{i:06d}"))

    rng.shuffle(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["enroll_audio", "test_audio", "label", "id"])
        writer.writerows(rows)

    summary = {
        "output": str(out.resolve()),
        "n_trials": len(rows),
        "n_positive": args.num_positive,
        "n_negative": args.num_negative,
        "wakeup_rows_with_audio": len(wake_rows),
        "local_rows": len(local_rows),
        "note": "AISHELL-WakeUp sample has one speaker-like id, so negatives come from local_data.",
    }
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
