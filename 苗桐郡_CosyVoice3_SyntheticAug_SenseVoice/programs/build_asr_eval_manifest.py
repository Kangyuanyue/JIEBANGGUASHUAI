#!/usr/bin/env python
"""Convert a TTS synthesis report into a SenseVoice-compatible manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthesis-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {args.out}")
    rows = []
    with args.synthesis_report.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            source = json.loads(line)
            rows.append(
                {
                    "uid": source["uid"],
                    "audio_cmd": source["output_audio_path"],
                    "label": source["text"],
                    "speaker": source["speaker"],
                    "is_positive": True,
                    "split": "stage2_pilot",
                }
            )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"items": len(rows), "out": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
