#!/usr/bin/env python
"""Create the fixed 12-item stage-2 CosyVoice pilot manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {args.out}")
    with args.config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    with args.source.open("r", encoding="utf-8") as handle:
        source_rows = {
            str(row["source_id"]): row
            for line in handle
            if line.strip()
            for row in [json.loads(line)]
        }

    selected_ids = [str(value) for value in config["selected_source_ids"]]
    missing = [source_id for source_id in selected_ids if source_id not in source_rows]
    if missing:
        raise ValueError(f"Selected source IDs are missing: {missing}")
    if len(selected_ids) != 12 or len(set(selected_ids)) != 12:
        raise ValueError("Stage 2 requires exactly 12 unique selected source IDs")

    speakers = config["speakers"]
    if len(speakers) < 3:
        raise ValueError("Stage 2 requires at least three speakers")
    for speaker in speakers:
        prompt_path = resolve_project_path(speaker["prompt_audio_path"])
        if not prompt_path.is_file():
            raise FileNotFoundError(f"Missing prompt audio: {prompt_path}")

    args.audio_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, source_id in enumerate(selected_ids, start=1):
        source = source_rows[source_id]
        speaker = speakers[(index - 1) % len(speakers)]
        uid = f"stage2_{index:02d}_{source_id}_{speaker['speaker']}"
        output = (args.audio_dir / f"{uid}.wav").resolve()
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite generated audio: {output}")
        rows.append(
            {
                "uid": uid,
                "source_id": source_id,
                "text": source["text"] + ("" if source["text"].endswith("。") else "。"),
                "speaker": speaker["speaker"],
                "speaker_display_name": speaker["display_name"],
                "prompt_text": speaker["prompt_text"],
                "prompt_audio_path": str(resolve_project_path(speaker["prompt_audio_path"])),
                "prompt_source": speaker["source"],
                "prompt_source_detail": speaker["source_detail"],
                "output_audio_path": str(output),
                "speed": float(config["speed"]),
                "split": "stage2_pilot",
                "allowed_for_training": True,
                "contains_official_test_data": False,
            }
        )

    with args.out.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"items": len(rows), "speakers": len(speakers), "out": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
