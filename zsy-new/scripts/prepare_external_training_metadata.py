#!/usr/bin/env python3
"""Prepare inventory metadata for external, non-DatasetA training data."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_aishell_wakeup(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    text_path = root / "SPEECHDATA" / "speech" / "text" / "160.txt"
    wav_root = root / "SPEECHDATA" / "speech" / "wav"
    rar_path = wav_root / "160.rar"
    extracted_wavs = {p.name: p for p in wav_root.rglob("*.wav")}
    rows: list[dict[str, Any]] = []
    if text_path.is_file():
        with text_path.open("r", encoding="utf-8", errors="ignore") as f:
            for idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split(maxsplit=1)
                wav_name = parts[0]
                text = parts[1] if len(parts) > 1 else ""
                speaker_id = wav_name.split("_", 1)[0]
                wav_path = extracted_wavs.get(wav_name)
                rows.append(
                    {
                        "id": f"aishell_wakeup_{idx:06d}",
                        "source": "AISHELL-WakeUp-1-sample",
                        "speaker_id": speaker_id,
                        "audio_name": wav_name,
                        "audio_path": "" if wav_path is None else str(wav_path.resolve()),
                        "text": text,
                        "has_audio": wav_path is not None,
                    }
                )

    summary = {
        "root": str(root.resolve()),
        "text_path": str(text_path),
        "text_exists": text_path.is_file(),
        "rar_path": str(rar_path),
        "rar_exists": rar_path.is_file(),
        "extracted_wav_count": len(extracted_wavs),
        "metadata_rows": len(rows),
        "rows_with_audio": sum(1 for r in rows if r["has_audio"]),
        "speaker_like_count": len({r["speaker_id"] for r in rows}),
        "top_texts": Counter(r["text"] for r in rows).most_common(10),
    }
    return rows, summary


_DATA_NAME_RE = re.compile(
    r"^(?P<speaker>\d+)_(?P<meta1>[^_]+)_(?P<meta2>[^_]+)_(?P<speed>[^_]+)_(?P<utt>\d+)\.wav$",
    re.IGNORECASE,
)


def parse_data_wavs(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    wavs = sorted(root.rglob("*.wav"))
    for path in wavs:
        m = _DATA_NAME_RE.match(path.name)
        if m:
            info = m.groupdict()
            speaker_id = info["speaker"]
        else:
            info = {}
            speaker_id = path.stem.split("_", 1)[0]
        rows.append(
            {
                "id": f"data_{len(rows) + 1:06d}",
                "source": "local_data_16k_wav_file",
                "speaker_id": speaker_id,
                "audio_path": str(path.resolve()),
                "audio_name": path.name,
                "meta": info,
                "text": "",
                "has_text": False,
            }
        )
    summary = {
        "root": str(root.resolve()),
        "wav_count": len(rows),
        "speaker_like_count": len({r["speaker_id"] for r in rows}),
        "speaker_like_top10": Counter(r["speaker_id"] for r in rows).most_common(10),
        "speed_top10": Counter((r["meta"] or {}).get("speed", "") for r in rows).most_common(10),
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare external training metadata inventory.")
    parser.add_argument(
        "--aishell-wakeup-root",
        default="AISHELL-WakeUp-1-sample/AISHELL-WakeUp-1-sample",
    )
    parser.add_argument("--local-data-root", default="data/16k_wav_file")
    parser.add_argument("--output-dir", default="output/external_training_metadata")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wake_rows, wake_summary = parse_aishell_wakeup(Path(args.aishell_wakeup_root))
    data_rows, data_summary = parse_data_wavs(Path(args.local_data_root))

    write_jsonl(out_dir / "aishell_wakeup_160.jsonl", wake_rows)
    write_jsonl(out_dir / "local_data_16k_wav_inventory.jsonl", data_rows)
    wake_note = (
        "AISHELL-WakeUp audio is extracted and linked in metadata."
        if wake_summary["rows_with_audio"] > 0
        else "AISHELL-WakeUp audio rows will have has_audio=false until 160.rar is extracted."
    )
    inventory = {
        "aishell_wakeup": wake_summary,
        "local_data": data_summary,
        "notes": [
            "DatasetA is intentionally excluded from this training inventory.",
            wake_note,
            "local_data has no text labels in the scanned directory, so it is initially usable for speaker verification and negative construction.",
        ],
    }
    (out_dir / "inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(inventory, ensure_ascii=False, indent=2))
    print(f"Saved metadata to: {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
