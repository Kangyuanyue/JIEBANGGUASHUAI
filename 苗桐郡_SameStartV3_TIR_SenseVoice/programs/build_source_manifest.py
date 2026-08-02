#!/usr/bin/env python3
"""Convert participant CSV files into one validated source JSONL manifest."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def resolve_audio(root: Path, folder: str, filename: str) -> Path:
    path = root / folder / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def build_rows(inventory_path: Path) -> tuple[list[dict], list[dict]]:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    skipped: list[dict] = []
    for speaker in inventory.get("speakers", []):
        if speaker.get("status") != "ready":
            skipped.append({"speaker_id": speaker["speaker_id"], "status": speaker.get("status")})
            continue
        manifest = Path(speaker["manifest"])
        audio_root = Path(speaker["audio_root"])
        if not manifest.is_file() or not audio_root.is_dir():
            raise FileNotFoundError(f"Invalid ready speaker paths: {speaker['speaker_id']}")
        with manifest.open(encoding="utf-8-sig", newline="") as csv_file:
            csv_rows = list(csv.DictReader(csv_file))
        for index, row in enumerate(csv_rows, start=1):
            row_type = row["类型"].strip().upper()
            common = {
                "speaker_id": speaker["speaker_id"],
                "member_number": int(speaker["member_number"]),
                "name": speaker["name"],
                "source_row": index,
                "assigned_row": row["分配行号"].strip(),
                "source_id": row["源id"].strip(),
            }
            if row_type == "POS":
                rows.append(
                    {
                        **common,
                        "kind": "positive_pair",
                        "wake_path": str(resolve_audio(audio_root, "pos", row["唤醒文件名"].strip())),
                        "wake_text": row["唤醒文本"].strip(),
                        "command_path": str(resolve_audio(audio_root, "pos", row["指令文件名"].strip())),
                        "command_text": row["指令文本"].strip(),
                    }
                )
            elif row_type == "NEG":
                rows.append(
                    {
                        **common,
                        "kind": "negative_wake",
                        "wake_path": str(resolve_audio(audio_root, "neg", row["唤醒文件名"].strip())),
                        "wake_text": row["唤醒文本"].strip(),
                    }
                )
            elif row_type == "NOISE":
                filename = row["指令文件名"].strip()
                rows.append(
                    {
                        **common,
                        "kind": "noise",
                        "noise_path": str(resolve_audio(audio_root, "noise", filename)),
                    }
                )
            else:
                raise ValueError(f"Unsupported 类型={row_type!r} in {manifest}, row {index}")
    rows.sort(key=lambda item: (item["speaker_id"], item["kind"], item["source_row"]))
    return rows, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists() or args.summary.exists():
        raise FileExistsError("Output already exists; choose a new path to preserve prior results")
    rows, skipped = build_rows(args.inventory)
    write_jsonl(args.out, rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["kind"]] = counts.get(row["kind"], 0) + 1
    summary = {
        "inventory": str(args.inventory.resolve()),
        "output": str(args.out.resolve()),
        "record_count": len(rows),
        "counts": counts,
        "ready_speakers": sorted({row["speaker_id"] for row in rows}),
        "skipped_speakers": skipped,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
