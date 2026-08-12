#!/usr/bin/env python3
"""Build a training-eligible source manifest from the verified personal CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_MEMBER = "8"
EXPECTED_NAME = "苗桐郡"


def resolve_path(project: Path, value: Path) -> Path:
    return value if value.is_absolute() else project / value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def stable_uid(source_id: str, filename: str) -> str:
    digest = hashlib.sha256(f"{source_id}|{filename}".encode("utf-8")).hexdigest()[:12]
    return f"spk08_cmd_{source_id}_{digest}"


def build_records(recording_list: Path, audio_root: Path) -> list[dict[str, Any]]:
    rows = read_csv(recording_list)
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_files: set[str] = set()

    for row in rows:
        if row.get("类型", "").strip().upper() != "POS":
            continue
        member = row.get("成员序号", "").strip()
        name = row.get("姓名", "").strip()
        if member != EXPECTED_MEMBER or name != EXPECTED_NAME:
            raise ValueError(f"Unexpected participant: member={member!r}, name={name!r}")

        source_id = row.get("源id", "").strip()
        assigned_row = row.get("分配行号", "").strip()
        filename = row.get("指令文件名", "").strip()
        text = row.get("指令文本", "").strip()
        if not source_id or not filename or not text:
            raise ValueError(f"Incomplete POS row at assigned row {assigned_row!r}")
        if source_id in seen_ids:
            raise ValueError(f"Duplicate source id: {source_id}")
        if filename.casefold() in seen_files:
            raise ValueError(f"Duplicate command filename: {filename}")

        audio_path = (audio_root / "pos" / filename).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(audio_path)
        if audio_path.suffix.casefold() != ".m4a":
            raise ValueError(f"Expected M4A command audio: {audio_path}")

        records.append(
            {
                "uid": stable_uid(source_id, filename),
                "text": text,
                "speaker": "spk08_苗桐郡",
                "prompt_audio_path": str(audio_path),
                "prompt_text": text,
                "audio_path": str(audio_path),
                "split": "source_pool",
                "tts_engine": None,
                "speed": 1.0,
                "source": "personal_verified_recording",
                "source_id": source_id,
                "assigned_row": int(assigned_row),
                "audio_format": "m4a",
                "is_synthetic": False,
                "allowed_for_training": True,
            }
        )
        seen_ids.add(source_id)
        seen_files.add(filename.casefold())

    records.sort(key=lambda item: item["assigned_row"])
    if len(records) != 80:
        raise ValueError(f"Expected 80 POS command rows, found {len(records)}")
    return records


def write_jsonl(path: Path, rows: list[dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, value: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-list", required=True, type=Path)
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    recording_list = resolve_path(project, args.recording_list).resolve()
    audio_root = (args.audio_root or recording_list.parent).resolve()
    out = resolve_path(project, args.out).resolve()
    summary_path = resolve_path(project, args.summary).resolve()

    records = build_records(recording_list, audio_root)
    write_jsonl(out, records, args.overwrite)
    summary = {
        "participant": EXPECTED_NAME,
        "member_number": int(EXPECTED_MEMBER),
        "seed": args.seed,
        "recording_list": str(recording_list),
        "audio_root": str(audio_root),
        "manifest": str(out),
        "record_count": len(records),
        "split_counts": dict(Counter(item["split"] for item in records)),
        "source_counts": dict(Counter(item["source"] for item in records)),
        "audio_format_counts": dict(Counter(item["audio_format"] for item in records)),
        "unique_text_count": len({item["text"] for item in records}),
        "all_audio_exists": all(Path(item["audio_path"]).is_file() for item in records),
        "training_status": "source_pool_requires_stage3_split",
    }
    write_json(summary_path, summary, args.overwrite)
    print(f"Wrote {len(records)} records to {out}")
    print(f"Wrote summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
