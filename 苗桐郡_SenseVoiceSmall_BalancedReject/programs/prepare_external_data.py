#!/usr/bin/env python3
"""Inventory external data without mixing datasetA into training manifests."""

from __future__ import annotations

import argparse
import json
import tarfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List

from common import write_json


def describe_path(path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "type": "missing",
        "file_count": 0,
        "wav_count": 0,
        "jsonl_count": 0,
        "txt_count": 0,
        "warning": "",
    }
    if not path.exists():
        return info
    if "dataseta" in str(path).lower():
        info["warning"] = "datasetA path detected; it must not be used as external training data."
    if path.is_dir():
        info["type"] = "dir"
        for child in path.rglob("*"):
            if not child.is_file():
                continue
            info["file_count"] += 1
            suffix = child.suffix.lower()
            if suffix == ".wav":
                info["wav_count"] += 1
            elif suffix == ".jsonl":
                info["jsonl_count"] += 1
            elif suffix == ".txt":
                info["txt_count"] += 1
        return info
    info["type"] = "file"
    info["file_count"] = 1
    suffix = path.suffix.lower()
    if suffix == ".zip":
        info["type"] = "zip"
        with zipfile.ZipFile(path, "r") as archive:
            names = archive.namelist()
        info["file_count"] = len(names)
        info["wav_count"] = sum(1 for name in names if name.lower().endswith(".wav"))
        info["jsonl_count"] = sum(1 for name in names if name.lower().endswith(".jsonl"))
        info["txt_count"] = sum(1 for name in names if name.lower().endswith(".txt"))
    elif suffix in {".tgz", ".gz", ".tar"}:
        info["type"] = "tar"
        with tarfile.open(path, "r:*") as archive:
            members = archive.getmembers()
        info["file_count"] = sum(1 for item in members if item.isfile())
        info["wav_count"] = sum(1 for item in members if item.isfile() and item.name.lower().endswith(".wav"))
        info["jsonl_count"] = sum(1 for item in members if item.isfile() and item.name.lower().endswith(".jsonl"))
        info["txt_count"] = sum(1 for item in members if item.isfile() and item.name.lower().endswith(".txt"))
    elif suffix == ".wav":
        info["wav_count"] = 1
    elif suffix == ".jsonl":
        info["jsonl_count"] = 1
    elif suffix == ".txt":
        info["txt_count"] = 1
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", action="append", default=[], type=Path, help="External data path to inventory")
    parser.add_argument("--out", default=Path("data/external/external_data_inventory.json"), type=Path)
    args = parser.parse_args()

    project = Path(__file__).resolve().parents[1]
    out = args.out if args.out.is_absolute() else project / args.out
    rows: List[Dict[str, Any]] = [describe_path(path.resolve()) for path in args.path]
    summary = {
        "purpose": "External data inventory. datasetA is excluded from formal training.",
        "items": rows,
        "usable_asr_training_data_found": any(
            item["exists"] and item["jsonl_count"] > 0 and not item["warning"] for item in rows
        ),
        "notes": [
            "AISHELL-WakeUp / HI-MIA-CW are wake-word or confusable wake-word data, not command ASR transcripts.",
            "WHAM-style data is noise/mixture data and should be used only for augmentation.",
            "datasetA must remain test-only.",
        ],
    }
    write_json(out, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
