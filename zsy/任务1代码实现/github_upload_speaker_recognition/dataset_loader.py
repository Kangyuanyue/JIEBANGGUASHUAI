"""
Load competition metadata (CSV / JSON / JSONL).

Expected fields (flexible column names):
  - wake audio path
  - wake text (optional, for logging)
  - command / recognition audio path  → used as submission id
  - command label (empty = rejection sample)
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class CompetitionSample:
    """One test item."""

    id: str
    wake_audio: str
    cmd_audio: str
    label: str
    wake_text: str = ""

    @property
    def is_rejection(self) -> bool:
        return not (self.label or "").strip()


# Aliases seen in the competition brief and common CSV exports.
_WAKE_AUDIO_KEYS = (
    "wake_audio",
    "wake_path",
    "唤醒音频名字",
    "唤醒音频",
    "wake_file",
    "wake",
)
_WAKE_TEXT_KEYS = ("wake_text", "唤醒文本名字", "唤醒标签", "wake_label")
_CMD_AUDIO_KEYS = (
    "cmd_audio",
    "cmd_path",
    "识别音频名字",
    "识别音频",
    "rec_audio",
    "audio",
    "file",
    "id",
)
_LABEL_KEYS = ("label", "识别文本名字", "识别标签", "text", "reference", "content_label")


def _pick(row: Dict[str, Any], keys: tuple[str, ...], default: str = "") -> str:
    for k in keys:
        if k in row and row[k] is not None:
            v = str(row[k]).strip()
            if v:
                return v
    # case-insensitive fallback
    lower_map = {str(k).lower(): v for k, v in row.items()}
    for k in keys:
        v = lower_map.get(k.lower())
        if v is not None and str(v).strip():
            return str(v).strip()
    return default


def _row_to_sample(row: Dict[str, Any], audio_root: Path) -> CompetitionSample:
    wake = _pick(row, _WAKE_AUDIO_KEYS)
    cmd = _pick(row, _CMD_AUDIO_KEYS)
    label = _pick(row, _LABEL_KEYS, default="")
    wake_text = _pick(row, _WAKE_TEXT_KEYS, default="")

    if not cmd:
        raise ValueError(f"Missing command audio id/path in row: {row}")

    sample_id = cmd if cmd else _pick(row, ("id",), default="unknown")

    def resolve(p: str) -> str:
        if not p:
            return ""
        path = Path(p)
        if path.is_file():
            return str(path.resolve())
        joined = audio_root / p
        if joined.is_file():
            return str(joined.resolve())
        return str(joined)

    return CompetitionSample(
        id=Path(sample_id).name,
        wake_audio=resolve(wake),
        cmd_audio=resolve(cmd),
        label=label,
        wake_text=wake_text,
    )


def load_meta(meta_path: str | Path, audio_root: Optional[str | Path] = None) -> List[CompetitionSample]:
    meta_path = Path(meta_path)
    root = Path(audio_root) if audio_root else meta_path.parent

    suffix = meta_path.suffix.lower()
    rows: List[Dict[str, Any]] = []

    if suffix == ".csv":
        with open(meta_path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    elif suffix == ".jsonl":
        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    elif suffix == ".json":
        with open(meta_path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict):
            if "samples" in data:
                rows = data["samples"]
            elif "data" in data:
                rows = data["data"]
            else:
                # single sample dict
                rows = [data]
        else:
            raise ValueError(f"Unsupported JSON structure in {meta_path}")
    else:
        raise ValueError(f"Unsupported meta format: {suffix} (use .csv, .json, .jsonl)")

    samples = [_row_to_sample(r, root) for r in rows]
    return samples


def iter_existing(samples: List[CompetitionSample]) -> Iterator[CompetitionSample]:
    for s in samples:
        missing = []
        if s.wake_audio and not Path(s.wake_audio).is_file():
            missing.append(f"wake={s.wake_audio}")
        if s.cmd_audio and not Path(s.cmd_audio).is_file():
            missing.append(f"cmd={s.cmd_audio}")
        if missing:
            continue
        yield s
