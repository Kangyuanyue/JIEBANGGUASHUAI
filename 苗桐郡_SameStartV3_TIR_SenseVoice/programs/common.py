#!/usr/bin/env python3
"""Shared utilities for the datasetA SenseVoice experiments."""

from __future__ import annotations

import json
import math
import os
import re
import string
import wave
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


SENSEVOICE_TAG_RE = re.compile(r"<\|[^|]+?\|>")
WHITESPACE_RE = re.compile(r"\s+")
ASCII_PUNCT = set(string.punctuation)
CN_PUNCT = set("，。！？；：、（）【】《》“”‘’—…·￥,.!?;:()[]{}<>\"'`~@#$%^&*_+=|\\/ -")
PUNCT = ASCII_PUNCT | CN_PUNCT

DOMAIN_TERMS = [
    "空调",
    "灯光",
    "洗碗机",
    "冰箱",
    "热水器",
    "洗衣机",
    "窗帘",
    "电视",
    "显示屏",
    "制冷",
    "制热",
    "除湿",
    "睡眠",
    "节能",
    "风量",
    "温度",
    "亮度",
    "百分之",
    "打开",
    "关闭",
    "暂停",
    "继续",
    "调到",
    "设置",
    "模式",
]

MOJIBAKE_MARKERS = set(
    "鍛冧綘閭ｄ浣犲氨鎺у埗鍠芥垜宸笉澶氳繖鍒颁綅缃彉洿病鐪嬫墜機"
    "鎴戝噯備嚭楗婅瘔鍡滑敹闊撳紑娲楃碗鏈"
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_jsonl(path: os.PathLike[str] | str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: os.PathLike[str] | str, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: os.PathLike[str] | str, default: Optional[Any] = None) -> Any:
    path = Path(path)
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: os.PathLike[str] | str, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def normalize_text(text: Optional[str], keep_punct: bool = False) -> str:
    if text is None:
        return ""
    text = str(text)
    text = repair_mojibake(text)
    text = SENSEVOICE_TAG_RE.sub("", text)
    text = WHITESPACE_RE.sub("", text)
    text = text.lower()
    if not keep_punct:
        text = "".join(ch for ch in text if ch not in PUNCT)
    return text


def repair_mojibake(text: str) -> str:
    """Recover UTF-8 Chinese text that was accidentally decoded as GBK.

    Some Windows/FunASR paths can return strings like "鍛冧綘" for "呃你".
    Normal labels usually fail this conversion, so this only accepts the repair
    when the source looks like that specific mojibake pattern.
    """
    if not text:
        return text
    marker_hits = sum(1 for ch in text if ch in MOJIBAKE_MARKERS)
    if marker_hits < 2:
        return text
    try:
        repaired = text.encode("gbk").decode("utf-8")
    except UnicodeError:
        return text
    if "\ufffd" in repaired:
        return text
    return repaired


def levenshtein(a: Sequence[str], b: Sequence[str]) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = cur[j - 1] + 1
            delete_cost = prev[j] + 1
            sub_cost = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(insert_cost, delete_cost, sub_cost))
        prev = cur
    return prev[-1]


def cer(reference: Optional[str], hypothesis: Optional[str]) -> Tuple[int, int]:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    return levenshtein(list(ref), list(hyp)), len(ref)


def load_simple_yaml(path: os.PathLike[str] | str) -> Dict[str, Any]:
    """Read the small key-value YAML files used by this project.

    PyYAML is intentionally optional. This parser supports the flat config files
    produced by optimize_reject.py.
    """
    data: Dict[str, Any] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value.lower() in {"true", "false"}:
                data[key] = value.lower() == "true"
            else:
                try:
                    if any(ch in value for ch in [".", "e", "E"]):
                        data[key] = float(value)
                    else:
                        data[key] = int(value)
                except ValueError:
                    data[key] = value.strip("\"'")
    return data


def write_simple_yaml(path: os.PathLike[str] | str, values: Dict[str, Any], header: str = "") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        if header:
            for line in header.rstrip().splitlines():
                f.write(f"# {line}\n")
        for key in sorted(values):
            value = values[key]
            if isinstance(value, str):
                f.write(f"{key}: \"{value}\"\n")
            else:
                f.write(f"{key}: {value}\n")


def audio_info(path: os.PathLike[str] | str) -> Dict[str, Any]:
    with wave.open(str(path), "rb") as wav:
        frames = wav.getnframes()
        sample_rate = wav.getframerate()
        return {
            "sample_rate": sample_rate,
            "channels": wav.getnchannels(),
            "sample_width": wav.getsampwidth(),
            "frames": frames,
            "duration": frames / float(sample_rate) if sample_rate else math.nan,
        }


def text_domain_score(text: Optional[str]) -> float:
    normalized = normalize_text(text)
    if not normalized:
        return 0.0
    hits = sum(1 for term in DOMAIN_TERMS if term in normalized)
    return min(1.0, hits / 2.0)


def char_ngrams(text: str, n: int = 2) -> Set[str]:
    text = normalize_text(text)
    if not text:
        return set()
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def label_similarity(text: Optional[str], labels: Optional[Sequence[str]] = None) -> float:
    if not labels:
        return 0.0
    text_grams = char_ngrams(text or "")
    if not text_grams:
        return 0.0
    best = 0.0
    for label in labels:
        label_grams = char_ngrams(label)
        if not label_grams:
            continue
        union = text_grams | label_grams
        if not union:
            continue
        score = len(text_grams & label_grams) / len(union)
        if score > best:
            best = score
            if best >= 1.0:
                return 1.0
    return best


def label_edit_similarity(text: Optional[str], labels: Optional[Sequence[str]] = None) -> float:
    """Best normalized edit similarity against known positive labels."""
    if not labels:
        return 0.0
    normalized = normalize_text(text)
    if not normalized:
        return 0.0
    best = 0.0
    for label in labels:
        candidate = normalize_text(label)
        if not candidate:
            continue
        distance = levenshtein(list(normalized), list(candidate))
        denom = max(len(normalized), len(candidate), 1)
        score = max(0.0, 1.0 - distance / denom)
        if score > best:
            best = score
            if best >= 1.0:
                return 1.0
    return best


def should_reject(
    text: Optional[str],
    config: Optional[Dict[str, Any]] = None,
    known_labels: Optional[Sequence[str]] = None,
    audio_duration: Optional[float] = None,
) -> bool:
    config = config or {}
    normalized = normalize_text(text)
    min_chars = int(config.get("min_chars", 1))
    max_chars = int(config.get("max_chars", 0))
    min_domain_score = float(config.get("min_domain_score", 0.0))
    min_label_similarity = float(config.get("min_label_similarity", 0.0))
    min_label_edit_similarity = float(config.get("min_label_edit_similarity", 0.0))
    max_duration_seconds = float(config.get("max_duration_seconds", 0.0))
    reject_empty = bool(config.get("reject_empty", True))
    if reject_empty and not normalized:
        return True
    if len(normalized) < min_chars:
        return True
    if max_chars > 0 and len(normalized) > max_chars:
        return True
    if audio_duration is not None and max_duration_seconds > 0 and audio_duration > max_duration_seconds:
        return True
    if text_domain_score(normalized) < min_domain_score:
        return True
    if known_labels is not None and min_label_similarity > 0:
        if label_similarity(normalized, known_labels) < min_label_similarity:
            return True
    if known_labels is not None and min_label_edit_similarity > 0:
        if label_edit_similarity(normalized, known_labels) < min_label_edit_similarity:
            return True
    return False


def resolve_manifest(project: Path, split_or_path: str) -> Path:
    candidate = Path(split_or_path)
    if candidate.exists():
        return candidate
    split_path = project / "data" / "splits" / f"{split_or_path}.jsonl"
    if split_path.exists():
        return split_path
    raise FileNotFoundError(f"Cannot resolve manifest or split: {split_or_path}")


def extract_prediction_text(row: Dict[str, Any]) -> str:
    for key in ["text", "prediction", "pred_text", "raw_text"]:
        value = row.get(key)
        if value is not None:
            return str(value)
    return ""
