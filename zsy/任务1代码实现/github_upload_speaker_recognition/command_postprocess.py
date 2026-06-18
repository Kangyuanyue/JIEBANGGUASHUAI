"""Conservative Chinese smart-home command text normalization."""

from __future__ import annotations

import re


_PUNCT_RE = re.compile(r"[\s\t\n\r，。！？、；：,.!?;:\"'“”‘’（）()【】\[\]《》<>]+")

_FILLERS = (
    "嗯",
    "啊",
    "呃",
    "那个",
    "这个",
    "请",
    "帮我",
    "给我",
)

_DIGIT_MAP = {
    "零": "0",
    "〇": "0",
    "一": "1",
    "二": "2",
    "两": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "七": "7",
    "八": "8",
    "九": "9",
}

_COMMAND_HINTS = (
    "打开",
    "关闭",
    "调高",
    "调低",
    "设置",
    "切换",
    "暂停",
    "继续",
    "空调",
    "灯",
    "电视",
    "窗帘",
    "净化器",
    "风扇",
    "温度",
    "风速",
    "模式",
    "亮度",
    "客厅",
    "卧室",
    "厨房",
    "书房",
)


def normalize_command_text(text: str | None, normalize_digits: bool = False) -> str:
    if text is None:
        return ""
    out = str(text).strip().lower()
    out = _PUNCT_RE.sub("", out)
    for filler in _FILLERS:
        out = out.replace(filler, "")
    if normalize_digits:
        out = "".join(_DIGIT_MAP.get(ch, ch) for ch in out)
    return out.strip()


def command_prior_score(text: str | None) -> float:
    t = normalize_command_text(text)
    if not t:
        return 0.0
    hits = sum(1 for word in _COMMAND_HINTS if word in t)
    return min(1.0, hits / 2.0)
