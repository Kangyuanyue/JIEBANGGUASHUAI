"""Conservative Chinese smart-home command text normalization."""

from __future__ import annotations

import re
import unicodedata

_SENSEVOICE_TAG_RE = re.compile(r"^(?:\|[^|]*\|)+")

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
    "开启",
    "关闭",
    "关掉",
    "暂停",
    "继续",
    "设置",
    "切换",
    "调到",
    "调为",
    "调高",
    "调低",
    "升高",
    "降低",
    "播放",
    "空调",
    "灯",
    "灯光",
    "电视",
    "窗帘",
    "洗碗机",
    "洗衣机",
    "干衣机",
    "冰箱",
    "净化器",
    "风扇",
    "新风",
    "油烟机",
    "烤箱",
    "微波炉",
    "扫地机",
    "温度",
    "风速",
    "风量",
    "模式",
    "亮度",
    "色温",
    "音量",
    "百分之",
    "客厅",
    "卧室",
    "厨房",
    "书房",
)


def normalize_command_text(text: str | None, normalize_digits: bool = False) -> str:
    if text is None:
        return ""
    out = unicodedata.normalize("NFKC", str(text)).strip().lower()
    out = _SENSEVOICE_TAG_RE.sub("", out).strip()
    chars = []
    for ch in out:
        if ch.isspace() or unicodedata.category(ch).startswith("P"):
            continue
        chars.append(ch)
    out = "".join(chars)
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
