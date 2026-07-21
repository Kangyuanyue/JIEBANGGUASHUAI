"""Feature extraction for Chinese smart-home command text.

The grammar is intentionally conservative. It does not decide the final answer
by itself; it provides interpretable features for fusion models and simple
runtime safeguards.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from command_postprocess import normalize_command_text


DEVICE_WORDS = (
    "空调",
    "灯",
    "灯光",
    "顶灯",
    "射灯",
    "窗帘",
    "卷帘",
    "洗衣机",
    "洗碗机",
    "干衣机",
    "烘干机",
    "油烟机",
    "烟机",
    "冰箱",
    "电视",
    "智慧屏",
    "屏幕",
    "音乐",
    "闹钟",
    "烤箱",
    "微波炉",
    "蒸箱",
    "饮水机",
    "新风",
    "风扇",
    "净化器",
    "扫地机",
    "地暖",
)

ACTION_WORDS = (
    "打开",
    "开启",
    "关",
    "关闭",
    "关掉",
    "调",
    "调到",
    "调为",
    "调高",
    "调低",
    "设置",
    "切换",
    "播放",
    "暂停",
    "继续",
    "取消",
    "预约",
    "恢复",
    "拉起",
    "放下",
    "升高",
    "降低",
    "洗",
    "烘",
    "制冷",
    "制热",
)

ATTRIBUTE_WORDS = (
    "温度",
    "风量",
    "风速",
    "模式",
    "亮度",
    "色温",
    "声音",
    "音量",
    "角度",
    "百分之",
    "最大",
    "最小",
    "最高",
    "最低",
    "最亮",
    "最暗",
    "最冷",
    "最热",
    "制热",
    "制冷",
    "换气",
    "扫风",
    "档",
    "度",
)

SCENE_WORDS = (
    "回家",
    "出门",
    "做饭",
    "吃饭",
    "用餐",
    "睡觉",
    "休息",
    "电影",
    "会议",
    "烹饪",
    "备餐",
    "日常",
    "浪漫",
    "呼吸",
    "光影",
)

ROOM_WORDS = (
    "客厅",
    "卧室",
    "厨房",
    "书房",
    "前厅",
    "茶室",
    "展台",
    "室内",
    "全屋",
    "所有",
)

ASSISTANT_WORDS = (
    "帮我",
    "我要",
    "我想",
    "准备",
    "怎么",
    "什么时候",
    "播放",
    "放一首",
)

NUMBER_WORDS = tuple("零〇一二两三四五六七八九十百千万0123456789") + (
    "百分之",
    "半",
    "k",
    "度",
    "点",
)

CONFUSION_REWRITES = {
    "自热": "制热",
    "直热": "制热",
    "智热": "制热",
    "新象空调": "新风空调",
    "新上空调": "新风空调",
    "风数": "风速",
    "室温": "色温",
    "声温": "色温",
    "先锋": "新风",
    "西风": "新风",
    "烘共": "烘干",
}

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[a-zA-Z]")
_KANA_HANGUL_RE = re.compile(r"[\u3040-\u30ff\uac00-\ud7af]")


@dataclass
class GrammarFeatures:
    length: int
    cjk_ratio: float
    latin_ratio: float
    kana_hangul_ratio: float
    device_hits: int
    action_hits: int
    attribute_hits: int
    scene_hits: int
    room_hits: int
    assistant_hits: int
    number_hits: int
    confusion_hits: int
    command_score: float
    weird_score: float

    def to_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


def apply_common_rewrites(text: str | None) -> str:
    out = normalize_command_text(text)
    for src, dst in CONFUSION_REWRITES.items():
        out = out.replace(src, dst)
    return out


def _count_hits(text: str, words: tuple[str, ...]) -> int:
    return sum(1 for w in words if w and w in text)


def grammar_features(text: str | None) -> GrammarFeatures:
    t = normalize_command_text(text)
    length = len(t)
    if length <= 0:
        return GrammarFeatures(0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 1.0)

    cjk = len(_CJK_RE.findall(t))
    latin = len(_LATIN_RE.findall(t))
    kana_hangul = len(_KANA_HANGUL_RE.findall(t))
    device_hits = _count_hits(t, DEVICE_WORDS)
    action_hits = _count_hits(t, ACTION_WORDS)
    attribute_hits = _count_hits(t, ATTRIBUTE_WORDS)
    scene_hits = _count_hits(t, SCENE_WORDS)
    room_hits = _count_hits(t, ROOM_WORDS)
    assistant_hits = _count_hits(t, ASSISTANT_WORDS)
    number_hits = sum(1 for w in NUMBER_WORDS if w and w in t)
    confusion_hits = _count_hits(t, tuple(CONFUSION_REWRITES))

    intent_groups = 0
    intent_groups += 1 if device_hits else 0
    intent_groups += 1 if action_hits else 0
    intent_groups += 1 if attribute_hits else 0
    intent_groups += 1 if scene_hits else 0
    intent_groups += 1 if assistant_hits else 0
    intent_groups += 1 if number_hits else 0
    command_score = min(1.0, intent_groups / 3.0)

    non_cjk_ratio = 1.0 - cjk / max(1, length)
    weird_score = min(1.0, 0.6 * non_cjk_ratio + 0.4 * (kana_hangul / max(1, length)))
    if length > 45:
        weird_score = min(1.0, weird_score + 0.2)

    return GrammarFeatures(
        length=length,
        cjk_ratio=cjk / max(1, length),
        latin_ratio=latin / max(1, length),
        kana_hangul_ratio=kana_hangul / max(1, length),
        device_hits=device_hits,
        action_hits=action_hits,
        attribute_hits=attribute_hits,
        scene_hits=scene_hits,
        room_hits=room_hits,
        assistant_hits=assistant_hits,
        number_hits=number_hits,
        confusion_hits=confusion_hits,
        command_score=command_score,
        weird_score=weird_score,
    )
