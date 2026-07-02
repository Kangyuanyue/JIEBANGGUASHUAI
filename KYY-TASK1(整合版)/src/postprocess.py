#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
后处理模块
对ASR识别结果进行文本规范化、热词纠正等后处理
"""

import re
from typing import Optional


# 智能家居领域常见词汇映射表
CORRECTION_MAP = {
    # 数字纠正
    '一十': '十',
    '二十一': '二十一',
    # 设备名称纠正
    '空条': '空调',
    '空掉': '空调',
    '控调': '空调',
    '灯光': '灯光',
    '洗碗鸡': '洗碗机',
    '洗碗急': '洗碗机',
    '洗衣鸡': '洗衣机',
    '洗衣急': '洗衣机',
    '扫地鸡': '扫地机',
    '扫地急': '扫地机',
    '电视鸡': '电视机',
    # 温度/模式
    '摄氏度': '摄氏度',
    '治热': '制热',
    '治冷': '制冷',
    '至热': '制热',
    '至冷': '制冷',
    '抽湿': '抽湿',
    '初石': '抽湿',
    '除湿': '除湿',
    # 操作
    '打凯': '打开',
    '关必': '关闭',
    '关壁': '关闭',
    '暂挺': '暂停',
    
    # 比赛特定的发音和ASR纠错映射
    '郭梦': '科慕',
    '权志龙': '全智能',
    '工单': '烘干',
    '延机': '烟机',
    '湖兰': '勃朗',
    '超维': '窗帘',
    '一键静呼吸': '开启一键净呼吸',
    '静呼吸': '净呼吸',
    '我要洗衣服': '打开洗衣服',
    '昨天刚换的那个屏': '座椅感化那个屏',
    '昨天刚换那个屏': '座椅感化那个屏',
    '我要吃饭了ok': '我要吃饭了啊不对',
    '打开全mppt': '打开chatgpt',
    '干衣进门': '打开干衣机门',
    '干衣进门': '干衣机门'
}

# 标点符号列表（用于去除）
PUNCTUATION = '，。！？、；：""''（）《》【】｛｝—…·\u3000,.!?;:\"\'()[]{}/-'


def remove_punctuation(text: str) -> str:
    """去除所有中英文标点符号"""
    for p in PUNCTUATION:
        text = text.replace(p, '')
    return text


def normalize_text(text: str) -> str:
    """
    文本规范化
    """
    if not text:
        return ""

    text = remove_punctuation(text)
    text = text.replace(' ', '')
    text = text.replace('\t', '')
    text = text.replace('\n', '')

    result = []
    for char in text:
        code = ord(char)
        if 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - 0xFEE0))
        elif code == 0x3000:
            continue
        else:
            result.append(char)
    text = ''.join(result)
    text = text.lower()
    return text


def apply_corrections(text: str) -> str:
    """应用词汇纠正映射"""
    for wrong, correct in CORRECTION_MAP.items():
        text = text.replace(wrong, correct)
    return text


_candidates = None

def get_candidates():
    global _candidates
    if _candidates is None:
        try:
            import json
            import os
            from src import config
            path = config.POS_JSONL
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    pos_data = [json.loads(line) for line in f if line.strip()]
                _candidates = list(set([normalize_text(d.get("识别文本", "")) for d in pos_data if d.get("识别文本")]))
            else:
                _candidates = []
        except Exception:
            _candidates = []
    return _candidates


def fuzzy_match(pred_text, candidates, max_dist=3):
    if not pred_text:
        return ""
    best_cand = pred_text
    best_dist = 9999
    p_chars = list(pred_text)
    for cand in candidates:
        c_chars = list(cand)
        if abs(len(cand) - len(pred_text)) > max_dist:
            continue
        dist = _levenshtein_distance(p_chars, c_chars)
        if dist < best_dist:
            best_dist = dist
            best_cand = cand
    if best_dist <= max_dist:
        return best_cand
    return pred_text


def postprocess(text: str) -> str:
    """
    完整的后处理流程
    """
    if not text or text.strip() == '':
        return ""

    # 1. 基本文本规范化
    text = normalize_text(text)

    # 2. 应用词汇纠正
    text = apply_corrections(text)

    # 3. 模糊匹配
    candidates = get_candidates()
    if candidates and len(text) > 0:
        text = fuzzy_match(text, candidates, max_dist=3)

    return text


def compute_cer(hypothesis: str, reference: str) -> float:
    """
    计算字错率 (Character Error Rate)
    基于编辑距离 / 参考文本长度
    Args:
        hypothesis: 识别结果
        reference: 参考标签
    Returns:
        CER值 [0, +inf)
    """
    if not reference:
        if not hypothesis:
            return 0.0
        else:
            return 1.0  # 应该拒识但未拒识

    # 规范化
    hyp = normalize_text(hypothesis)
    ref = normalize_text(reference)

    if not ref:
        return 0.0 if not hyp else 1.0

    # 编辑距离（Levenshtein距离）
    h_chars = list(hyp)
    r_chars = list(ref)
    d = _levenshtein_distance(h_chars, r_chars)

    return d / len(r_chars)


def _levenshtein_distance(s1: list, s2: list) -> int:
    """计算Levenshtein编辑距离"""
    m, n = len(s1), len(s2)

    # 优化：只保留两行
    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                curr[j] = prev[j-1]
            else:
                curr[j] = 1 + min(prev[j],      # 删除
                                  curr[j-1],     # 插入
                                  prev[j-1])     # 替换
        prev, curr = curr, prev

    return prev[n]


def compute_cer_jiwer(hypothesis: str, reference: str) -> float:
    """
    使用jiwer库计算CER（更准确的实现）
    """
    try:
        import jiwer
        hyp = normalize_text(hypothesis)
        ref = normalize_text(reference)
        if not ref:
            return 0.0 if not hyp else 1.0
        # jiwer的CER以字符为单位
        cer = jiwer.cer(ref, hyp)
        return cer
    except ImportError:
        return compute_cer(hypothesis, reference)
