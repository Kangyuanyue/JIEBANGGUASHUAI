#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局配置模块
包含所有路径、模型参数、阈值等配置项
"""

import os

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "datasetA")
POS_JSONL = os.path.join(DATASET_DIR, "pos.jsonl")
NEG_JSONL = os.path.join(DATASET_DIR, "neg.jsonl")
POS_AUDIO_DIR = os.path.join(DATASET_DIR, "pos")
NEG_AUDIO_DIR = os.path.join(DATASET_DIR, "neg")

# 输出目录
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
RESULT_JSON = os.path.join(OUTPUT_DIR, "result.json")
REPORT_DIR = os.path.join(OUTPUT_DIR, "report")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")

# 模型缓存目录
MODEL_CACHE_DIR = os.path.join(PROJECT_ROOT, "models")

# ============================================================
# 音频参数
# ============================================================
SAMPLE_RATE = 16000          # 采样率
MAX_AUDIO_LEN_SEC = 30       # 最大音频长度（秒）

# ============================================================
# 说话人验证参数
# ============================================================
# 相似度阈值：高于此值认为是同一人
SPEAKER_SIMILARITY_THRESHOLD = 0.26
# 使用的说话人验证模型
SPEAKER_MODEL_TYPE = "ensemble"  # "ensemble", "campplus", or "eres2netv2"

# 动态相似度阈值参数: T = a * SNR + b
SPEAKER_DYNAMIC_A = 0.0075
SPEAKER_DYNAMIC_B = 0.155
SPEAKER_DYNAMIC_MIN = 0.24
SPEAKER_DYNAMIC_MAX = 0.33


# ============================================================
# ASR引擎配置
# ============================================================
# ASR引擎选择
ASR_ENGINE = "funasr"  # "funasr" or "whisper"

# FunASR 模型配置
FUNASR_MODEL = "paraformer-zh"  # 中文Paraformer模型
FUNASR_VAD_MODEL = "fsmn-vad"
FUNASR_PUNC_MODEL = "ct-punc"

# Whisper 模型配置
WHISPER_MODEL_SIZE = "large-v3"

# ============================================================
# 语音增强参数
# ============================================================
ENABLE_SPEECH_ENHANCE = True
# 频谱减法参数
SPEC_SUB_ALPHA = 2.0           # 过减因子
SPEC_SUB_BETA = 0.01           # 频谱下限
# 噪声估计帧数
NOISE_ESTIMATE_FRAMES = 5

# ============================================================
# 推理参数
# ============================================================
BATCH_SIZE = 1                 # 竞赛要求batch=1
NUM_WORKERS = 4                # 数据加载线程数
DEVICE = "cpu"                 # "cuda" or "cpu"

# ============================================================
# 评估参数
# ============================================================
# SNR测试条件（dB）
SNR_TEST_CONDITIONS = [5, 0, -5]

# 确保输出目录存在
for d in [OUTPUT_DIR, REPORT_DIR, FIGURE_DIR, MODEL_CACHE_DIR]:
    os.makedirs(d, exist_ok=True)
