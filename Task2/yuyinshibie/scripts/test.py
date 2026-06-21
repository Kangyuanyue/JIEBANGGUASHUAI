#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
语音识别批量测试与正确率评估
目录结构：
YUYINSHIBIE/
├── data/
│   ├── wav/C0936/*.wav
│   ├── wav.scp
│   └── trans.txt
├── test.py  (本脚本)
"""
import funasr
print("FunASR 实际加载路径:", funasr.__file__)
import os
import re          # 新增：用于清洗文本
import time
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from jiwer import cer, wer

# ====================== 配置区 ======================
# 文件路径（相对于脚本所在目录）
DATA_DIR = "./data"              # 数据文件夹
WAV_SCP = os.path.join(DATA_DIR, "wav.scp")
TRANS_FILE = os.path.join(DATA_DIR, "trans.txt")
OUTPUT_RESULT = "./output/results.txt"
OUTPUT_SUMMARY = "./output/summary.txt"

# 模型参数
MODEL_NAME = "paraformer-zh"     # 模型名称
DEVICE = "cuda"                  # 或 "cpu"
BATCH_SIZE = 16                  # 批量大小（如果启用 VAD/PUNC，会被自动修正为1）
ENABLE_VAD = False               # 是否启用 VAD（短句建议 False）
ENABLE_PUNC = False              # 是否启用标点恢复（批量推理时需 False 才能加速）
# ===================================================

# 如果启用 VAD 或 PUNC，batch_size 必须为 1
if ENABLE_VAD or ENABLE_PUNC:
    BATCH_SIZE = 1
    print("⚠️ 注意：已启用 VAD 或 PUNC，batch_size 自动设为 1。")

def clean_text(text):
    """
    清洗文本：只保留中文、字母和数字，去除所有标点、空格。
    用于正确计算 CER/WER，避免因格式差异导致虚高。
    """
    return re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)

def load_file_pairs(scp_path, trans_path):
    """读取 wav.scp 和 trans.txt，返回 ID、音频路径、参考文本"""
    # 读取 wav.scp
    audio_map = {}
    with open(scp_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                idx = parts[0]
                rel_path = parts[1]
                # 路径是相对路径，拼接 DATA_DIR 后转为绝对路径
                abs_path = os.path.join(DATA_DIR, rel_path)
                # 如果文件不存在，尝试直接使用相对路径
                if not os.path.exists(abs_path):
                    abs_path = rel_path
                audio_map[idx] = abs_path

    # 读取 trans.txt
    trans_map = {}
    with open(trans_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                idx = parts[0]
                text = " ".join(parts[1:])
                trans_map[idx] = text

    # 取交集，按 wav.scp 中的顺序
    common_ids = [idx for idx in audio_map.keys() if idx in trans_map]
    audio_paths = [audio_map[idx] for idx in common_ids]
    ref_texts = [trans_map[idx] for idx in common_ids]
    return common_ids, audio_paths, ref_texts

def main():
    start_time = time.time()
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("="*60)
    print("FunASR 批量测试与评估")
    print(f"脚本目录: {script_dir}")
    print(f"数据目录: {os.path.join(script_dir, DATA_DIR)}")
    print(f"模型: {MODEL_NAME}, 设备: {DEVICE}, Batch Size: {BATCH_SIZE}")
    print("="*60)

    # 检查文件是否存在
    if not os.path.exists(WAV_SCP):
        print(f"错误：找不到 {WAV_SCP}")
        print("请确保 data/ 文件夹下有 wav.scp 和 trans.txt")
        return
    if not os.path.exists(TRANS_FILE):
        print(f"错误：找不到 {TRANS_FILE}")
        return

    # 加载数据
    ids, audio_paths, ref_texts = load_file_pairs(WAV_SCP, TRANS_FILE)
    if not ids:
        print("错误：wav.scp 和 trans.txt 中没有匹配的 ID")
        return

    print(f"有效样本数: {len(ids)}")
    print(f"第一个音频: {audio_paths[0]}")

    # 检查第一个音频是否存在
    if not os.path.exists(audio_paths[0]):
        print(f"警告：音频文件不存在: {audio_paths[0]}")
        print("请检查 wav.scp 中的路径是否正确")

    # 加载模型
    print("正在加载模型...")
    model_kwargs = {
        "model": MODEL_NAME,
        "device": DEVICE,
    }
    if ENABLE_VAD:
        model_kwargs["vad_model"] = "fsmn-vad"
    if ENABLE_PUNC:
        model_kwargs["punc_model"] = "ct-punc"

    try:
        model = AutoModel(**model_kwargs)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 批量推理
    print("开始批量推理...")
    try:
        results = model.generate(input=audio_paths, batch_size=BATCH_SIZE)
    except Exception as e:
        print(f"推理过程中出错: {e}")
        return

    # 保存结果并收集假设文本
    hypotheses = []
    with open(OUTPUT_RESULT, "w", encoding="utf-8") as f:
        f.write("ID\t识别结果\t参考文本\n")
        for idx, res in zip(ids, results):
            text = rich_transcription_postprocess(res["text"])
            # 去除多余空格（保留中文连续）
            text = " ".join(text.split())
            hypotheses.append(text)
            ref = ref_texts[ids.index(idx)]
            f.write(f"{idx}\t{text}\t{ref}\n")

    print(f"识别完成，结果已保存至 {OUTPUT_RESULT}")

    # ========== 修改部分：清洗文本再计算 CER/WER ==========
    # 清洗：去除空格、标点，只保留中文、字母、数字
    ref_texts_clean = [clean_text(t) for t in ref_texts]
    hypotheses_clean = [clean_text(t) for t in hypotheses]

    overall_cer = cer(ref_texts_clean, hypotheses_clean) * 100
    overall_wer = wer(ref_texts_clean, hypotheses_clean) * 100
    # ===================================================

    elapsed = time.time() - start_time

    print("\n" + "="*60)
    print("评估结果")
    print(f"测试样本数: {len(ref_texts)}")
    print(f"字错误率 (CER): {overall_cer:.2f}%")
    print(f"词错误率 (WER): {overall_wer:.2f}%")
    print(f"总耗时: {elapsed:.2f} 秒")
    print("="*60)

    with open(OUTPUT_SUMMARY, "w", encoding="utf-8") as sf:
        sf.write(f"模型: {MODEL_NAME}\n")
        sf.write(f"设备: {DEVICE}, 批大小: {BATCH_SIZE}\n")
        sf.write(f"VAD: {ENABLE_VAD}, PUNC: {ENABLE_PUNC}\n")
        sf.write(f"测试样本数: {len(ref_texts)}\n")
        sf.write(f"CER: {overall_cer:.2f}%\n")
        sf.write(f"WER: {overall_wer:.2f}%\n")
        sf.write(f"耗时: {elapsed:.2f} 秒\n")
    print(f"摘要已保存至 {OUTPUT_SUMMARY}")

if __name__ == "__main__":
    main()