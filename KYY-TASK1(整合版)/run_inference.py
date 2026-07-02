#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
主推理入口
加载测试集A，运行完整Pipeline，输出竞赛要求的JSON格式结果
"""

import json
import os
import sys
import time
import argparse

# 确保项目根目录在Python路径中
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src import config
from src.data_loader import load_dataset
from src.pipeline import InferencePipeline
from src.postprocess import compute_cer, normalize_text


def run_inference(speaker_threshold: float = None,
                  asr_engine: str = None,
                  enable_enhance: bool = None):
    """
    运行完整推理流程
    Args:
        speaker_threshold: 说话人验证阈值
        asr_engine: ASR引擎类型
        enable_enhance: 是否启用语音增强
    """
    print("=" * 70)
    print("  复杂交互场景的抗干扰语音指令识别系统 - 推理")
    print("=" * 70)

    # ============================================================
    # 1. 加载数据集
    # ============================================================
    print("\n[Step 1] 加载数据集...")
    pos_data = load_dataset(config.POS_JSONL, config.DATASET_DIR)
    neg_data = load_dataset(config.NEG_JSONL, config.DATASET_DIR)
    print(f"  正样本(pos): {len(pos_data)} 条")
    print(f"  拒识样本(neg): {len(neg_data)} 条")
    print(f"  总计: {len(pos_data) + len(neg_data)} 条")

    # ============================================================
    # 2. 初始化Pipeline
    # ============================================================
    print("\n[Step 2] 初始化Pipeline...")
    pipeline = InferencePipeline(
        speaker_threshold=speaker_threshold,
        asr_engine=asr_engine,
        enable_enhance=enable_enhance
    )

    # ============================================================
    # 3. 处理正样本（需要识别文本）
    # ============================================================
    print("\n[Step 3] 处理正样本（目标发音人语音识别）...")
    total_start = time.time()
    pipeline.is_positive_mode = True
    pos_results = pipeline.process_dataset(pos_data, progress_interval=100)

    # ============================================================
    # 4. 处理负样本（需要拒识）
    # ============================================================
    print("\n[Step 4] 处理负样本（非目标发音人拒识）...")
    pipeline.is_positive_mode = False
    neg_results = pipeline.process_dataset(neg_data, progress_interval=100)

    total_duration = time.time() - total_start

    # ============================================================
    # 5. 计算评估指标
    # ============================================================
    print("\n[Step 5] 计算评估指标...")

    # 正样本CER
    cer_list = []
    for r in pos_results:
        label = r.get('label', '')
        if label:
            cer = compute_cer(r['text'], label)
            cer_list.append(cer)
            r['cer'] = cer
        else:
            r['cer'] = 0.0

    avg_cer = sum(cer_list) / len(cer_list) if cer_list else 0.0

    # 负样本拒识率
    correct_reject = sum(1 for r in neg_results if not r['is_target'])
    total_neg = len(neg_results)
    rr = correct_reject / total_neg if total_neg > 0 else 0.0

    # ============================================================
    # 6. 生成竞赛格式JSON
    # ============================================================
    print("\n[Step 6] 生成提交结果...")

    submission_results = []

    # 正样本结果
    for r in pos_results:
        submission_results.append({
            "id": f"pos/cmd_{r['id']}",
            "content": r['text'],
            "label": r.get('label', ''),
            "cer": f"{r['cer']:.4f}"
        })

    # 负样本结果
    for r in neg_results:
        pred_text = r['text'] if r['is_target'] else ""
        submission_results.append({
            "id": f"neg/cmd_{r['id']}",
            "content": pred_text,
            "label": "",
            "cer": "0.0000" if not pred_text else "1.0000"
        })

    submission = {
        "result": {
            "results": submission_results,
            "final_cer": f"{avg_cer:.4f}",
            "duration": f"{total_duration:.2f}"
        }
    }

    # 保存提交文件
    with open(config.RESULT_JSON, 'w', encoding='utf-8') as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)
    print(f"  竞赛提交文件: {config.RESULT_JSON}")

    # ============================================================
    # 7. 保存详细结果（供评估模块使用）
    # ============================================================
    detailed_results = {
        'pos_results': pos_results,
        'neg_results': neg_results,
        'metrics': {
            'avg_cer': avg_cer,
            'rejection_rate': rr,
            'correct_reject': correct_reject,
            'total_neg': total_neg,
            'total_duration': total_duration
        },
        'timing_summary': pipeline.get_timing_summary(),
        'memory_usage': pipeline.get_memory_usage(),
        'model_summary': pipeline.get_model_summary()
    }

    detail_path = os.path.join(config.OUTPUT_DIR, "detailed_results.json")
    with open(detail_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2,
                  default=str)
    print(f"  详细结果文件: {detail_path}")

    # ============================================================
    # 8. 打印摘要
    # ============================================================
    print("\n" + "=" * 70)
    print("  推理结果摘要")
    print("=" * 70)
    print(f"  正样本平均CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
    print(f"  识别准确率 (1-CER): {(1-avg_cer)*100:.2f}%")
    print(f"  拒识率 (RR): {rr:.4f} ({rr*100:.2f}%)")
    print(f"  正确拒识: {correct_reject}/{total_neg}")
    print(f"  总推理时间: {total_duration:.2f}s")
    print(f"  平均推理时间: {total_duration/(len(pos_data)+len(neg_data)):.3f}s/条")

    # 综合得分估算
    # Score = 0.4 × (1-CER) + 0.4 × RR + 0.2 × 效率分
    efficiency_score = min(1.0, 2.0 / (total_duration / (len(pos_data)+len(neg_data))))
    estimated_score = 0.4 * (1 - avg_cer) + 0.4 * rr + 0.2 * efficiency_score
    print(f"\n  综合得分估算: {estimated_score:.4f}")
    print(f"    CER贡献: {0.4*(1-avg_cer):.4f}")
    print(f"    RR贡献:  {0.4*rr:.4f}")
    print(f"    效率贡献: {0.2*efficiency_score:.4f}")
    print("=" * 70)

    return detailed_results


def main():
    parser = argparse.ArgumentParser(
        description='复杂交互场景的抗干扰语音指令识别系统 - 推理')
    parser.add_argument('--threshold', type=float, default=None,
                        help='说话人验证阈值 (默认使用config配置)')
    parser.add_argument('--asr', type=str, default=None,
                        choices=['funasr', 'whisper'],
                        help='ASR引擎类型')
    parser.add_argument('--enhance', action='store_true', default=None,
                        help='启用语音增强')
    parser.add_argument('--no-enhance', dest='enhance', action='store_false',
                        help='禁用语音增强')
    args = parser.parse_args()

    run_inference(
        speaker_threshold=args.threshold,
        asr_engine=args.asr,
        enable_enhance=args.enhance
    )


if __name__ == "__main__":
    main()
