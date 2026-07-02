#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估与测试报告生成模块
生成完整的测试报告：CER分析、拒识率、混淆矩阵、SNR鲁棒性、
推理耗时统计、错误案例分析、可视化图表
"""

import json
import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from collections import Counter, defaultdict
from typing import Dict, List

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src import config
from src.postprocess import compute_cer, normalize_text
from src.data_loader import load_audio, add_noise, estimate_snr

# ============================================================
# 设置中文字体
# ============================================================
def setup_chinese_font():
    """设置matplotlib中文字体"""
    font_candidates = [
        'SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong',
        'KaiTi', 'STHeiti', 'STSong', 'Arial Unicode MS'
    ]
    for font_name in font_candidates:
        try:
            fm.findfont(font_name, fallback_to_default=False)
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False
            print(f"[Font] 使用中文字体: {font_name}")
            return
        except:
            continue
    # 如果都找不到，用默认字体并忽略中文
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    print("[Font] 警告：未找到中文字体，图表中文可能显示异常")


setup_chinese_font()


def load_detailed_results(path: str = None) -> Dict:
    """加载推理详细结果"""
    if path is None:
        path = os.path.join(config.OUTPUT_DIR, "detailed_results.json")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_cer(pos_results: List[Dict]) -> Dict:
    """
    分析CER详情
    Returns:
        {
            'avg_cer': float,
            'median_cer': float,
            'cer_distribution': list,
            'perfect_count': int,
            'high_error_samples': list,
            ...
        }
    """
    cer_values = []
    for r in pos_results:
        label = r.get('label', '')
        pred = r.get('text', '')
        if label:
            cer = compute_cer(pred, label)
            cer_values.append({
                'id': r['id'],
                'label': label,
                'pred': pred,
                'cer': cer,
                'is_target': r.get('is_target', True),
                'similarity': r.get('similarity', 0),
                'snr_estimate': r.get('snr_estimate', 0)
            })

    if not cer_values:
        return {'avg_cer': 0, 'count': 0}

    cers = [v['cer'] for v in cer_values]

    # 分段统计
    perfect = sum(1 for c in cers if c == 0.0)
    low_error = sum(1 for c in cers if 0 < c <= 0.1)
    medium_error = sum(1 for c in cers if 0.1 < c <= 0.3)
    high_error = sum(1 for c in cers if c > 0.3)

    # 错误最多的样本
    sorted_by_cer = sorted(cer_values, key=lambda x: x['cer'], reverse=True)
    top_errors = sorted_by_cer[:20]

    # 被误拒的样本（目标人但被判为非目标人）
    false_rejects = [v for v in cer_values if not v['is_target']]

    return {
        'avg_cer': float(np.mean(cers)),
        'median_cer': float(np.median(cers)),
        'std_cer': float(np.std(cers)),
        'min_cer': float(np.min(cers)),
        'max_cer': float(np.max(cers)),
        'count': len(cers),
        'perfect_count': perfect,
        'low_error_count': low_error,
        'medium_error_count': medium_error,
        'high_error_count': high_error,
        'cer_values': cers,
        'top_errors': top_errors,
        'false_rejects': false_rejects,
        'all_details': cer_values
    }


def analyze_rejection(neg_results: List[Dict]) -> Dict:
    """分析拒识性能"""
    total = len(neg_results)
    correct_rejects = [r for r in neg_results if not r.get('is_target', True)]
    false_accepts = [r for r in neg_results if r.get('is_target', True)]

    rr = len(correct_rejects) / total if total > 0 else 0

    # 按唤醒词分组统计
    by_kws = defaultdict(lambda: {'total': 0, 'correct': 0})
    for r in neg_results:
        kws = r.get('kws_text', 'unknown')
        by_kws[kws]['total'] += 1
        if not r.get('is_target', True):
            by_kws[kws]['correct'] += 1

    kws_stats = {}
    for kws, stats in by_kws.items():
        kws_stats[kws] = {
            'total': stats['total'],
            'correct': stats['correct'],
            'rr': stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        }

    return {
        'total': total,
        'correct_rejects': len(correct_rejects),
        'false_accepts': len(false_accepts),
        'rejection_rate': rr,
        'false_accept_details': [{
            'id': r['id'],
            'similarity': r.get('similarity', 0),
            'pred_text': r.get('text', ''),
            'kws_text': r.get('kws_text', '')
        } for r in false_accepts],
        'by_kws': kws_stats
    }


def analyze_timing(timing_summary: Dict) -> Dict:
    """分析推理耗时"""
    return timing_summary


def plot_cer_distribution(cer_values: List[float], save_path: str):
    """绘制CER分布直方图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 直方图
    axes[0].hist(cer_values, bins=50, color='steelblue', edgecolor='white',
                 alpha=0.8)
    axes[0].set_xlabel('CER', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('CER Distribution (Histogram)', fontsize=14)
    axes[0].axvline(np.mean(cer_values), color='red', linestyle='--',
                    label=f'Mean={np.mean(cer_values):.4f}')
    axes[0].axvline(np.median(cer_values), color='orange', linestyle='--',
                    label=f'Median={np.median(cer_values):.4f}')
    axes[0].legend()

    # 累积分布
    sorted_cers = sorted(cer_values)
    cumulative = np.arange(1, len(sorted_cers) + 1) / len(sorted_cers)
    axes[1].plot(sorted_cers, cumulative, color='steelblue', linewidth=2)
    axes[1].set_xlabel('CER', fontsize=12)
    axes[1].set_ylabel('Cumulative Ratio', fontsize=12)
    axes[1].set_title('CER Cumulative Distribution', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    # 标注关键分位点
    for quantile in [0.5, 0.9, 0.95]:
        idx = int(quantile * len(sorted_cers))
        if idx < len(sorted_cers):
            axes[1].axhline(quantile, color='gray', linestyle=':', alpha=0.5)
            axes[1].axvline(sorted_cers[idx], color='gray', linestyle=':',
                           alpha=0.5)
            axes[1].annotate(f'{quantile*100:.0f}%: CER={sorted_cers[idx]:.3f}',
                           xy=(sorted_cers[idx], quantile),
                           fontsize=9, color='red')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [图表] CER分布图: {save_path}")


def plot_confusion_matrix(pos_results: List[Dict], neg_results: List[Dict],
                          save_path: str):
    """绘制二分类混淆矩阵（说话人验证：目标/非目标）"""
    # 真实标签 vs 预测标签
    # pos -> 实际是目标人
    # neg -> 实际是非目标人
    tp = sum(1 for r in pos_results if r.get('is_target', True))   # 正确识别
    fn = sum(1 for r in pos_results if not r.get('is_target', True))  # 误拒
    fp = sum(1 for r in neg_results if r.get('is_target', True))   # 误接
    tn = sum(1 for r in neg_results if not r.get('is_target', True))  # 正确拒识

    cm = np.array([[tp, fn], [fp, tn]])
    labels = ['Target\n(Pos)', 'Non-Target\n(Neg)']

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Pred: Target', 'Pred: Non-Target'],
                yticklabels=['Actual: Target', 'Actual: Non-Target'],
                ax=ax, annot_kws={'size': 16})
    ax.set_title('Speaker Verification Confusion Matrix', fontsize=14)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_xlabel('Predicted', fontsize=12)

    # 添加指标文本
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    rr = tn / (tn + fp) if (tn + fp) > 0 else 0

    stats_text = f'Precision: {precision:.4f}\nRecall: {recall:.4f}\n' \
                 f'F1-Score: {f1:.4f}\nRejection Rate: {rr:.4f}'
    ax.text(2.3, 0.5, stats_text, transform=ax.transData,
            fontsize=11, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [图表] 混淆矩阵: {save_path}")

    return {'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
            'precision': precision, 'recall': recall, 'f1': f1, 'rr': rr}


def plot_similarity_distribution(pos_results: List[Dict],
                                 neg_results: List[Dict],
                                 threshold: float,
                                 save_path: str):
    """绘制说话人相似度分布图"""
    pos_sims = [r.get('similarity', 0) for r in pos_results]
    neg_sims = [r.get('similarity', 0) for r in neg_results]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(pos_sims, bins=50, alpha=0.6, color='green',
            label=f'Target Speaker (n={len(pos_sims)})', density=True)
    ax.hist(neg_sims, bins=50, alpha=0.6, color='red',
            label=f'Non-Target Speaker (n={len(neg_sims)})', density=True)
    ax.axvline(threshold, color='black', linestyle='--', linewidth=2,
               label=f'Threshold={threshold:.2f}')
    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Speaker Similarity Distribution', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [图表] 相似度分布: {save_path}")


def plot_timing_breakdown(timing_summary: Dict, save_path: str):
    """绘制推理耗时分解图"""
    stages = ['speaker_verify', 'speech_enhance', 'asr_inference',
              'postprocess']
    stage_names = ['Speaker\nVerification', 'Speech\nEnhancement',
                   'ASR\nInference', 'Post\nProcessing']

    means = []
    for s in stages:
        if s in timing_summary:
            means.append(timing_summary[s]['mean'] * 1000)  # 转为ms
        else:
            means.append(0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 柱状图
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    bars = axes[0].bar(stage_names, means, color=colors, edgecolor='white')
    axes[0].set_ylabel('Time (ms)', fontsize=12)
    axes[0].set_title('Average Inference Time Breakdown', fontsize=14)
    for bar, val in zip(bars, means):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{val:.1f}ms', ha='center', fontsize=10)

    # 饼图
    total = sum(means)
    if total > 0:
        percentages = [m/total*100 for m in means]
        axes[1].pie(percentages, labels=stage_names, colors=colors,
                   autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
        axes[1].set_title(f'Time Proportion (Total: {total:.1f}ms)', fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [图表] 推理耗时: {save_path}")


def plot_snr_performance(pos_results: List[Dict], save_path: str):
    """绘制不同SNR条件下的CER性能曲线（基于估计SNR分箱）"""
    # 按估计SNR分组
    snr_cer_data = defaultdict(list)
    for r in pos_results:
        snr = r.get('snr_estimate', 20)
        label = r.get('label', '')
        pred = r.get('text', '')
        if label:
            cer = compute_cer(pred, label)
            # 分箱
            if snr < 0:
                snr_bin = '<0dB'
            elif snr < 5:
                snr_bin = '0-5dB'
            elif snr < 10:
                snr_bin = '5-10dB'
            elif snr < 20:
                snr_bin = '10-20dB'
            else:
                snr_bin = '>20dB'
            snr_cer_data[snr_bin].append(cer)

    # 排序
    bin_order = ['<0dB', '0-5dB', '5-10dB', '10-20dB', '>20dB']
    bins_present = [b for b in bin_order if b in snr_cer_data]
    avg_cers = [np.mean(snr_cer_data[b]) for b in bins_present]
    counts = [len(snr_cer_data[b]) for b in bins_present]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color1 = '#e74c3c'
    ax1.plot(bins_present, avg_cers, 'o-', color=color1, linewidth=2,
             markersize=8, label='Average CER')
    ax1.set_xlabel('Estimated SNR', fontsize=12)
    ax1.set_ylabel('Average CER', fontsize=12, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_title('CER vs Estimated SNR', fontsize=14)

    # 右轴显示样本数
    ax2 = ax1.twinx()
    color2 = '#3498db'
    ax2.bar(bins_present, counts, alpha=0.3, color=color2, label='Sample Count')
    ax2.set_ylabel('Sample Count', fontsize=12, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    ax1.grid(True, alpha=0.3)
    fig.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [图表] SNR性能: {save_path}")


def plot_kws_rejection_rate(rejection_analysis: Dict, save_path: str):
    """按唤醒词绘制拒识率"""
    by_kws = rejection_analysis.get('by_kws', {})
    if not by_kws:
        return

    kws_names = list(by_kws.keys())
    rr_values = [by_kws[k]['rr'] for k in kws_names]
    counts = [by_kws[k]['total'] for k in kws_names]

    # 按拒识率排序
    sorted_idx = sorted(range(len(kws_names)), key=lambda i: rr_values[i])
    kws_names = [kws_names[i] for i in sorted_idx]
    rr_values = [rr_values[i] for i in sorted_idx]
    counts = [counts[i] for i in sorted_idx]

    fig, ax = plt.subplots(figsize=(12, max(6, len(kws_names) * 0.35)))
    colors = ['#e74c3c' if v < 0.9 else '#2ecc71' for v in rr_values]
    bars = ax.barh(kws_names, rr_values, color=colors, edgecolor='white')

    for bar, val, cnt in zip(bars, rr_values, counts):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
               f'{val:.2f} (n={cnt})', va='center', fontsize=9)

    ax.set_xlabel('Rejection Rate', fontsize=12)
    ax.set_title('Rejection Rate by Wake Word', fontsize=14)
    ax.set_xlim(0, 1.15)
    ax.axvline(0.95, color='gray', linestyle='--', alpha=0.5, label='Target: 95%')
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [图表] 唤醒词拒识率: {save_path}")


def generate_report(detailed_results: Dict = None):
    """
    生成完整测试报告
    """
    print("\n" + "=" * 70)
    print("  生成测试报告")
    print("=" * 70)

    if detailed_results is None:
        detailed_results = load_detailed_results()

    pos_results = detailed_results['pos_results']
    neg_results = detailed_results['neg_results']
    metrics = detailed_results['metrics']
    timing_summary = detailed_results.get('timing_summary', {})
    memory_usage = detailed_results.get('memory_usage', {})
    model_summary = detailed_results.get('model_summary', {})

    # ============================================================
    # 1. CER分析
    # ============================================================
    print("\n[1/7] CER分析...")
    cer_analysis = analyze_cer(pos_results)

    # ============================================================
    # 2. 拒识分析
    # ============================================================
    print("[2/7] 拒识率分析...")
    rejection_analysis = analyze_rejection(neg_results)

    # ============================================================
    # 3. 生成图表
    # ============================================================
    print("[3/7] 生成可视化图表...")

    # CER分布
    if cer_analysis.get('cer_values'):
        plot_cer_distribution(
            cer_analysis['cer_values'],
            os.path.join(config.FIGURE_DIR, 'cer_distribution.png')
        )

    # 混淆矩阵
    cm_stats = plot_confusion_matrix(
        pos_results, neg_results,
        os.path.join(config.FIGURE_DIR, 'confusion_matrix.png')
    )

    # 相似度分布
    threshold = config.SPEAKER_SIMILARITY_THRESHOLD
    if model_summary and 'speaker_verifier' in model_summary:
        threshold = model_summary['speaker_verifier'].get('threshold', threshold)
    plot_similarity_distribution(
        pos_results, neg_results, threshold,
        os.path.join(config.FIGURE_DIR, 'similarity_distribution.png')
    )

    # 推理耗时
    if timing_summary:
        plot_timing_breakdown(
            timing_summary,
            os.path.join(config.FIGURE_DIR, 'timing_breakdown.png')
        )

    # SNR性能
    plot_snr_performance(
        pos_results,
        os.path.join(config.FIGURE_DIR, 'snr_performance.png')
    )

    # 唤醒词拒识率
    plot_kws_rejection_rate(
        rejection_analysis,
        os.path.join(config.FIGURE_DIR, 'kws_rejection_rate.png')
    )

    # ============================================================
    # 4. 生成文本报告
    # ============================================================
    print("[4/7] 生成文本报告...")

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("  复杂交互场景的抗干扰语音指令识别系统 - 测试报告")
    report_lines.append("=" * 70)

    # --- 总体指标 ---
    report_lines.append("\n📊 一、总体性能指标")
    report_lines.append("-" * 50)
    report_lines.append(f"  目标发音人识别字错率 (CER): {cer_analysis['avg_cer']:.4f} ({cer_analysis['avg_cer']*100:.2f}%)")
    report_lines.append(f"  识别准确率 (1-CER):          {(1-cer_analysis['avg_cer'])*100:.2f}%")
    report_lines.append(f"  拒识率 (RR):                  {rejection_analysis['rejection_rate']:.4f} ({rejection_analysis['rejection_rate']*100:.2f}%)")
    report_lines.append(f"  CER中位数:                    {cer_analysis['median_cer']:.4f}")
    report_lines.append(f"  CER标准差:                    {cer_analysis['std_cer']:.4f}")

    # --- 竞赛得分 ---
    cer_score = 0.4 * (1 - cer_analysis['avg_cer'])
    rr_score = 0.4 * rejection_analysis['rejection_rate']
    if timing_summary and 'total' in timing_summary:
        avg_time = timing_summary['total']['mean']
        eff_score = 0.2 * min(1.0, 2.0 / avg_time)
    else:
        eff_score = 0.2 * 0.5
    total_score = cer_score + rr_score + eff_score

    report_lines.append(f"\n📈 二、竞赛综合得分")
    report_lines.append("-" * 50)
    report_lines.append(f"  CER贡献 (40%):   {cer_score:.4f}")
    report_lines.append(f"  RR贡献 (40%):    {rr_score:.4f}")
    report_lines.append(f"  效率贡献 (20%):  {eff_score:.4f}")
    report_lines.append(f"  ━━━━━━━━━━━━━━━━━━━━━━━━")
    report_lines.append(f"  综合得分:         {total_score:.4f}")

    # --- CER详细分析 ---
    report_lines.append(f"\n📝 三、CER详细分析")
    report_lines.append("-" * 50)
    report_lines.append(f"  正样本总数:     {cer_analysis['count']}")
    report_lines.append(f"  完全正确 (CER=0): {cer_analysis['perfect_count']} ({cer_analysis['perfect_count']/max(1,cer_analysis['count'])*100:.1f}%)")
    report_lines.append(f"  低错误 (0<CER≤0.1): {cer_analysis['low_error_count']}")
    report_lines.append(f"  中错误 (0.1<CER≤0.3): {cer_analysis['medium_error_count']}")
    report_lines.append(f"  高错误 (CER>0.3): {cer_analysis['high_error_count']}")

    # --- 说话人验证性能 ---
    report_lines.append(f"\n🔐 四、说话人验证性能")
    report_lines.append("-" * 50)
    report_lines.append(f"  True Positive (正确识别目标人):  {cm_stats['tp']}")
    report_lines.append(f"  False Negative (误拒目标人):     {cm_stats['fn']}")
    report_lines.append(f"  False Positive (误接非目标人):   {cm_stats['fp']}")
    report_lines.append(f"  True Negative (正确拒识):        {cm_stats['tn']}")
    report_lines.append(f"  Precision: {cm_stats['precision']:.4f}")
    report_lines.append(f"  Recall:    {cm_stats['recall']:.4f}")
    report_lines.append(f"  F1-Score:  {cm_stats['f1']:.4f}")

    # --- 拒识详情 ---
    report_lines.append(f"\n🚫 五、拒识性能详情")
    report_lines.append("-" * 50)
    report_lines.append(f"  非目标样本总数:   {rejection_analysis['total']}")
    report_lines.append(f"  正确拒识:         {rejection_analysis['correct_rejects']}")
    report_lines.append(f"  误接受:           {rejection_analysis['false_accepts']}")
    report_lines.append(f"  拒识率:           {rejection_analysis['rejection_rate']:.4f}")

    if rejection_analysis['false_accept_details']:
        report_lines.append(f"\n  误接受样本详情:")
        for fa in rejection_analysis['false_accept_details'][:10]:
            report_lines.append(f"    ID={fa['id']}, "
                              f"Sim={fa['similarity']:.4f}, "
                              f"KWS='{fa['kws_text']}', "
                              f"Pred='{fa['pred_text']}'")

    # --- 推理耗时 ---
    report_lines.append(f"\n⏱ 六、推理耗时统计")
    report_lines.append("-" * 50)
    if timing_summary:
        for stage, stats in timing_summary.items():
            report_lines.append(f"  {stage}:")
            report_lines.append(f"    平均: {stats['mean']*1000:.1f}ms | "
                              f"最小: {stats['min']*1000:.1f}ms | "
                              f"最大: {stats['max']*1000:.1f}ms | "
                              f"标准差: {stats['std']*1000:.1f}ms")
        if 'total' in timing_summary:
            total_time = timing_summary['total']['total']
            count = timing_summary['total']['count']
            throughput = count / total_time if total_time > 0 else 0
            report_lines.append(f"\n  总推理时间: {total_time:.2f}s")
            report_lines.append(f"  平均每条: {total_time/count*1000:.1f}ms")
            report_lines.append(f"  吞吐量: {throughput:.2f} 条/秒")

    # --- 内存与模型 ---
    report_lines.append(f"\n💾 七、模型与内存统计")
    report_lines.append("-" * 50)
    if memory_usage:
        report_lines.append(f"  RSS内存: {memory_usage.get('rss_mb', 0):.1f} MB")
        report_lines.append(f"  VMS内存: {memory_usage.get('vms_mb', 0):.1f} MB")
        report_lines.append(f"  内存占比: {memory_usage.get('percent', 0):.1f}%")
    if model_summary:
        for module, info in model_summary.items():
            report_lines.append(f"  {module}: {json.dumps(info, ensure_ascii=False)}")

    # --- 错误案例 ---
    report_lines.append(f"\n❌ 八、错误案例分析 (Top-20)")
    report_lines.append("-" * 50)
    for i, err in enumerate(cer_analysis.get('top_errors', [])[:20]):
        report_lines.append(f"  [{i+1}] ID={err['id']}, CER={err['cer']:.4f}")
        report_lines.append(f"      标签: {err['label']}")
        report_lines.append(f"      预测: {err['pred']}")
        report_lines.append(f"      相似度={err['similarity']:.4f}, "
                          f"SNR≈{err['snr_estimate']:.1f}dB")

    # --- 误拒案例 ---
    false_rejects = cer_analysis.get('false_rejects', [])
    if false_rejects:
        report_lines.append(f"\n⚠ 九、误拒案例（目标人被错误拒识）")
        report_lines.append("-" * 50)
        for fr in false_rejects[:10]:
            report_lines.append(f"  ID={fr['id']}, Sim={fr['similarity']:.4f}, "
                              f"Label='{fr['label']}'")

    report_lines.append("\n" + "=" * 70)
    report_lines.append("  报告生成完毕")
    report_lines.append("=" * 70)

    # 保存报告
    report_text = '\n'.join(report_lines)
    report_path = os.path.join(config.REPORT_DIR, 'test_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n  文本报告: {report_path}")

    # 打印报告
    print(report_text)

    return {
        'cer_analysis': cer_analysis,
        'rejection_analysis': rejection_analysis,
        'cm_stats': cm_stats,
        'total_score': total_score
    }


def main():
    """主函数"""
    generate_report()


if __name__ == "__main__":
    main()
