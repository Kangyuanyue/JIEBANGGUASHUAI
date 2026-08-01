#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CAM++ vs ERes2NetV2 — 真正拒识能力对比 (对齐官方评估协议)

DatasetA 评估协议:
  - pos: "识别文本"=label, 测 CER (识别准确率)
  - neg: "识别文本"=null,  测 RR (拒识句准)

本脚本评估声纹拒识:
  1. 对每条数据的 kws(唤醒词) 和 cmd(指令) 分别提取声纹
  2. 计算余弦相似度
  3. pos 样本 → 相似度应该高 (同一个人 → 接受)
  4. neg 样本 → 相似度应该低 (非目标人/干扰 → 拒绝)
  5. 输出 EER, AUC, RR(句准)@阈值

核心指标 RR (Rejection Rate):
  RR = neg 中被正确拒绝的样本数 / neg 总样本数
  即: 相似度 < 阈值 → 拒绝 → 正确
"""

import os, sys, json, time
import numpy as np

import torch
from funasr import AutoModel

# ===== 配置（按需修改）=====
DATASET_DIR = os.environ.get("DATASET_DIR", r"C:\Users\Wuyc\Desktop\datasetA")
POS_JSONL   = os.path.join(DATASET_DIR, "pos.jsonl")
NEG_JSONL   = os.path.join(DATASET_DIR, "neg.jsonl")
OUTPUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEG     = 0  # 0=使用全部 neg 样本


def cosine(a, b):
    """余弦相似度"""
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-10)
    return (a_norm * b_norm).sum(axis=-1)


def extract_embeddings(model, audio_paths, model_name, device):
    """批量提取声纹"""
    embs = []
    for i, path in enumerate(audio_paths):
        try:
            res = model.generate(input=path)
            emb = res[0]["spk_embedding"].squeeze().cpu().numpy()
            embs.append(emb)
        except Exception as e:
            print(f"  [{model_name}] ERROR on {path}: {e}")
            embs.append(np.zeros(192))  # fallback
        if (i + 1) % 400 == 0:
            print(f"  [{model_name}] {i+1}/{len(audio_paths)}")
    return np.array(embs)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def compute_eer_auc(scores_pos, scores_neg):
    """计算 EER 和 AUC"""
    all_scores = np.concatenate([scores_pos, scores_neg])
    all_labels = np.concatenate([
        np.ones(len(scores_pos)),
        np.zeros(len(scores_neg))
    ])

    # 按分数降序排列（高分=正类）
    idx = np.argsort(all_scores)[::-1]
    labels_sorted = all_labels[idx]

    # TPR 和 FPR
    tpr = np.cumsum(labels_sorted) / len(scores_pos)
    fpr = np.cumsum(1 - labels_sorted) / len(scores_neg)

    # AUC = trapz(TPR, FPR)
    auc = np.trapz(tpr, fpr)

    # EER: 1 - TPR == FPR 的交点
    far = fpr
    frr = 1 - tpr
    diff = np.abs(far - frr)
    eer_idx = np.argmin(diff)
    eer = (far[eer_idx] + frr[eer_idx]) / 2 * 100
    threshold = all_scores[idx[eer_idx]]

    # Recall at specific FAR levels
    recall_at_far = {}
    for target_far in [0.05, 0.10, 0.20]:
        # 找到对应的 FPR 阈值
        far_idx = np.searchsorted(far, target_far)
        if far_idx < len(tpr):
            recall_at_far[target_far] = tpr[far_idx] * 100
        else:
            recall_at_far[target_far] = 0.0

    return eer, threshold, auc, recall_at_far


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t_total = time.time()

    # Load data
    pos_samples = load_jsonl(POS_JSONL)
    neg_samples = load_jsonl(NEG_JSONL)
    if MAX_NEG > 0:
        neg_samples = neg_samples[:MAX_NEG]

    print(f"Pos samples: {len(pos_samples)}")
    print(f"Neg samples: {len(neg_samples)}")

    # Build audio paths
    pos_kws = [os.path.join(DATASET_DIR, s["唤醒音频"]) for s in pos_samples]
    pos_cmd = [os.path.join(DATASET_DIR, s["识别音频"]) for s in pos_samples]
    neg_kws = [os.path.join(DATASET_DIR, s["唤醒音频"]) for s in neg_samples]
    neg_cmd = [os.path.join(DATASET_DIR, s["识别音频"]) for s in neg_samples]

    # ================================================================
    # Load models
    # ================================================================
    print("\n" + "=" * 60)
    print("Loading models...")
    print("=" * 60)

    print("\n[1] Loading CAM++ ...")
    cam_model = AutoModel(model="cam++", device=DEVICE, disable_update=True)

    print("\n[2] Loading ERes2NetV2 ...")
    eres_model = AutoModel(model="iic/speech_eres2netv2_sv_zh-cn_16k-common",
                           device=DEVICE, disable_update=True)

    results = {}

    for model, name in [(cam_model, "CAM++"), (eres_model, "ERes2NetV2")]:
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")

        # Extract POS embeddings
        print(f"\n  POS KWS ({len(pos_kws)} files):")
        t0 = time.time()
        pos_kws_emb = extract_embeddings(model, pos_kws, name, DEVICE)
        t_pos_kws = time.time() - t0
        print(f"    Time: {t_pos_kws:.1f}s")

        print(f"  POS CMD ({len(pos_cmd)} files):")
        t0 = time.time()
        pos_cmd_emb = extract_embeddings(model, pos_cmd, name, DEVICE)
        t_pos_cmd = time.time() - t0
        print(f"    Time: {t_pos_cmd:.1f}s")

        # Extract NEG embeddings
        print(f"  NEG KWS ({len(neg_kws)} files):")
        t0 = time.time()
        neg_kws_emb = extract_embeddings(model, neg_kws, name, DEVICE)
        t_neg_kws = time.time() - t0
        print(f"    Time: {t_neg_kws:.1f}s")

        print(f"  NEG CMD ({len(neg_cmd)} files):")
        t0 = time.time()
        neg_cmd_emb = extract_embeddings(model, neg_cmd, name, DEVICE)
        t_neg_cmd = time.time() - t0
        print(f"    Time: {t_neg_cmd:.1f}s")

        # Compute similarities
        print(f"\n  Computing similarities...")
        # POS: kws_i vs cmd_i → should ACCEPT
        pos_sims = np.array([
            cosine(pos_kws_emb[i], pos_cmd_emb[i])
            for i in range(len(pos_samples))
        ])

        # NEG: kws_i vs cmd_i → should REJECT
        neg_sims = np.array([
            cosine(neg_kws_emb[i], neg_cmd_emb[i])
            for i in range(len(neg_samples))
        ])

        # Metrics
        eer, thresh, auc, _ = compute_eer_auc(pos_sims, neg_sims)

        # ---- RR (Rejection Rate / 句准) 在不同阈值下的表现 ----
        print(f"\n  RR (拒识句准) 分析:")
        print(f"  {'Threshold':<14} {'pos通过率':<12} {'RR(neg拒绝率)':<16} {'FAR(neg误接受)':<16}")
        print(f"  {'-'*58}")

        for th in np.arange(0.05, 0.51, 0.05):
            pos_pass = (pos_sims >= th).mean() * 100
            neg_reject = (neg_sims < th).mean() * 100
            neg_false_accept = (neg_sims >= th).mean() * 100
            print(f"  {th:<14.2f} {pos_pass:<12.1f}% {neg_reject:<16.1f}% {neg_false_accept:<16.1f}%")

        total_time = t_pos_kws + t_pos_cmd + t_neg_kws + t_neg_cmd

        # EER threshold 处的 RR
        pos_pass_at_eer = (pos_sims >= thresh).mean() * 100
        rr_at_eer = (neg_sims < thresh).mean() * 100

        print(f"""
  --- {name} Results ---
  POS (应接受=高sim):  mean={pos_sims.mean():.4f}, std={pos_sims.std():.4f}
  NEG (应拒绝=低sim):  mean={neg_sims.mean():.4f}, std={neg_sims.std():.4f}
  Separation:          {pos_sims.mean() - neg_sims.mean():.4f}
  EER:                 {eer:.2f}%
  AUC:                 {auc:.4f}
  EER Threshold:       {thresh:.4f}
  pos通过率@EER阈值:    {pos_pass_at_eer:.1f}%
  RR(句准)@EER阈值:    {rr_at_eer:.1f}%
  Total Time:          {total_time:.1f}s
""")

        results[name] = {
            "pos_sims": pos_sims,
            "neg_sims": neg_sims,
            "eer": eer,
            "auc": auc,
            "threshold": thresh,
            "rr_at_eer": rr_at_eer,
            "pos_pass_at_eer": pos_pass_at_eer,
            "total_time": total_time,
            "pos_mean": pos_sims.mean(),
            "neg_mean": neg_sims.mean(),
        }

    # ================================================================
    # Final comparison
    # ================================================================
    cam = results["CAM++"]
    eres = results["ERes2NetV2"]

    winner = "ERes2NetV2" if eres["eer"] < cam["eer"] else "CAM++"

    summary = f"""
{'='*70}
  CAM++ vs ERes2NetV2 — 真正拒识能力对比 (pos vs neg)
{'='*70}

数据:
  pos (有效指令): {len(pos_samples)} 条
  neg (非指令):   {len(neg_samples)} 条

--- 相似度分布 ---
                      CAM++          ERes2NetV2      更好
  POS Mean (应高)     {cam['pos_mean']:.4f}          {eres['pos_mean']:.4f}          {'ERes2NetV2' if eres['pos_mean'] > cam['pos_mean'] else 'CAM++'}
  POS Std             {cam['pos_sims'].std():.4f}          {eres['pos_sims'].std():.4f}
  NEG Mean (应低)     {cam['neg_mean']:.4f}          {eres['neg_mean']:.4f}          {'ERes2NetV2' if eres['neg_mean'] < cam['neg_mean'] else 'CAM++'}
  NEG Std             {cam['pos_sims'].std():.4f}          {eres['neg_sims'].std():.4f}

--- 核心指标 ---
  EER (%)             {cam['eer']:.2f}%            {eres['eer']:.2f}%            {winner}
  AUC                 {cam['auc']:.4f}          {eres['auc']:.4f}          {'ERes2NetV2' if eres['auc'] > cam['auc'] else 'CAM++'}
  Separation Margin   {cam['pos_mean'] - cam['neg_mean']:.4f}          {eres['pos_mean'] - eres['neg_mean']:.4f}

--- 官方 RR(句准)@EER阈值 ---
  pos通过率            {cam['pos_pass_at_eer']:.1f}%             {eres['pos_pass_at_eer']:.1f}%
  RR(neg拒识率)       {cam['rr_at_eer']:.1f}%             {eres['rr_at_eer']:.1f}%

--- 速度 ---
  Total Time          {cam['total_time']:.1f}s           {eres['total_time']:.1f}s

=== 结论 ===
  EER Winner: {winner}
  RR Winner: {'ERes2NetV2' if eres['rr_at_eer'] > cam['rr_at_eer'] else 'CAM++'}
  推荐用于拒识: {winner}

  注意: 声纹拒识只能判断"是否同一个人说话"，
  无法判断"内容是否为有效指令"。
  如需内容层面的拒识，还需结合 WavLM/Whisper 等方案。
"""

    print(summary)

    # Save
    summary_path = os.path.join(OUTPUT_DIR, "compare_rejection_pos_neg.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"\nSummary saved to: {summary_path}")

    # Save raw data
    npz_path = os.path.join(OUTPUT_DIR, "compare_rejection_pos_neg.npz")
    np.savez_compressed(
        npz_path,
        cam_pos=cam["pos_sims"], cam_neg=cam["neg_sims"],
        eres_pos=eres["pos_sims"], eres_neg=eres["neg_sims"],
        cam_eer=cam["eer"], eres_eer=eres["eer"],
        cam_auc=cam["auc"], eres_auc=eres["auc"],
    )
    print(f"Raw data saved to: {npz_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
