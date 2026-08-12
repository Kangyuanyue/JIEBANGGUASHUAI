#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
全量评估脚本：复杂度检测 + 拒识 + MossFormer2分离 + ASR → CER

覆盖 DatasetA + dataset_v1 全部样本，离线扫描阈值找最优配置

用法:
  cd yuyinshibie
  set HF_ENDPOINT=https://hf-mirror.com
  python scripts/eval_all.py

预计耗时: ~20-30 分钟 (RTX 4070, ~1838 样本)
"""

import os, sys, json, re, time
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
import torch

SR = 16000
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# ===== 数据集 =====
DATASETS = {
    "DatasetA":  r"C:\Users\Wuyc\Desktop\datasetA",
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output", "eval_all")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) + 1e-10) / (np.linalg.norm(b) + 1e-10))


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def clean_text(text):
    if not isinstance(text, str):
        return ''
    return re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)


def calc_cer(ref, hyp):
    r, h = clean_text(ref), clean_text(hyp)
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    m, n = len(r), len(h)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n] / m


def sweep_thresholds(records, output_dir):
    """离线阈值扫描，不需要模型"""
    reject_candidates = [round(x, 2) for x in np.arange(0.10, 0.50, 0.05)]
    sep_candidates = [round(x, 2) for x in np.arange(0.25, 0.60, 0.05)]

    pos_records = [r for r in records if r["label"] == "POS"]
    neg_records = [r for r in records if r["label"] == "NEG"]

    # 基线: 直接 ASR（不拒识不分高）
    baseline_cer = np.mean([r["cer_clean"] for r in pos_records]) * 100

    results = []
    for rej_th in reject_candidates:
        for sep_th in sep_candidates:
            if sep_th <= rej_th:
                continue

            pos_rej, pos_total = 0, 0
            pos_cer_list = []

            neg_rej, neg_total = 0, 0

            for r in pos_records:
                pos_total += 1
                if r["sim"] < rej_th:
                    pos_rej += 1
                elif r["sim"] < sep_th:
                    pos_cer_list.append(r["cer_sep"])
                else:
                    pos_cer_list.append(r["cer_clean"])

            for r in neg_records:
                neg_total += 1
                if r["sim"] < rej_th:
                    neg_rej += 1

            frr = pos_rej / max(pos_total, 1) * 100
            far = (neg_total - neg_rej) / max(neg_total, 1) * 100
            avg_cer = np.mean(pos_cer_list) * 100 if pos_cer_list else 0.0

            # 综合评分：CER 低 + FRR 低 + FAR 低 = 好，三项等权
            score = avg_cer + frr * 0.8 + far * 0.5

            results.append({
                "reject_th": rej_th, "sep_th": sep_th,
                "frr": round(frr, 2), "far": round(far, 2),
                "cer": round(avg_cer, 2),
                "pos_total": pos_total, "pos_rejected": pos_rej,
                "neg_total": neg_total, "neg_rejected": neg_rej,
                "score": round(score, 2),
            })

    results.sort(key=lambda x: x["score"])

    # 输出报告
    report_path = os.path.join(output_dir, "threshold_sweep.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Threshold Sweep Results\n")
        f.write(f"{'='*80}\n")
        f.write(f"Baseline (direct ASR, no reject): CER = {baseline_cer:.2f}%\n")
        f.write(f"Total POS={pos_total}, NEG={neg_total}\n\n")
        f.write(f"{'Rank':<6} {'RejTh':<8} {'SepTh':<8} {'FRR%':<8} {'FAR%':<8} {'CER%':<8} {'ΔCER':<8} {'Rej(P/N)':<14} {'Score':<8}\n")
        f.write(f"{'-'*80}\n")
        for i, r in enumerate(results[:25]):
            delta_cer = r["cer"] - baseline_cer
            f.write(f"{i+1:<6} {r['reject_th']:<8.2f} {r['sep_th']:<8.2f} "
                    f"{r['frr']:<8.2f} {r['far']:<8.2f} {r['cer']:<8.2f} "
                    f"{delta_cer:<+8.2f} "
                    f"{r['pos_rejected']}/{r['neg_rejected']:<12} "
                    f"{r['score']:<8.2f}\n")

        # 推荐
        f.write(f"\n{'='*80}\n")
        f.write(f"RECOMMENDED CONFIGURATIONS\n")
        f.write(f"{'='*80}\n\n")

        # 找三类最优
        best_cer = min(results, key=lambda x: x["cer"])
        best_frr = min(results, key=lambda x: x["frr"])
        # 均衡最优: FAR<20, FRR<15
        balanced = [r for r in results if r["far"] < 20 and r["frr"] < 15]
        best_balanced = balanced[0] if balanced else results[0]

        f.write(f"1. Best CER (最低识别错误):\n")
        f.write(f"   REJECT_THRESHOLD = {best_cer['reject_th']}, SEP_THRESHOLD = {best_cer['sep_th']}\n")
        f.write(f"   CER={best_cer['cer']}%  FRR={best_cer['frr']}%  FAR={best_cer['far']}%\n\n")

        f.write(f"2. Balanced (均衡):\n")
        f.write(f"   REJECT_THRESHOLD = {best_balanced['reject_th']}, SEP_THRESHOLD = {best_balanced['sep_th']}\n")
        f.write(f"   CER={best_balanced['cer']}%  FRR={best_balanced['frr']}%  FAR={best_balanced['far']}%\n\n")

        f.write(f"3. Best FRR (最低误拒):\n")
        f.write(f"   REJECT_THRESHOLD = {best_frr['reject_th']}, SEP_THRESHOLD = {best_frr['sep_th']}\n")
        f.write(f"   CER={best_frr['cer']}%  FRR={best_frr['frr']}%  FAR={best_frr['far']}%\n")

        f.write(f"\nBaseline CER (no pipeline): {baseline_cer:.2f}%\n")

    print(f"\nThreshold sweep report saved to: {report_path}")
    return results


def main():
    t_start = time.time()

    from funasr import AutoModel
    from clearvoice import ClearVoice

    # ==================== 加载模型 ====================
    print("=" * 60)
    print("Loading models...")
    print("=" * 60)

    print("[1/3] ERes2NetV2 (speaker embedding)...")
    spk_model = AutoModel(
        model="iic/speech_eres2netv2_sv_zh-cn_16k-common",
        device="cuda", disable_update=True
    )
    print("  Done.")

    print("[2/3] Paraformer (ASR)...")
    asr_model = AutoModel(
        model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        device="cuda", disable_update=True
    )
    print("  Done.")

    print("[3/3] MossFormer2_SS_16K (speech separation)...")
    separator = ClearVoice(task='speech_separation', model_names=['MossFormer2_SS_16K'])
    print("  Done.")

    t_load = time.time()
    print(f"\nModels loaded in {t_load - t_start:.0f}s\n")

    # ==================== 主循环 ====================
    all_records = []
    total_samples = 0

    for ds_name, ds_dir in DATASETS.items():
        pos_path = os.path.join(ds_dir, "pos.jsonl")
        neg_path = os.path.join(ds_dir, "neg.jsonl")

        if not os.path.exists(pos_path):
            print(f"Skip {ds_name}: pos.jsonl not found at {pos_path}")
            continue

        pos = load_jsonl(pos_path)
        neg = load_jsonl(neg_path) if os.path.exists(neg_path) else []
        total_samples += len(pos) + len(neg)

        for label, samples in [("POS", pos), ("NEG", neg)]:
            print(f"\n{'─'*50}")
            print(f"{ds_name} / {label}: {len(samples)} samples")
            print(f"{'─'*50}")

            t_ds_start = time.time()

            for i, s in enumerate(samples):
                sid = s["id"]
                kws_path = os.path.join(ds_dir, s["唤醒音频"])
                cmd_path = os.path.join(ds_dir, s["识别音频"])
                ref_text = s["识别文本"]

                # ---- 声纹提取 & 相似度 ----
                kws_res = spk_model.generate(input=kws_path)
                kws_emb = kws_res[0]["spk_embedding"].squeeze().cpu().numpy()

                cmd_res = spk_model.generate(input=cmd_path)
                cmd_emb = cmd_res[0]["spk_embedding"].squeeze().cpu().numpy()
                sim = cosine(kws_emb, cmd_emb)

                # ---- Path A: 干净路径 CER ----
                asr_res = asr_model.generate(input=cmd_path)
                hyp_clean = (asr_res[0].get("text") or "") if asr_res else ""
                cer_clean = calc_cer(ref_text, hyp_clean)

                # ---- Path B: 分离路径 CER ----
                cer_sep = cer_clean  # 默认回退
                n_sep_outputs = 0
                try:
                    outputs = separator.call_io_mode(cmd_path)
                    if outputs and len(outputs) >= 2:
                        n_sep_outputs = len(outputs)
                        best_sim = -1.0
                        best_stream = None
                        for j in range(len(outputs)):
                            out_audio = outputs[j].squeeze()
                            if len(out_audio) < SR * 0.3:
                                continue
                            tmp = os.path.join(OUTPUT_DIR, "_tmp.wav")
                            sf.write(tmp, out_audio.astype(np.float32), SR)
                            sep_res = spk_model.generate(input=tmp)
                            sep_emb = sep_res[0]["spk_embedding"].squeeze().cpu().numpy()
                            s_sim = cosine(kws_emb, sep_emb)
                            if s_sim > best_sim:
                                best_sim = s_sim
                                best_stream = out_audio
                            if os.path.exists(tmp):
                                os.remove(tmp)
                        if best_stream is not None:
                            sep_path = os.path.join(OUTPUT_DIR, "_sep.wav")
                            sf.write(sep_path, best_stream.astype(np.float32), SR)
                            asr_res_sep = asr_model.generate(input=sep_path)
                            hyp_sep = (asr_res_sep[0].get("text") or "") if asr_res_sep else ""
                            cer_sep = calc_cer(ref_text, hyp_sep)
                            if os.path.exists(sep_path):
                                os.remove(sep_path)
                except Exception as e:
                    pass

                all_records.append({
                    "id": sid, "dataset": ds_name, "label": label,
                    "ref": ref_text,
                    "sim": round(sim, 6),
                    "cer_clean": round(cer_clean, 6),
                    "cer_sep": round(cer_sep, 6),
                    "hyp_clean": hyp_clean,
                    "n_sep_outputs": n_sep_outputs,
                })

                # 进度
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - t_ds_start
                    eta = elapsed / (i + 1) * (len(samples) - i - 1)
                    # 当前 batch 的平均 cer
                    batch_clean = np.mean([r["cer_clean"] for r in all_records[-100:]])
                    batch_sep = np.mean([r["cer_sep"] for r in all_records[-100:]])
                    print(f"  [{i+1}/{len(samples)}] "
                          f"sim_avg={np.mean([r['sim'] for r in all_records[-100:]]):.3f}, "
                          f"cer_clean={batch_clean:.4f}, cer_sep={batch_sep:.4f} | "
                          f"elapsed={elapsed:.0f}s, ETA={eta:.0f}s")

            elapsed_ds = time.time() - t_ds_start
            print(f"  Done in {elapsed_ds:.0f}s ({elapsed_ds/60:.1f}min)")

    # ==================== 保存原始数据 ====================
    records_path = os.path.join(OUTPUT_DIR, "all_records.json")
    with open(records_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)
    print(f"\nRecords saved to: {records_path} ({len(all_records)} samples)")

    # ==================== 离线阈值扫描 ====================
    sweep_thresholds(all_records, OUTPUT_DIR)

    # ==================== 总耗时 ====================
    t_total = time.time() - t_start
    print(f"\nTotal time: {t_total/60:.1f} min")

    # ==================== 简要统计 ====================
    pos = [r for r in all_records if r["label"] == "POS"]
    neg = [r for r in all_records if r["label"] == "NEG"]
    print(f"\nQuick stats:")
    print(f"  POS sim: avg={np.mean([r['sim'] for r in pos]):.4f}, "
          f"min={np.min([r['sim'] for r in pos]):.4f}, "
          f"max={np.max([r['sim'] for r in pos]):.4f}")
    print(f"  NEG sim: avg={np.mean([r['sim'] for r in neg]):.4f}, "
          f"min={np.min([r['sim'] for r in neg]):.4f}, "
          f"max={np.max([r['sim'] for r in neg]):.4f}")
    print(f"  POS CER (clean):  {np.mean([r['cer_clean'] for r in pos])*100:.2f}%")
    print(f"  POS CER (sep):    {np.mean([r['cer_sep'] for r in pos])*100:.2f}%")
    print(f"  NEG CER (clean):  {np.mean([r['cer_clean'] for r in neg])*100:.2f}%")

    print(f"\nDone. All outputs in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
