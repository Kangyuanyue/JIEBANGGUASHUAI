"""
speech_campplus_sv_zh-cn_16k-common 声纹验证测试

测试在 datasetA/neg 上的拒识率 (RR):
  1. 从 kws 音频提取目标说话人声纹
  2. 从 cmd 音频提取说话人声纹
  3. 计算余弦相似度
  4. 若 similarity < threshold → 拒识成功
"""

import os, sys, json, io
from pathlib import Path

_seen = set()
_o = os.add_dll_directory
os.add_dll_directory = lambda p: _o(p) if p not in _seen and not _seen.add(p) else type(
    "_D", (), {"close": lambda _: None, "__enter__": lambda s: s, "__exit__": lambda *_: None}
)()
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
from sklearn.metrics import roc_curve, auc

BASE = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE / "dataset" / "datasetA"
NEG_JSONL = DATASET_DIR / "neg.jsonl"
POS_JSONL = DATASET_DIR / "pos.jsonl"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.flatten()
    b = b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def main():
    print("=" * 55)
    print("  CAM++ 声纹验证: datasetA 拒识率测试")
    print("=" * 55)

    # 加载声纹模型
    print("\n[加载] speech_campplus_sv_zh-cn_16k-common ...")
    from funasr import AutoModel

    sv_model = AutoModel(
        model="iic/speech_campplus_sv_zh-cn_16k-common",
        # CAM++ 不需要 vad/punc
    )
    print("[完成] 模型加载成功\n")

    # --- 负样本测试 (RR) ---
    print("[测试] 负样本 (neg) 拒识率测试 ...")
    with open(NEG_JSONL, "r", encoding="utf-8") as f:
        neg_samples = [json.loads(l) for l in f if l.strip()]

    neg_results = []
    for i, item in enumerate(neg_samples):
        kws_path = str(DATASET_DIR / item["唤醒音频"])
        cmd_path = str(DATASET_DIR / item["识别音频"])

        # 提取 kws 声纹
        kws_emb = sv_model.generate(input=kws_path)
        kws_vec = np.array(kws_emb[0]["spk_embedding"], dtype=np.float32)

        # 提取 cmd 声纹
        cmd_emb = sv_model.generate(input=cmd_path)
        cmd_vec = np.array(cmd_emb[0]["spk_embedding"], dtype=np.float32)

        sim = cosine_similarity(kws_vec, cmd_vec)
        neg_results.append({"id": item["id"], "similarity": sim})

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(neg_samples)}]")

    # --- 正样本测试 (参考对比) ---
    print("\n[测试] 正样本 (pos) 声纹相似度参考 ...")
    with open(POS_JSONL, "r", encoding="utf-8") as f:
        pos_samples = [json.loads(l) for l in f if l.strip()]

    # 抽检前 200 条正样本 (id <= 363 的干净子集)
    pos_results = []
    pos_test = [s for s in pos_samples if s["id"] <= 363][:200]
    for i, item in enumerate(pos_test):
        kws_path = str(DATASET_DIR / item["唤醒音频"])
        cmd_path = str(DATASET_DIR / item["识别音频"])

        kws_emb = sv_model.generate(input=kws_path)
        kws_vec = np.array(kws_emb[0]["spk_embedding"], dtype=np.float32)

        cmd_emb = sv_model.generate(input=cmd_path)
        cmd_vec = np.array(cmd_emb[0]["spk_embedding"], dtype=np.float32)

        sim = cosine_similarity(kws_vec, cmd_vec)
        pos_results.append({"id": item["id"], "similarity": sim})

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(pos_test)}]")

    # --- 统计 ---
    neg_sims = [r["similarity"] for r in neg_results]
    pos_sims = [r["similarity"] for r in pos_results]

    print(f"\n{'='*55}")
    print(f"  声纹相似度统计")
    print(f"{'='*55}")
    print(f"  负样本 (neg): n={len(neg_sims)}")
    print(f"    均值: {np.mean(neg_sims):.4f}")
    print(f"    中位数: {np.median(neg_sims):.4f}")
    print(f"    标准差: {np.std(neg_sims):.4f}")
    print(f"    最大: {max(neg_sims):.4f}")
    print(f"    最小: {min(neg_sims):.4f}")

    print(f"\n  正样本 (pos): n={len(pos_sims)}")
    print(f"    均值: {np.mean(pos_sims):.4f}")
    print(f"    中位数: {np.median(pos_sims):.4f}")
    print(f"    标准差: {np.std(pos_sims):.4f}")
    print(f"    最大: {max(pos_sims):.4f}")
    print(f"    最小: {min(pos_sims):.4f}")

    # --- RR 扫描 ---
    print(f"\n{'='*55}")
    print(f"  拒识率 (RR) 扫描 (threshold 从 0.1 到 0.9)")
    print(f"{'='*55}")
    print(f"  {'threshold':<12} {'RR(neg)':<12} {'误拒(pos)':<12} {'说明':<16}")
    print(f"  {'-'*55}")
    for t in [x / 10 for x in range(1, 10)]:
        r_neg = sum(1 for s in neg_sims if s < t) / len(neg_sims)
        r_pos = sum(1 for s in pos_sims if s < t) / len(pos_sims)
        desc = ""
        if r_neg > 0.95 and r_pos < 0.05:
            desc = "★ 推荐"
        elif r_neg > 0.90 and r_pos < 0.10:
            desc = "良好"
        print(f"  {t:<12.1f} {r_neg:<12.2%} {r_pos:<12.2%} {desc}")

    # --- EER ---
    labels = [1] * len(pos_sims) + [0] * len(neg_sims)
    scores = pos_sims + neg_sims
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    print(f"\n  EER (等错误率): {eer:.2%}")
    print(f"  AUC: {auc(fpr, tpr):.4f}")

    # --- 保存结果 ---
    output = BASE / "outputs" / "sv_eval_result.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump({
            "neg": neg_results,
            "pos": pos_results,
            "neg_stats": {"mean": round(float(np.mean(neg_sims)), 4),
                          "median": round(float(np.median(neg_sims)), 4),
                          "std": round(float(np.std(neg_sims)), 4)},
            "pos_stats": {"mean": round(float(np.mean(pos_sims)), 4),
                          "median": round(float(np.median(pos_sims)), 4),
                          "std": round(float(np.std(pos_sims)), 4)},
            "eer": round(float(eer), 4),
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[保存] {output}")


if __name__ == "__main__":
    main()
