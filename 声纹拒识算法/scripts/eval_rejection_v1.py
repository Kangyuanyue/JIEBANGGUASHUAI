#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CAM++ vs ERes2NetV2 拒识对比 — dataset_v1 版本
用法:
  C:\ProgramData\anaconda3\envs\funasr_env\python.exe scripts/compare_rejection_v1.py
"""

import os, sys, json, time
import numpy as np

import torch
from funasr import AutoModel

DATASET_DIR = os.environ.get("DATASET_V1_DIR", r"C:\Users\Wuyc\Desktop\dataset_v1\dataset_v1")
POS_JSONL   = os.path.join(DATASET_DIR, "pos.jsonl")
NEG_JSONL   = os.path.join(DATASET_DIR, "neg.jsonl")
OUTPUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def extract_embeddings(model, audio_paths, model_name):
    embs = []
    for i, path in enumerate(audio_paths):
        try:
            res = model.generate(input=path)
            emb = res[0]["spk_embedding"].squeeze().cpu().numpy()
            embs.append(emb)
        except Exception as e:
            print(f"  [{model_name}] ERROR on {path}: {e}")
            embs.append(np.zeros(192))
        if (i + 1) % 300 == 0:
            print(f"  [{model_name}] {i+1}/{len(audio_paths)}")
    return np.array(embs)


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pos_samples = load_jsonl(POS_JSONL)
    neg_samples = load_jsonl(NEG_JSONL)
    print(f"pos={len(pos_samples)}, neg={len(neg_samples)}")

    pos_kws = [os.path.join(DATASET_DIR, s["唤醒音频"]) for s in pos_samples]
    pos_cmd = [os.path.join(DATASET_DIR, s["识别音频"]) for s in pos_samples]
    neg_kws = [os.path.join(DATASET_DIR, s["唤醒音频"]) for s in neg_samples]
    neg_cmd = [os.path.join(DATASET_DIR, s["识别音频"]) for s in neg_samples]

    print("\nLoading CAM++ ...")
    cam = AutoModel(model="cam++", device=DEVICE, disable_update=True)
    print("Loading ERes2NetV2 ...")
    eres = AutoModel(model="iic/speech_eres2netv2_sv_zh-cn_16k-common",
                     device=DEVICE, disable_update=True)

    lines = []
    lines.append(f"dataset_v1: pos={len(pos_samples)}, neg={len(neg_samples)}")
    lines.append("")

    for model, name in [(cam, "CAM++"), (eres, "ERes2NetV2")]:
        print(f"\n{'='*50}")
        print(f"Evaluating: {name}")
        print(f"{'='*50}")

        t0 = time.time()

        print(f"Extracting POS KWS ({len(pos_kws)} files)...")
        pk = extract_embeddings(model, pos_kws, f"{name}-pos_kws")
        print(f"Extracting POS CMD ({len(pos_cmd)} files)...")
        pc = extract_embeddings(model, pos_cmd, f"{name}-pos_cmd")
        print(f"Extracting NEG KWS ({len(neg_kws)} files)...")
        nk = extract_embeddings(model, neg_kws, f"{name}-neg_kws")
        print(f"Extracting NEG CMD ({len(neg_cmd)} files)...")
        nc = extract_embeddings(model, neg_cmd, f"{name}-neg_cmd")

        ps = np.array([cosine(pk[i], pc[i]) for i in range(len(pos_samples))])
        ns = np.array([cosine(nk[i], nc[i]) for i in range(len(neg_samples))])

        # EER / AUC
        scores = np.concatenate([ps, ns])
        labels = np.concatenate([np.ones(len(ps)), np.zeros(len(ns))])
        idx = np.argsort(scores)[::-1]
        ls = labels[idx]
        tpr = np.cumsum(ls) / len(ps)
        fpr = np.cumsum(1 - ls) / len(neg_samples)
        diff = np.abs(fpr - (1 - tpr))
        ei = np.argmin(diff)
        eer = (fpr[ei] + (1 - tpr[ei])) / 2 * 100
        thresh = scores[idx[ei]]
        auc_val = float(np.trapz(tpr, fpr))
        rr_at_eer = (ns < thresh).mean() * 100
        pp_at_eer = (ps >= thresh).mean() * 100
        elapsed = time.time() - t0

        lines.append(f"--- {name} ---")
        lines.append(f"POS Mean={ps.mean():.4f}  Std={ps.std():.4f}")
        lines.append(f"NEG Mean={ns.mean():.4f}  Std={ns.std():.4f}")
        lines.append(f"EER={eer:.2f}%  AUC={auc_val:.4f}")
        lines.append(f"Threshold={thresh:.4f}")
        lines.append(f"pos通过率@EER={pp_at_eer:.1f}%")
        lines.append(f"RR(拒识句准)@EER={rr_at_eer:.1f}%")
        lines.append(f"Time={elapsed:.1f}s")
        lines.append("")

        print(f"\n--- {name} Results ---")
        print(f"POS Mean={ps.mean():.4f}  NEG Mean={ns.mean():.4f}")
        print(f"EER={eer:.2f}%  AUC={auc_val:.4f}")
        print(f"RR@EER阈值={rr_at_eer:.1f}%  pos通过率={pp_at_eer:.1f}%")
        print(f"\n{'Threshold':<12} {'pos_pass':<12} {'RR':<12} {'FAR':<12}")
        for th in np.arange(0.05, 0.51, 0.05):
            pr = (ps >= th).mean() * 100
            nr = (ns < th).mean() * 100
            fa = (ns >= th).mean() * 100
            print(f"{th:<12.2f} {pr:<12.1f}% {nr:<12.1f}% {fa:<12.1f}%")
            lines.append(f"th={th:.2f}  pos_pass={pr:.1f}%  RR={nr:.1f}%  FAR={fa:.1f}%")

        lines.append("")

        # Save npz
        np.savez_compressed(
            os.path.join(OUTPUT_DIR, f"v1_{name.lower()}_sims.npz"),
            pos_sims=ps, neg_sims=ns, eer=eer, auc=auc_val, threshold=thresh
        )

    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, "compare_rejection_v1.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
