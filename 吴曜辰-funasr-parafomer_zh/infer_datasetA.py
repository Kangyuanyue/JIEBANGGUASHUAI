#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DatasetA 语音识别 + CER 评估

模型: paraformer-zh (FunASR)
输入: pos.jsonl 中 cmd 音频 → 转写文本
输出: 逐条 CER + 整体 CER
"""

import os, sys, re, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from jiwer import cer

# ===== 配置 =====
DATASET_DIR = r"C:\Users\Wuyc\Desktop\datasetA"
POS_JSONL   = os.path.join(DATASET_DIR, "pos.jsonl")
OUTPUT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
MODEL_NAME  = "paraformer-zh"
DEVICE      = "cuda"
BATCH_SIZE  = 32

def clean_text(text):
    return re.sub(r'[^一-龥a-z0-9]', '', text.lower())

def load_jsonl(path):
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t_total = time.time()

    print(f"Model: {MODEL_NAME} | Device: {DEVICE}")
    print(f"Dataset: {POS_JSONL}")

    samples = load_jsonl(POS_JSONL)
    print(f"Samples: {len(samples)}")

    # Collect audio paths and references
    audio_paths = []
    ref_texts = []
    for s in samples:
        audio_paths.append(os.path.normpath(os.path.join(DATASET_DIR, s["识别音频"])))
        ref_texts.append(s["识别文本"])

    # Load model
    model = AutoModel(model=MODEL_NAME, device=DEVICE)

    # Batch inference
    print("Running ASR...")
    t0 = time.time()
    results = model.generate(input=audio_paths, batch_size=BATCH_SIZE)
    infer_time = time.time() - t0
    print(f"Done: {infer_time:.0f}s ({infer_time/len(samples):.2f}s/sample)")

    # Extract hypotheses
    hyps = []
    hyps_raw = []
    for res in results:
        text = rich_transcription_postprocess(res["text"])
        text = " ".join(text.split())
        hyps_raw.append(text)
        hyps.append(clean_text(text))

    refs = [clean_text(t) for t in ref_texts]

    # Compute CER
    overall_cer = cer(refs, hyps) * 100
    per_cer = [cer([r], [h]) * 100 for r, h in zip(refs, hyps)]

    print(f"\nOverall CER: {overall_cer:.2f}%")
    print(f"Time: {infer_time:.0f}s")

    # Save TSV result
    tsv_path = os.path.join(OUTPUT_DIR, "datasetA_result.txt")
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("ID\tAudio\tHypothesis\tReference\tCER(%)\n")
        for i, s in enumerate(samples):
            f.write(f"{s['id']}\t{s['识别音频']}\t{hyps_raw[i]}\t{ref_texts[i]}\t{per_cer[i]:.2f}\n")
    print(f"TSV: {tsv_path}")

    # Save JSON result (competition format)
    json_path = os.path.join(OUTPUT_DIR, "submission.json")
    json_results = []
    for i, s in enumerate(samples):
        json_results.append({
            "id": str(s["id"]),
            "content": hyps_raw[i],
            "label": ref_texts[i],
            "cer": f"{per_cer[i]:.2f}"
        })
    submission = {
        "result": {
            "results": json_results,
            "final_cer": f"{overall_cer:.2f}",
            "duration": f"{infer_time:.2f}"
        }
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(submission, f, ensure_ascii=False, indent=2)
    print(f"JSON: {json_path}")

    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, "datasetA_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {MODEL_NAME} | Device: {DEVICE}\n")
        f.write(f"Samples: {len(samples)}\n")
        f.write(f"Overall CER: {overall_cer:.2f}%\n")
        f.write(f"Inference Time: {infer_time:.0f}s\n")

if __name__ == "__main__":
    main()
