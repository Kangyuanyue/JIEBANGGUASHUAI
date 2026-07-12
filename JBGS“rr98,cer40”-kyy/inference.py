import os
import sys
import json
import time
import torch
import numpy as np
import pickle
import argparse

# Apply monkeypatch for PyTorch/transformers compatibility
import torch.utils._pytree
if not hasattr(torch.utils._pytree, 'register_pytree_node'):
    torch.utils._pytree.register_pytree_node = lambda typ, flat, unflat, *args, **kwargs: (
        kwargs.pop('serialized_type_name', None), 
        torch.utils._pytree._register_pytree_node(typ, flat, unflat, *args, **kwargs)
    )

from funasr import AutoModel

def edit_distance(pred, label):
    # Standard dynamic programming edit distance
    d = [[0] * (len(label) + 1) for _ in range(len(pred) + 1)]
    for i in range(len(pred) + 1):
        d[i][0] = i
    for j in range(len(label) + 1):
        d[0][j] = j
        
    for i in range(1, len(pred) + 1):
        for j in range(1, len(label) + 1):
            if pred[i-1] == label[j-1]:
                d[i][j] = d[i-1][j-1]
            else:
                d[i][j] = min(
                    d[i-1][j] + 1,      # deletion
                    d[i][j-1] + 1,      # insertion
                    d[i-1][j-1] + 1     # substitution
                )
    return d[len(pred)][len(label)]

def compute_similarity(emb1, emb2):
    if hasattr(emb1, 'cpu'):
        emb1 = emb1.cpu().numpy()
    else:
        emb1 = np.array(emb1)
    if hasattr(emb2, 'cpu'):
        emb2 = emb2.cpu().numpy()
    else:
        emb2 = np.array(emb2)
    emb1 = emb1.flatten()
    emb2 = emb2.flatten()
    sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return float(sim) if not np.isnan(sim) else 0.0

def main():
    parser = argparse.ArgumentParser(description="Inference for Anti-Interference Voice Command Recognition")
    parser.add_argument("--input_jsonl", type=str, required=True, help="Path to input jsonl file")
    parser.add_argument("--output_json", type=str, required=True, help="Path to output json file")
    parser.add_argument("--label_jsonl", type=str, default=None, help="Optional path to label jsonl file if separate")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Determine paths for classifier and parameters (relative to script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    params_path = os.path.join(script_dir, "best_params.json")
    
    if not os.path.exists(params_path):
        print(f"Error: Parameters file not found at {params_path}. Please run train_pipeline.py first.")
        sys.exit(1)
        
    with open(params_path, "r", encoding="utf-8") as f:
        params = json.load(f)
        
    classifier_path = os.path.join(script_dir, params["model_path"])
    if not os.path.exists(classifier_path):
        print(f"Error: Classifier model not found at {classifier_path}.")
        sys.exit(1)
        
    with open(classifier_path, "rb") as f:
        model_assets = pickle.load(f)
        
    vectorizer = model_assets["vectorizer"]
    classifier = model_assets["classifier"]
    
    # Load FunASR models
    print("Loading CAM++ Speaker Verification model...")
    sv_model = AutoModel(model="iic/speech_campplus_sv_zh-cn_16k-common", device=device, disable_update=True)
    print("Loading Paraformer ASR model...")
    asr_model = AutoModel(model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch", device=device, disable_update=True)

    # Load input jsonl
    input_items = []
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                input_items.append(json.loads(line))
                
    # Load label jsonl if provided
    labels_map = {}
    if args.label_jsonl:
        with open(args.label_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    labels_map[item["id"]] = item.get("识别文本") or ""

    # Resolve audio folder relative to the input jsonl directory
    input_dir = os.path.dirname(os.path.abspath(args.input_jsonl))

    results = []
    total_time = 0.0
    
    print(f"Running inference on {len(input_items)} samples...")
    for idx, item in enumerate(input_items):
        kws_rel = item["唤醒音频"]
        cmd_rel = item["识别音频"]
        
        # Candidate 1: relative to input jsonl directory
        kws_path = os.path.join(input_dir, kws_rel)
        cmd_path = os.path.join(input_dir, cmd_rel)
        
        # Candidate 2: input_dir / datasetA / rel
        if not os.path.exists(kws_path):
            kws_path = os.path.join(input_dir, "datasetA", kws_rel)
        if not os.path.exists(cmd_path):
            cmd_path = os.path.join(input_dir, "datasetA", cmd_rel)
            
        # Candidate 3: CWD / datasetA / rel
        if not os.path.exists(kws_path):
            kws_path = os.path.join(os.getcwd(), "datasetA", kws_rel)
        if not os.path.exists(cmd_path):
            cmd_path = os.path.join(os.getcwd(), "datasetA", cmd_rel)
            
        # Candidate 4: CWD / rel
        if not os.path.exists(kws_path):
            kws_path = os.path.join(os.getcwd(), kws_rel)
        if not os.path.exists(cmd_path):
            cmd_path = os.path.join(os.getcwd(), cmd_rel)
            
        t_start = time.time()
        
        # 1. SV
        try:
            res_kws = sv_model.generate(input=kws_path)
            res_cmd = sv_model.generate(input=cmd_path)
            emb_kws = res_kws[0]["spk_embedding"]
            emb_cmd = res_cmd[0]["spk_embedding"]
            sv_score = compute_similarity(emb_kws, emb_cmd)
        except Exception as e:
            print(f"Error running SV for ID {item['id']}: {e}")
            sv_score = 0.0
            
        # 2. ASR
        try:
            res_asr = asr_model.generate(input=cmd_path)
            asr_text = res_asr[0]["text"]
        except Exception as e:
            print(f"Error running ASR for ID {item['id']}: {e}")
            asr_text = ""
            
        # 3. Intent classification
        try:
            if asr_text:
                x_feat = vectorizer.transform([asr_text])
                prob = float(classifier.predict_proba(x_feat)[0, 1])
            else:
                prob = 0.0
        except Exception as e:
            print(f"Error running Text Classifier for ID {item['id']}: {e}")
            prob = 0.0
            
        # 4. Decision logic
        sv_low = params["sv_low"]
        sv_high = params["sv_high"]
        prob_low = params["prob_low"]
        prob_high = params["prob_high"]
        
        accepted = (sv_score >= sv_low and prob >= prob_high) or (sv_score >= sv_high and prob >= prob_low)
        content = asr_text if accepted else ""
        
        t_end = time.time()
        total_time += (t_end - t_start)
        
        # Determine ground truth label
        label = ""
        if item["id"] in labels_map:
            label = labels_map[item["id"]]
        elif "识别文本" in item:
            label = item["识别文本"] or ""
            
        # Compute individual CER
        cer_val = 0.0
        if label:
            cer_val = float(edit_distance(content, label) / len(label))
        elif content:
            # Predicted content but ground truth is empty (false acceptance)
            # CER is only defined on positive samples, so we keep individual CER as 0.0 or 1.0.
            # Usually individual CER for negative samples is set to 0.0 since it is not counted in CER.
            cer_val = 0.0

        results.append({
            "id": item["id"],
            "content": content,
            "label": label,
            "cer": f"{cer_val:.4f}"
        })
        
        if (idx + 1) % 50 == 0 or (idx + 1) == len(input_items):
            print(f"  Processed {idx + 1}/{len(input_items)}")

    # Calculate overall metrics
    pos_cer_sum = 0
    pos_char_count = 0
    
    for r in results:
        # Check if there is a ground truth label to determine if it is a positive sample
        label = r["label"]
        if label:  # Positive sample
            dist = edit_distance(r["content"], label)
            pos_cer_sum += dist
            pos_char_count += len(label)
            
    final_cer = float(pos_cer_sum / pos_char_count) if pos_char_count > 0 else 0.0
    
    # Save output JSON
    output_data = {
        "result": {
            "results": results,
            "final_cer": f"{final_cer:.4f}",
            "duration": f"{total_time:.2f}"
        }
    }
    
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)
        
    print(f"Inference completed in {total_time:.2f}s. Results written to {args.output_json}")
    print(f"Final calculated CER: {final_cer:.4f}")

if __name__ == "__main__":
    main()
