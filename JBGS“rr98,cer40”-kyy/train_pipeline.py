import os
import json
import torch
import numpy as np
import pickle
import time
from funasr import AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Paths
pos_path = 'datasetA/pos.jsonl'
neg_path = 'datasetA/neg.jsonl'
features_cache_path = 'extracted_features.json'
classifier_model_path = 'intent_classifier.pkl'

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

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

def evaluate_predictions(data, predictions):
    pos_cer_sum = 0
    pos_char_count = 0
    neg_total = 0
    neg_correct = 0
    
    for item, pred in zip(data, predictions):
        if item['is_positive']:
            label = item['label_text']
            dist = edit_distance(pred, label)
            pos_cer_sum += dist
            pos_char_count += len(label)
        else:
            neg_total += 1
            if pred == "":
                neg_correct += 1
                
    cer = pos_cer_sum / pos_char_count if pos_char_count > 0 else 0.0
    rr = neg_correct / neg_total if neg_total > 0 else 0.0
    score = 0.4 * (1.0 - cer) + 0.4 * rr
    return cer, rr, score

def load_data():
    pos_items = []
    with open(pos_path, 'r', encoding='utf-8') as f:
        for line in f:
            pos_items.append(json.loads(line))
            
    neg_items = []
    with open(neg_path, 'r', encoding='utf-8') as f:
        for line in f:
            neg_items.append(json.loads(line))
            
    return pos_items, neg_items

def extract_features(items, is_positive, sv_model, asr_model):
    results = []
    batch_size = 32
    total = len(items)
    print(f"Processing {'positive' if is_positive else 'negative'} items (total={total})...")
    
    for start_idx in range(0, total, batch_size):
        end_idx = min(start_idx + batch_size, total)
        batch = items[start_idx:end_idx]
        
        kws_paths = [os.path.join('datasetA', item['唤醒音频']) for item in batch]
        cmd_paths = [os.path.join('datasetA', item['识别音频']) for item in batch]
        
        # Run SV
        try:
            res_kws = sv_model.generate(input=kws_paths)
            res_cmd = sv_model.generate(input=cmd_paths)
        except Exception as e:
            print(f"SV batch failed for range {start_idx}-{end_idx}: {e}")
            continue
            
        # Run ASR
        try:
            res_asr = asr_model.generate(input=cmd_paths, batch_size_s=300)
        except Exception as e:
            print(f"ASR batch failed for range {start_idx}-{end_idx}: {e}")
            continue
            
        # Extract and calculate
        for i, item in enumerate(batch):
            try:
                emb_kws = res_kws[i]['spk_embedding']
                emb_cmd = res_cmd[i]['spk_embedding']
                
                if hasattr(emb_kws, 'cpu'):
                    emb_kws = emb_kws.cpu().numpy()
                else:
                    emb_kws = np.array(emb_kws)
                if hasattr(emb_cmd, 'cpu'):
                    emb_cmd = emb_cmd.cpu().numpy()
                else:
                    emb_cmd = np.array(emb_cmd)
                    
                emb_kws = emb_kws.flatten()
                emb_cmd = emb_cmd.flatten()
                
                sim = float(np.dot(emb_kws, emb_cmd) / (np.linalg.norm(emb_kws) * np.linalg.norm(emb_cmd)))
                if np.isnan(sim):
                    sim = 0.0
            except Exception as e:
                print(f"Failed to calculate similarity for ID {item['id']}: {e}")
                sim = 0.0
                
            try:
                asr_text = res_asr[i]['text']
            except Exception as e:
                print(f"Failed to get ASR text for ID {item['id']}: {e}")
                asr_text = ""
                
            results.append({
                "id": item['id'],
                "is_positive": is_positive,
                "kws_file": item['唤醒音频'],
                "cmd_file": item['识别音频'],
                "label_text": item.get('识别文本') or "",
                "sv_score": sim,
                "asr_text": asr_text
            })
        print(f"  Processed {end_idx}/{total}")
    return results

def main():
    if os.path.exists(features_cache_path):
        print(f"Loading cached features from {features_cache_path}...")
        with open(features_cache_path, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
    else:
        print("Loading models...")
        sv_model = AutoModel(model="iic/speech_campplus_sv_zh-cn_16k-common", device=device, disable_update=True)
        asr_model = AutoModel(model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch", device=device, disable_update=True)
        
        pos_items, neg_items = load_data()
        
        pos_results = extract_features(pos_items, True, sv_model, asr_model)
        neg_results = extract_features(neg_items, False, sv_model, asr_model)
        
        all_results = pos_results + neg_results
        with open(features_cache_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"Saved extracted features to {features_cache_path}")

    # Split into train (80%) and test (20%) stratified by class
    pos_res = [r for r in all_results if r['is_positive']]
    neg_res = [r for r in all_results if not r['is_positive']]
    
    np.random.seed(42)
    np.random.shuffle(pos_res)
    np.random.shuffle(neg_res)
    
    pos_train_len = int(len(pos_res) * 0.8)
    neg_train_len = int(len(neg_res) * 0.8)
    
    train_data = pos_res[:pos_train_len] + neg_res[:neg_train_len]
    test_data = pos_res[pos_train_len:] + neg_res[neg_train_len:]
    
    print(f"Train set: {len(train_data)} samples ({pos_train_len} pos, {neg_train_len} neg)")
    print(f"Test set: {len(test_data)} samples ({len(pos_res)-pos_train_len} pos, {len(neg_res)-neg_train_len} neg)")
    
    # Train Text Classifier
    print("Training TF-IDF + Logistic Regression Intent Classifier...")
    train_texts = [r['asr_text'] for r in train_data]
    train_labels = [1 if r['is_positive'] else 0 for r in train_data]
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), analyzer='char')
    X_train = vectorizer.fit_transform(train_texts)
    
    clf = LogisticRegression(class_weight='balanced', max_iter=1000)
    clf.fit(X_train, train_labels)
    
    # Save the vectorizer and classifier
    model_assets = {
        "vectorizer": vectorizer,
        "classifier": clf
    }
    with open(classifier_model_path, 'wb') as f:
        pickle.dump(model_assets, f)
    print(f"Saved intent classifier assets to {classifier_model_path}")
    
    # Evaluate thresholds on train data to find the best policy
    print("Searching for the best decision boundary on train data...")
    
    # Pre-calculate probability predictions
    train_probs = clf.predict_proba(X_train)[:, 1]
    for i, r in enumerate(train_data):
        r['text_prob'] = float(train_probs[i])
        
    test_texts = [r['asr_text'] for r in test_data]
    X_test = vectorizer.transform(test_texts)
    test_probs = clf.predict_proba(X_test)[:, 1]
    for i, r in enumerate(test_data):
        r['text_prob'] = float(test_probs[i])

    best_score = -1.0
    best_params = {}
    
    # Grid search parameters
    # Logic: accept if (sv_score >= sv_low and text_prob >= prob_high) or (sv_score >= sv_high and text_prob >= prob_low)
    for sv_low in np.arange(0.10, 0.25, 0.02):
        for sv_high in np.arange(0.25, 0.45, 0.02):
            for prob_low in np.arange(0.05, 0.35, 0.05):
                for prob_high in np.arange(0.60, 0.98, 0.05):
                    # Evaluate on train
                    predictions = []
                    for r in train_data:
                        sv = r['sv_score']
                        prob = r['text_prob']
                        # Decision boundary
                        accepted = (sv >= sv_low and prob >= prob_high) or (sv >= sv_high and prob >= prob_low)
                        if accepted:
                            predictions.append(r['asr_text'])
                        else:
                            predictions.append("")
                            
                    cer, rr, score = evaluate_predictions(train_data, predictions)
                    if score > best_score:
                        best_score = score
                        best_params = {
                            "sv_low": float(sv_low),
                            "sv_high": float(sv_high),
                            "prob_low": float(prob_low),
                            "prob_high": float(prob_high),
                            "train_cer": cer,
                            "train_rr": rr,
                            "train_score": score
                        }
                        
    print("\nBest Parameters found on Train set:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
        
    # Evaluate on Test set using best parameters
    test_predictions = []
    for r in test_data:
        sv = r['sv_score']
        prob = r['text_prob']
        accepted = (sv >= best_params['sv_low'] and prob >= best_params['prob_high']) or \
                   (sv >= best_params['sv_high'] and prob >= best_params['prob_low'])
        if accepted:
            test_predictions.append(r['asr_text'])
        else:
            test_predictions.append("")
            
    test_cer, test_rr, test_score = evaluate_predictions(test_data, test_predictions)
    print("\nTest Set Performance with Best Parameters:")
    print(f"  CER: {test_cer:.4f} (Error rate on positive commands)")
    print(f"  RR:  {test_rr:.4f} (Rejection rate on negative commands)")
    print(f"  Combined Score (0.4*(1-CER) + 0.4*RR): {test_score:.4f}")
    
    # Save the parameters
    best_params['model_path'] = classifier_model_path
    with open('best_params.json', 'w', encoding='utf-8') as f:
        json.dump(best_params, f, ensure_ascii=False, indent=2)
    print("Saved best parameters to best_params.json")

if __name__ == '__main__':
    main()
