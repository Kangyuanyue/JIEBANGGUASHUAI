import json
import numpy as np
from train_pipeline import edit_distance

with open('extracted_features.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

pos_res = [r for r in data if r['is_positive']]

cers = []
exact_matches = 0
for r in pos_res:
    dist = edit_distance(r['asr_text'], r['label_text'])
    cer = dist / len(r['label_text'])
    cers.append(cer)
    if dist == 0:
        exact_matches += 1

cers = np.array(cers)
print(f"Total positive samples: {len(pos_res)}")
print(f"Exact matches (CER = 0): {exact_matches} ({exact_matches/len(pos_res)*100:.1f}%)")
print(f"Mean CER: {np.mean(cers):.4f}")
print(f"Median CER: {np.median(cers):.4f}")
print(f"Percent of samples with CER > 0.5: {np.sum(cers > 0.5)} ({np.sum(cers > 0.5)/len(pos_res)*100:.1f}%)")
print(f"Percent of samples with CER = 1.0 (empty or completely wrong ASR): {np.sum(cers >= 1.0)} ({np.sum(cers >= 1.0)/len(pos_res)*100:.1f}%)")

print("\nTop 15 worst samples (high edit distance):")
sorted_worst = sorted(pos_res, key=lambda x: edit_distance(x['asr_text'], x['label_text']) / len(x['label_text']), reverse=True)
for r in sorted_worst[:15]:
    dist = edit_distance(r['asr_text'], r['label_text'])
    cer = dist / len(r['label_text'])
    # Write to log in safe ascii/unicode escape if needed, or write a txt file to inspect
    
# Let's write the details to a file for safe inspection
out = []
for r in sorted_worst[:50]:
    dist = edit_distance(r['asr_text'], r['label_text'])
    cer = dist / len(r['label_text'])
    out.append(f"ID {r['id']} | CER={cer:.4f} | ASR: {r['asr_text']} | Label: {r['label_text']}")
    
with open('worst_samples_utf8.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print("Worst samples written to worst_samples_utf8.txt")
