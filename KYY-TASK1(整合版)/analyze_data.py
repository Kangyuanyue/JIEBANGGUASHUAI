#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, json, wave, os
sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = r'C:\Users\kk200\Desktop\JBGS\datasetA'

# Read data
with open(os.path.join(DATA_DIR, 'pos.jsonl'), 'r', encoding='utf-8') as f:
    pos_data = [json.loads(l) for l in f if l.strip()]
with open(os.path.join(DATA_DIR, 'neg.jsonl'), 'r', encoding='utf-8') as f:
    neg_data = [json.loads(l) for l in f if l.strip()]

print(f'Pos samples: {len(pos_data)}')
print(f'Neg samples: {len(neg_data)}')

# Neg labels
neg_labels = set()
for d in neg_data:
    neg_labels.add(str(d.get('\u8bc6\u522b\u6587\u672c')))
print(f'Neg label values: {neg_labels}')

# Audio durations
pos_cmd_dur = []
pos_kws_dur = []
for d in pos_data[:100]:
    try:
        cmd_path = os.path.join(DATA_DIR, d['\u8bc6\u522b\u97f3\u9891'])
        with wave.open(cmd_path, 'r') as w:
            pos_cmd_dur.append(w.getnframes() / w.getframerate())
    except:
        pass
    try:
        kws_path = os.path.join(DATA_DIR, d['\u5524\u9192\u97f3\u9891'])
        with wave.open(kws_path, 'r') as w:
            pos_kws_dur.append(w.getnframes() / w.getframerate())
    except:
        pass

print(f'Pos cmd duration (100): min={min(pos_cmd_dur):.2f}, max={max(pos_cmd_dur):.2f}, avg={sum(pos_cmd_dur)/len(pos_cmd_dur):.2f}s')
print(f'Pos kws duration (100): min={min(pos_kws_dur):.2f}, max={max(pos_kws_dur):.2f}, avg={sum(pos_kws_dur)/len(pos_kws_dur):.2f}s')

# PDF
try:
    import fitz
    doc = fitz.open(os.path.join(DATA_DIR, '..', 'XH-202615_\u590d\u6742\u4ea4\u4e92\u573a\u666f\u7684\u6297\u5e72\u6270\u8bed\u97f3\u6307\u4ee4\u8bc6\u522b\u6280\u672f.pdf'))
    for i in range(min(4, doc.page_count)):
        print(f'\n=== PDF Page {i+1} ===')
        print(doc[i].get_text())
    doc.close()
except Exception as e:
    print(f'PDF error: {e}')

# Check wake text distribution in neg
from collections import Counter
neg_wake = Counter([d.get('\u5524\u9192\u6587\u672c', '') for d in neg_data])
print(f'\nNeg wake word dist: {neg_wake}')
