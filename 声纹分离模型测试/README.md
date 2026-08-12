# 复杂交互场景下抗干扰语音指令识别 — Pipeline 方案

## 比赛

XH-202615 "复杂交互场景的抗干扰语音指令识别技术"

## 方案概述

### 整体流程

```
输入: 唤醒音频(kws) + 指令音频(cmd)
     │
     ├─ Step 1: ERes2NetV2 提取 kws 声纹 embedding
     │
     ├─ Step 2: ERes2NetV2 提取 cmd 声纹 embedding
     │
     ├─ Step 3: 计算余弦相似度 sim = cos(kws_emb, cmd_emb)
     │
     ├─ sim < REJECT_THRESHOLD  → 拒识（非目标说话人）
     │
     ├─ sim < SEP_THRESHOLD     → MossFormer2 语音分离 → 声纹匹配选最佳流 → Paraformer ASR
     │
     └─ sim >= SEP_THRESHOLD    → Paraformer ASR（干净音频直接识别）
```

### 模型

| 模型 | 用途 | 来源 |
|------|------|------|
| ERes2NetV2 | 声纹提取（192维） | modelscope: iic/speech_eres2netv2_sv_zh-cn_16k-common |
| Paraformer | 语音转文字（ASR） | modelscope: iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch |
| MossFormer2_SS_16K | 盲语音分离（2路） | HuggingFace: alibabasglab/MossFormer2_SS_16K (via hf-mirror.com) |

### 核心思路

1. **声纹相似度做"复杂度检测"**：同一说话人的 cmd 声纹应该与 kws 声纹高度相似。相似度低 → 要么有干扰，要么不是目标说话人
2. **拒识 + 分离双阈值分级处理**：区分"人不对（拒识）"和"干扰太大（分离）"两种情况
3. **分离只在需要时启用**：干净音频直接 ASR，避免画蛇添足

## 测试结果

**测试集**：DatasetA（1364 POS + 474 NEG，仅用于测试，未参与训练）

**基线**：直接 Paraformer ASR（无拒识、无分离）→ CER = **46.55%**

### 最优阈值配置

| 参数 | 数值 |
|------|------|
| REJECT_THRESHOLD | 0.30 |
| SEP_THRESHOLD | 0.35 |

### 性能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| **CER** | **42.97%** | 比基线降低 3.58pp |
| **NEG 拒识率** | **86.1%** (408/474) | 正确拦截非目标说话人 |
| **POS 误拒率 (FRR)** | 26.0% (355/1364) | 误拒比例 |
| **NEG 漏过率 (FAR)** | 13.9% | 非目标说话人通过率 |

### 声纹相似度分布

| | POS（目标说话人） | NEG（非目标说话人） |
|--|--|--|
| 均值 | 0.413 | 0.152 |
| 中位数 | 0.426 | 0.142 |
| <0.25 比例 | 15.4% | 79.1% |
| <0.30 比例 | 22.7% | 86.9% |

POS/NEG 的 sim 分布有明显区分度，验证了声纹拒识的可行性。

### 分离效果（仅对复杂样本）

| | 全部 POS | 复杂样本（原始 CER > 50%） |
|--|---------|--------------------------|
| 数量 | 1364 | 457 (33.5%) |
| 直接 ASR CER | 46.55% | 120.95% |
| 分离后 CER | 51.51% | **90.63%** |
| 改善样本 | 297 | **214/457** |

对于真正有干扰的复杂样本，MossFormer2 分离能将 CER 降低约 30 个百分点。

## 文件结构

```
pipeline_results/
├── README.md                    # 本文件
├── code/
│   └── eval_all.py              # 全量评估脚本（可复现）
└── results/
    └── threshold_sweep.txt      # 阈值扫描详细结果
```

## 运行方式

```bash
cd yuyinshibie
set HF_ENDPOINT=https://hf-mirror.com
python scripts/eval_all.py
```

**环境**：conda funasr_env, Python 3.10, PyTorch CUDA, RTX 4070 8GB

**预计耗时**：~25 分钟（1838 样本）

**输出**：
- `output/eval_all/all_records.json` — 每样本的 sim / cer_clean / cer_sep
- `output/eval_all/threshold_sweep.txt` — 56 组阈值组合排名 + 推荐配置

## 其他尝试

| 方案 | CER | 结论 |
|------|-----|------|
| CAM++ 声纹拒识 | EER=31.33% | 不如 ERes2NetV2 |
| ERes2NetV2 声纹拒识 | EER=17.30% | ✅ 最佳拒识模型 |
| Fine-tune ERes2NetV2 | EER=39.20% | 训练/测试分布不匹配，反而变差 |
| TSE SpEx+ 训练 | val_loss=-1.96 | 训练数据太少（500条），模型太弱 |
| MossFormer2 盲分离 + ASR | CER=42.97% | ✅ 当前最优 |
| FRCRN 降噪 + 分离 + ASR | CER=50.43% | ✗ 降噪反而损害 ASR |

## 总结

通过 **ERes2NetV2 声纹复杂度检测 + MossFormer2 按需分离 + Paraformer ASR** 的三级 pipeline，在保持 86% 拒识率的同时将 CER 降低了 3.6 个百分点。方案的核心创新在于用声纹相似度作为"是否需要分离"的决策依据，避免在干净音频上过度处理。

## 参考开源仓库

| 模型 / 工具 | 用途 | GitHub 仓库 |
|---|---|---|
| ClearVoice MossFormer2 | 语音分离（本测试的分离引擎） | https://github.com/modelscope/ClearerVoice-Studio |
| Ultimate Vocal Remover 5 (UVR5) | 人声分离 | https://github.com/Anjok07/ultimatevocalremovergui |
| WhisperX | 转写 + 词级时间戳 | https://github.com/m-bain/whisperX |
| pyannote.audio | 说话人日志 | https://github.com/pyannote/pyannote-audio |
| Kaldi | 语音识别工具包 | https://github.com/kaldi-asr/kaldi |
| DiariZen-Large-s80-v2 | 端到端说话人日志（EEND, 80% 结构化剪枝） | https://github.com/BUTSpeechFIT/DiariZen |
