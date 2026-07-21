# 语音指令识别 — 声纹验证 + 语音分离 + 降噪 Pipeline

> **CAM++ 声纹验证 + ConvTasNet TSE 说话人提取 + Paraformer-Large ASR**
> 赛题: 复杂交互场景的抗干扰语音指令识别 (XH-202615)

---

## 任务简介

从含干扰人声的短音频指令中识别目标说话人的语音指令。

**核心挑战**:
- **短音频**: 指令仅 3~4 秒，唤醒词仅 0.5~1.5 秒
- **强重叠**: 目标说话人与干扰人声高度重叠
- **多 SNR**: 测试覆盖 5dB / 0dB / -5dB 多种信噪比

---

## 算法架构

```
kws_0.wav (唤醒词)
    │
    ├── [降噪] noisereduce
    │
    ├──▶ CAM++ SV (声纹注册) ──▶ enroll_embedding [256]
    │                                  │
    │                                  ▼
    │                         ┌──────────────────┐
    │                         │   联合判决        │
    │                         │  SV score ≥ 0.3  │───→ 输出 ASR 结果                   
    │                         │  ASR conf ≥ 0.8  |───→ （容错机制）
    │                         │ 否则 → 拒识(空)   |
    │                         └──────────────────┘
    │
    └──▶ ConvTasNet TSE  ──▶ 目标语音提取
         enroll(kws) + extract(cmd) → enhanced
              │
              ▼
         CAM++ SV + Paraformer-Large ASR
```

### 三条 Pipeline

| # | Pipeline | 适用场景 |
|---|----------|---------|
| 1 | **降噪 + SV + ASR + 联合判决** | 噪声环境优化 |
| 2 | **TSE + SV + ASR + 联合判决** | 强重叠人声场景 |

---

## 项目结构

```
github/
├── scripts/
│   ├── eval_joint_v2.py          # [主入口] 降噪 + SV + ASR + 联合判决
│   ├── eval_tse_full.py          # TSE + SV + ASR 完整测试
│   ├── eval_sv.py                # 声纹验证单独评估 (计算 EER)
│   ├── train_tse.py              # ConvTasNet TSE 模型训练
│   └── prepare_tse_aishell.py    # AISHELL-1 TSE 训练数据生成
├── docs/
│   └── CER_requirement.md        # CER 计算标准
├── outputs/
│   ├── pretrained_convtasnet.pt  # ConvTasNet 预训练权重 (Asteroid, 19MB)
│   └── tse_model/
│       └── model.pt              # 训练好的 TSE 模型权重
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 安装

```bash
# 基础依赖
pip install torch torchaudio
pip install funasr soundfile numpy noisereduce scikit-learn librosa

# TSE 训练额外依赖
pip install tensorboardX

# 一键安装
pip install -r requirements.txt
```

---

## 快速开始

### 1. 联合判决 Pipeline (SV + ASR)

```bash
# 快速测试 100 条
python scripts/eval_joint_v2.py --limit 100

# 全部 1364 条测试 + 阈值扫描
python scripts/eval_joint_v2.py
```

### 2. 声纹验证单独评估

```bash
python scripts/eval_sv.py
```

### 3. TSE 训练 (实验性)

```bash
# 数据准备 (需要 AISHELL-1)
python scripts/prepare_tse_aishell.py

# CPU 训练 (轻量模式, 0.24M 参数)
python scripts/train_tse.py --light --epochs 10

# GPU 训练 (完整模型, 5M 参数)
python scripts/train_tse.py --epochs 20 --batch_size 4
```

### 4. TSE 完整测试

```bash
python scripts/eval_tse_full.py
```

---

## 联合判决逻辑

```python
# 三步判决 (eval_joint_v2.py):
if sv_score >= 0.3:
    accept = True           # ① 声纹确认 → 接受
elif asr_conf >= 0.8:
    accept = True           # ② ASR 极高置信 → 接受 (容错)
else:
    accept = False          # ③ 拒识
```

---

## ConvTasNet TSE 模型

### 架构

```
ConvTasNet 风格 + VE-VE 说话人提取:

Enroll:  kws → Conv1D(512,16,8) → TCN×24 → Global Avg Pool → embedding [128]
Extract: cmd → Conv1D(512,16,8) → FiLM(embedding) → TCN×24 → Decoder → mask
```

### 预训练权重

使用 Asteroid 提供的 ConvTasNet (WHAM! sepclean) 作为初始化:
- 匹配 97/328 层 (encoder + TCN conv + decoder)
- 新增 speaker conditioning 层 (spk_proj, cc) 随机初始化
- 下载: `outputs/pretrained_convtasnet.pt`


---

## 预训练模型

首次运行时自动从 ModelScope 下载:

| 模型 | 用途 | 参数量 |
|---|---|---|
| `speech_campplus_sv_zh-cn_16k-common` | 声纹验证 (SV) | ~5M |
| `speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch` | 语音识别 (ASR) | 220M |
| `speech_fsmn_vad_zh-cn-16k-common-pytorch` | 语音活动检测 | ~2M |
| `outputs/pretrained_convtasnet.pt` | TSE 预训练 (可选) | 19MB |

---

## 测试结果 (datasetA / pos, 1364 条)

| 方案 | CER | 误拒率 | 说明 |
|---|---|---|---|
| **降噪 + Paraformer-Large + SV 联合判决** | **0.406** | **0%** | 当前最优方案 |
| TSE + SV + ASR  | 0.416 | **1.5%** |  21/1364 条被拒识 |

- **TSE 误拒率 1.5%**: SV 单独拒识率高达 37.3%，但 ASR 置信度容错机制救回 98% 被拒样本。最终被拒的 21 条是 TSE 输出信号过弱导致 ASR 也输出空的极端样本。

### 按子集分布

| 子集 | 样本数 | CER |
|---|---|---|
| id 0~363 (干净) | 364 | **0.098** |
| id 2000~2999 (噪声) | 1000 | 0.588 |
| 全部 1364 | 1364 | 0.406 |

> CER 遵循 `docs/CER_requirement.md` 标准: NFKC 归一化 + 去标点 + Levenshtein 编辑距离。

---

## 数据集格式

```
datasetA/
├── pos.jsonl             # 正样本: kws + cmd + 识别文本
├── neg.jsonl             # 负样本: kws + cmd (无文本)
├── pos/kws_*.wav         # 唤醒词
├── pos/cmd_*.wav         # 指令
└── neg/kws_*.wav, neg/cmd_*.wav
```

pos.jsonl:
```json
{"id": 0, "唤醒音频": "pos/kws_0.wav", "唤醒文本": "你好科慕",
 "识别音频": "pos/cmd_0.wav", "识别文本": "空调开到制热调到二十五度风量调到百分之三十"}
```

---

## 依赖

```
torch>=2.0.0
torchaudio>=2.0.0
funasr>=1.0.0
modelscope
soundfile>=0.12.0
numpy>=1.24.0
noisereduce>=3.0.0
scikit-learn>=1.0.0
scipy>=1.10.0
librosa>=0.9.0
```

---

## 引用

```bibtex
@misc{funasr2024,
  title = {FunASR: A Fundamental End-to-End Speech Recognition Toolkit},
  author = {DAMO Academy, Alibaba Group},
  year = {2024}
}

@inproceedings{luo2019conv,
  title={Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for Speech Separation},
  author={Luo, Yi and Mesgarani, Nima},
  booktitle={IEEE/ACM TASLP},
  year={2019}
}

@inproceedings{yang2022veve,
  title={Target Speaker Extraction with Ultra-Short Reference Speech by VE-VE Framework},
  author={Yang, Lei and Liu, Wei and Tan, Lufen and Yang, Jaemo and Moon, Han-gil},
  booktitle={ICASSP 2022},
  year={2022}
}
```


