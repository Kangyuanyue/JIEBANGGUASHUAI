# 以 SenseVoice Small 为模型的语音识别测试

基于阿里达摩院开源的 [SenseVoice Small](https://github.com/FunAudioLLM/SenseVoice) 模型，在干净音频与噪声音频环境下进行语音识别性能评测，测量字符错误率（CER）与推理速度。

---

## 项目简介

SenseVoice Small 是阿里通义语音团队发布的多功能语音理解模型，支持：

- 🌍 多语言语音识别（中文、英文、粤语、日语、韩语）
- 😊 情绪识别（高兴、悲伤、愤怒、中性）
- 🔊 音频事件检测（背景音乐、笑声、掌声等）
- ⚡ 极低推理延迟（10秒音频仅需 70ms，非自回归架构）

本项目在标准中文语音数据集 AISHELL-1 上进行评测，分别测试**干净音频**和**不同噪声强度**下的识别准确率，分析模型的抗干扰能力。

---

## 评测结果

### 干净音频（AISHELL-1 测试集，500条）

| 指标 | 结果 |
|------|------|
| 平均 CER | 4.12% |
| 平均推理耗时 | 0.38s/条 |
| 运行设备 | CPU（无GPU） |

### 带噪音频对比（高斯噪声，100条）

| 场景 | 平均 CER | 平均耗时 |
|------|----------|----------|
| 干净音频 | 4.12% | 0.38s/条 |
| SNR = 20dB（轻微噪声） | 14.27% | 0.48s/条 |
| SNR = 10dB（中等噪声） | 14.27% | 0.48s/条 |
| SNR = 5dB（强噪声） | 21.75% | 0.50s/条 |

> SNR 越小表示噪声越强；CER 越低表示识别越准确

### 结果分析

- 干净音频下 CER 仅 **4.12%**，识别表现优秀
- 加入轻微噪声（SNR=20dB）后 CER 上升至 **14.27%**，增幅约 10 个百分点
- 中等噪声（SNR=10dB）与轻微噪声结果相近，说明该噪声区间内模型表现较稳定
- 强噪声（SNR=5dB）下 CER 进一步升至 **21.75%**，抗干扰能力明显下降
- 总体来看，SenseVoice Small 在干净环境下表现出色，但在强噪声环境下仍有较大提升空间

---

## 环境要求

- Python 3.8+（本项目使用 Python 3.13.9）
- PyTorch 2.x（CPU 版本即可运行）
- Windows / Linux / macOS

---

## 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/FunAudioLLM/SenseVoice.git
cd SenseVoice
```

### 2. 创建虚拟环境

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install funasr --no-deps
pip install torch torchaudio
pip install modelscope huggingface_hub scipy numpy soundfile librosa transformers sentencepiece
pip install omegaconf hydra-core jaconv jamo jieba kaldiio tensorboardX tiktoken torch_complex umap_learn oss2
pip install gradio datasets audiomentations
```

> ⚠️ **Python 3.13 用户注意**：需要额外修复两处兼容性问题，详见下方[兼容性修复](#兼容性修复)

### 4. 快速测试

```bash
python test_run.py
```

成功输出示例：
```
<|en|><|NEUTRAL|><|Speech|><|withitn|>The tribal chieftain called for the boy and presented him with 50 pieces of gold.
```

### 5. 启动 WebUI

```bash
python webui.py
# 浏览器打开 http://127.0.0.1:7860
```

---

## 评测脚本使用方法

### 准备数据集

从 [OpenSLR](https://www.openslr.org/33/) 下载 AISHELL-1 数据集，解压后放到以下路径：

```
data/
└── data_aishell/
    ├── wav/
    │   └── test/
    │       ├── S0764/
    │       └── ...
    └── transcript/
        └── aishell_transcript_v0.8.txt
```

### 测试干净音频

```bash
python evaluate.py
```

### 测试带噪音频

```bash
python evaluate_noisy.py
```

---

## 兼容性修复

Python 3.13 + PyTorch 2.12 环境下需要修复以下两处代码：

**修复1**：`model.py` 第20行

```python
# 修改前
def __init__(self, d_model=80, dropout_rate=0.1):
    pass

# 修改后
def __init__(self, d_model=80, dropout_rate=0.1):
    super().__init__()
```

**修复2**：`venv/Lib/site-packages/funasr/train_utils/load_pretrained_model.py` 第34行

```python
# 修改前
dst_state = obj.state_dict()

# 修改后
try:
    dst_state = obj.state_dict()
except AttributeError:
    for m in obj.modules():
        if not hasattr(m, '_state_dict_pre_hooks'):
            m._state_dict_pre_hooks = {}
        if not hasattr(m, '_state_dict_hooks'):
            m._state_dict_hooks = {}
    dst_state = obj.state_dict()
```

---

## 项目结构

```
SenseVoice/
├── model.py              # 模型核心代码（已修复兼容性）
├── webui.py              # WebUI 图形界面
├── test_run.py           # 快速测试脚本
├── evaluate.py           # 干净音频评测脚本
├── evaluate_noisy.py     # 带噪音频评测脚本
├── utils/                # 工具函数
├── data/
│   └── data_aishell/     # AISHELL-1 测试数据集
│       ├── wav/          # 音频文件（不含在仓库中）
│       └── transcript/   # 文字标注（不含在仓库中）
├── requirements.txt      # 依赖列表
└── README.md             # 项目说明
```

---

## 参考资料

- 论文：[FunAudioLLM: Voice Understanding and Generation Foundation Models](https://arxiv.org/abs/2407.04051)
- 模型主页：[ModelScope](https://www.modelscope.cn/models/iic/SenseVoiceSmall)
- 原始代码库：[FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice)
- 测试数据集：[AISHELL-1](https://www.openslr.org/33/)
