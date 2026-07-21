# 复杂交互场景的抗干扰语音指令识别

## 项目结构

```
yuyinshibie/
├── scripts/
│   └── infer_datasetA.py          # 推理脚本：ASR转写 + CER评估
├── model/
│   └── funasr/                    # FunASR 模型源码
├── output/
│   ├── datasetA_result.txt        # 推理结果（逐条）
│   └── datasetA_summary.txt       # 整体CER摘要
└── readme.md
```

## 运行方式

### 环境要求

- Python 3.10+
- CUDA 12.4+, GPU 8GB+
- Windows 11

### 依赖安装

```bash
pip install funasr jiwer soundfile numpy torch torchaudio
```

### 推理

```bash
cd scripts
python infer_datasetA.py
```

输出：
- `output/datasetA_result.txt` — 逐条ID、转写结果、标注文本、CER
- `output/datasetA_summary.txt` — 整体CER、样本数、耗时

## 模型架构

```
cmd音频 → paraformer-zh → 转写文本 → 与标注比对 → CER
```

### 预训练模型

| 模型 | 来源 | 用途 |
|------|------|------|
| paraformer-zh (SeacoParaformer) | FunASR / ModelScope | 中文语音识别 |

## 测试结果

### DatasetA (开发集)

| 指标 | 数值 |
|------|------|
| 样本数 | 1364 |
| 整体 CER | ~40% |

## 后续方向

1. 唤醒词声纹引导的目标说话人提取（TSE）
2. AISHELL-1 仿真数据训练分离模型
3. 端到端 Target-Speaker ASR
