# 复杂交互场景的抗干扰语音指令识别系统

## 题目编号：XH-202615

## 项目简介

本项目为"挑战杯"竞赛——美的集团"复杂交互场景的抗干扰语音指令识别技术"赛题的参赛作品。

系统设计了一套结合**唤醒音频的发音人信息**，有效**识别远场家居控制语音指令**，并**拒识非唤醒说话人**语音的智能语音识别系统。

## 系统架构

```
唤醒音频 (kws) ──→ CAM++ 说话人嵌入 ──┐
                                         ├── 余弦相似度 ──→ 目标人判定
识别音频 (cmd) ──→ CAM++ 说话人嵌入 ──┘        │
                                          ┌─────┘
                                          ↓
                                    是目标人？
                                    ├── 是 → 语音增强 → FunASR(Paraformer) → 后处理 → 识别文本
                                    └── 否 → 拒识（输出空串）
```

### 核心模块

| 模块 | 文件 | 说明 |
|------|------|------|
| 全局配置 | `src/config.py` | 路径、参数、阈值配置 |
| 数据加载 | `src/data_loader.py` | JSONL解析、音频读取、数据增强 |
| 说话人验证 | `src/speaker_verify.py` | CAM++/MFCC说话人嵌入 + 余弦相似度 |
| 语音增强 | `src/speech_enhance.py` | 频谱减法降噪、VAD、预加重 |
| ASR引擎 | `src/asr_engine.py` | FunASR Paraformer / Whisper |
| 后处理 | `src/postprocess.py` | 文本规范化、领域纠错、CER计算 |
| 推理Pipeline | `src/pipeline.py` | 完整推理流程编排 |
| 推理入口 | `run_inference.py` | 主推理脚本 |
| 评估报告 | `run_evaluation.py` | 测试报告与可视化生成 |

## 环境要求

- Python 3.9+
- PyTorch 2.0+
- CUDA (可选，用于GPU加速)

### 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 运行推理

```bash
# 默认配置运行
python run_inference.py

# 自定义参数
python run_inference.py --threshold 0.75 --asr funasr --enhance
```

### 2. 生成测试报告

```bash
python run_evaluation.py
```

### 3. 输出结果

- 竞赛提交JSON: `output/result.json`
- 详细结果: `output/detailed_results.json`
- 测试报告: `output/report/test_report.txt`
- 可视化图表: `output/figures/`

## 评估指标

| 指标 | 权重 | 说明 |
|------|------|------|
| CER (字错率) | 40% | 目标发音人语音内容识别准确度 |
| RR (拒识率) | 40% | 正确拒识非目标发音人语音的比例 |
| 推理效率 | 20% | 推理时间(10%) + 内存占用(10%) |

## 抗干扰策略

1. **说话人验证**：利用CAM++深度说话人嵌入，准确区分目标/非目标发音人
2. **频谱减法降噪**：降低环境噪声对ASR的影响
3. **领域热词优化**：针对智能家居指令的热词增强
4. **文本后处理**：领域纠错、规范化，减少常见ASR错误

## 提交格式

```json
{
  "result": {
    "results": [
      {"id": "pos/cmd_0", "content": "识别结果", "label": "标签", "cer": "0.0000"},
      ...
    ],
    "final_cer": "0.0500",
    "duration": "120.00"
  }
}
```
