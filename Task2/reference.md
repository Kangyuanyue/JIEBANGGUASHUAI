# 参考文献与资源链接


## 一、论文链接

### 1. Qwen3-ASR
- **论文标题**：Qwen3-ASR Technical Report
- **arXiv链接**：https://arxiv.org/abs/2601.21337
- **说明**：阿里通义实验室，2026年1月发布

### 2. FunAudioLLM (SenseVoice + CosyVoice)
- **论文标题**：FunAudioLLM: Voice Understanding and Generation Foundation Models for Natural Interaction Between Humans and LLMs
- **arXiv链接**：https://arxiv.org/abs/2407.04051
- **说明**：阿里通义语音团队，2024年7月发布

### 3. Paraformer
- **论文标题**：Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
- **arXiv链接**：https://arxiv.org/abs/2206.08317
- **说明**：INTERSPEECH 2022

### 4. FireRedASR2S
- **论文标题**：FireRedASR2S: A State-of-the-Art Industrial-Grade All-in-One Automatic Speech Recognition System
- **arXiv链接**：https://arxiv.org/abs/2603.10420
- **说明**：FireRedTeam，2026年3月发布

### 5. Whisper
- **论文标题**：Robust Speech Recognition via Large-Scale Weak Supervision
- **arXiv链接**：https://arxiv.org/abs/2212.04356
- **说明**：OpenAI，2023年（ICML）

### 6. Mega-ASR
- **论文标题**：Mega-ASR: Towards In-the-wild^2 Speech Recognition via Scaling up Real-world Acoustic Simulation
- **arXiv链接**：https://arxiv.org/abs/2605.19833
- **说明**：南洋理工大学、新加坡国立大学等，2026年5月发布

### 7. Fun-ASR
- **论文标题**：Fun-ASR Technical Report
- **arXiv链接**：https://arxiv.org/abs/2509.12508
- **说明**：阿里通义实验室，2025年9月发布

### 8. FM-Refiner
- **论文标题**：Latent-Level Enhancement with Flow Matching for Robust Automatic Speech Recognition
- **arXiv链接**：https://arxiv.org/abs/2601.04459
- **说明**：2026年1月发布

### 9. FormalASR
- **论文标题**：FormalASR: End-to-End Spoken Chinese to Formal Text
- **arXiv链接**：https://arxiv.org/abs/2605.19266
- **说明**：2026年5月发布

### 10. Moonshine
- **论文标题**：Flavors of Moonshine: Tiny Specialized ASR Models for Edge Devices
- **arXiv链接**：https://arxiv.org/abs/2509.02523
- **说明**：Useful Sensors，2025年9月发布

### 11. SpecASR
- **论文标题**：SpecASR: Accelerating LLM-based Automatic Speech Recognition via Speculative Decoding
- **arXiv链接**：https://arxiv.org/abs/2505.21067
- **说明**：北京大学，2025年5月发布


## 二、代码与模型链接

### 1. SenseVoiceSmall
- **HuggingFace模型页**：https://huggingface.co/FunAudioLLM/SenseVoiceSmall
- **ModelScope模型页（国内镜像）**：https://www.modelscope.cn/models/iic/SenseVoiceSmall
- **GitHub代码库**：https://github.com/FunAudioLLM/SenseVoice
- **开源许可**：Model License
- **说明**：主线候选ASR模型，234M参数，支持5种语言，非自回归架构，10s音频延迟仅70ms

### 2. Paraformer-zh
- **HuggingFace模型页**：https://huggingface.co/funasr/paraformer-zh
- **ModelScope模型页（国内镜像）**：https://www.modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
- **GitHub代码库**：https://github.com/modelscope/FunASR
- **开源许可**：Apache-2.0
- **说明**：稳定基线ASR模型，220M参数，非自回归架构，RTF仅0.009

### 3. Qwen3-ASR-0.6B
- **HuggingFace模型页**：https://huggingface.co/Qwen/Qwen3-ASR-0.6B
- **ModelScope模型页（国内镜像）**：https://www.modelscope.cn/models/Qwen/Qwen3-ASR-0.6B
- **GitHub代码库**：https://github.com/QwenLM/Qwen3-ASR
- **开源许可**：Apache-2.0
- **说明**：冲高分候选ASR模型，0.6B参数，支持52种语言+22种中文方言，LALM范式，首Token延迟92ms

### 4. FireRedASR-AED-L
- **HuggingFace模型页**：https://huggingface.co/FireRedTeam/FireRedASR-AED-L
- **GitHub代码库**：https://github.com/FireRedTeam/FireRedASR
- **开源许可**：Apache-2.0
- **说明**：Teacher/高精度上界，1B+参数，AISHELL-1 CER仅0.57%

### 5. Whisper-large-v3-turbo
- **HuggingFace模型页**：https://huggingface.co/openai/whisper-large-v3-turbo
- **GitHub代码库**：https://github.com/openai/whisper
- **开源许可**：MIT
- **说明**：鲁棒性对照模型，多语言ASR，自回归架构

### 6. sherpa-onnx
- **GitHub代码库**：https://github.com/k2-fsa/sherpa-onnx
- **在线文档**：https://k2-fsa.github.io/sherpa/onnx/
- **开源许可**：Apache-2.0
- **说明**：ONNX推理部署框架，跨平台支持，提供12种编程语言API

### 7. Mega-ASR
- **HuggingFace模型页**：https://huggingface.co/zhifeixie/Mega-ASR
- **GitHub代码库**：https://github.com/xzf-thu/Mega-ASR
- **开源许可**：MIT
- **说明**：复合声学环境抗噪SOTA，基于Qwen3-ASR 1.7B微调

### 8. FunASR
- **GitHub代码库**：https://github.com/alibaba/FunASR
- **说明**：阿里语音识别工具包，集成Paraformer等多种模型

### 9. FM-Refiner
- **GitHub代码库**：https://github.com/sp-uhh/sgmse
- **说明**：流匹配潜层增强模块，即插即用抗噪，支持与ESPnet/SpeechBrain等框架集成

### 10. FormalASR
- **HuggingFace模型页**：https://huggingface.co/TaurenMountain/FormalASR-0.6B
- **GitHub代码库**：https://github.com/TaurenMountain/FormalASR
- **说明**：端到端口语转书面语，支持GGUF量化，Q8_0量化近无损

### 11. Moonshine
- **GitHub代码库**：https://github.com/moonshine-ai/moonshine
- **开源许可**：宽松许可证
- **说明**：27M超轻量端侧专用ASR模型，已发布中文版本


## 三、模型下载地址

### SenseVoiceSmall
- **HuggingFace**：https://huggingface.co/FunAudioLLM/SenseVoiceSmall
- **ModelScope**：https://www.modelscope.cn/models/iic/SenseVoiceSmall

### Paraformer-zh
- **HuggingFace**：https://huggingface.co/funasr/paraformer-zh
- **ModelScope**：https://www.modelscope.cn/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch

### Qwen3-ASR-0.6B
- **HuggingFace**：https://huggingface.co/Qwen/Qwen3-ASR-0.6B
- **ModelScope**：https://www.modelscope.cn/models/Qwen/Qwen3-ASR-0.6B

### FireRedASR-AED-L
- **HuggingFace**：https://huggingface.co/FireRedTeam/FireRedASR-AED-L

### sherpa-onnx 预转换模型
- **Qwen3-ASR-0.6B ONNX**：https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-qwen3-asr-0.6b-zh-en-2026-03-29.tar.bz2


