# 开源代码、论文、模型和数据资源清单

> 说明：具体使用前请再次核查各项目 License、模型权重协议、数据集下载条款和比赛对外部数据 / 预训练模型的限制。

## 1. 主推荐开源生态

| 名称 | 类型 | 主要用途 | 是否适合本比赛 | 链接 |
|---|---|---|---|---|
| FunASR | 代码 / 模型生态 | 中文 ASR、VAD、标点、部署 | 很适合，用于快速 baseline 和 Paraformer 推理 | https://github.com/modelscope/FunASR |
| Paraformer-zh | ASR 模型 | 快速中文非自回归 ASR | 很适合，主 ASR 候选 | https://huggingface.co/funasr/paraformer-zh |
| SenseVoice | ASR / 多任务模型 | 中文 ASR、多语种、音频事件 | 适合，主模型或 teacher | https://github.com/FunAudioLLM/SenseVoice |
| WeNet | ASR 工具链 | 工业级端到端 ASR | 适合训练与部署参考 | https://github.com/wenet-e2e/wenet |
| ESPnet | 研究工具链 | ASR、增强、分离、diarization | 适合研究型实验和分离 recipe | https://github.com/espnet/espnet |
| Silero VAD | VAD 模型 | 轻量语音活动检测 | 很适合前置 VAD | https://github.com/snakers4/silero-vad |
| WeSpeaker | 说话人验证 | speaker embedding、SV、diarization | 很适合目标说话人验证 | https://github.com/wenet-e2e/wespeaker |
| 3D-Speaker | 说话人模型 | CAM++、ERes2Net、ECAPA | 很适合 wake / query embedding | https://github.com/modelscope/3D-Speaker |
| sherpa-onnx | 部署 | ONNX ASR / TTS / VAD 推理 | 适合效率优化 | https://github.com/k2-fsa/sherpa-onnx |

## 2. 目标说话人识别 / 提取方向

| 名称 | 类型 | 方法 | 适配价值 | 链接 |
|---|---|---|---|---|
| Personal VAD | 论文 | 使用目标 speaker embedding 做帧级目标语音检测 | 直接对应 target / non-target / silence 判断 | https://google.github.io/speaker-id/publications/PersonalVAD/ |
| VoiceFilter | 论文 | 使用 speaker embedding 生成谱掩码提取目标语音 | 适合目标与干扰人重叠场景 | https://google.github.io/speaker-id/publications/VoiceFilter/ |
| SpeakerBeam | 代码 / 方法 | 目标说话人提取 | 可借鉴 TSE 结构 | https://github.com/BUTSpeechFIT/speakerbeam |
| Asteroid | 代码库 | 语音分离 / 增强工具链 | 适合快速实现分离 baseline | https://github.com/asteroid-team/asteroid |
| SpeechBrain | 代码库 | 说话人识别、分离、ASR | 可借鉴 ECAPA、separation recipe | https://github.com/speechbrain/speechbrain |

## 3. ASR 论文与模型方向

| 名称 | 类型 | 方法 | 适配价值 | 链接 |
|---|---|---|---|---|
| Whisper | 论文 / 模型 | 大规模弱监督多语言 ASR | 鲁棒 teacher，不推荐作为主推理大模型 | https://arxiv.org/abs/2212.04356 |
| Conformer | 论文 | Transformer + CNN encoder | 高性能 ASR encoder 参考 | https://arxiv.org/abs/2005.08100 |
| Zipformer | 论文 / 代码生态 | 高效 encoder 结构 | 适合效率优化 | https://github.com/k2-fsa/icefall |
| icefall | 代码库 | k2 / Transducer / Zipformer recipe | 适合高效 ASR 研究与部署 | https://github.com/k2-fsa/icefall |

## 4. 中文 / 远场 / 噪声数据

| 名称 | 类型 | 用途 | 适配价值 | 链接 |
|---|---|---|---|---|
| AISHELL-1 | 中文 ASR 数据 | 普通话 ASR / 说话人数据补充 | 可用于 ASR 微调和 speaker 对构造 | https://www.openslr.org/33/ |
| WenetSpeech | 大规模中文 ASR 数据 | ASR 预训练 / 微调 | 中文覆盖广 | https://www.openslr.org/121/ |
| AISHELL-4 | 远场会议数据 | 多人、远场、重叠 | 适合 overlap / far-field 验证 | https://www.openslr.org/111/ |
| AliMeeting | 中文会议数据 | 远场、多说话人、重叠 | 适合多人重叠鲁棒性训练 | https://www.openslr.org/119/ |
| MUSAN | 噪声数据 | 噪声、音乐、人声干扰 | 适合 SNR 和背景噪声增强 | https://www.openslr.org/17/ |
| RIRS_NOISES | RIR / 噪声 | 混响和噪声增强 | 适合远场模拟 | https://www.openslr.org/28/ |

## 5. 推荐组合

### 快速 baseline

```text
Silero VAD + 3D-Speaker CAM++ + Paraformer-zh + 简单阈值拒识
```

### 高分主方案

```text
Silero/FunASR VAD
+ CAM++ / ERes2Net 双 speaker embedding
+ target-speaker pVAD
+ 条件式 VoiceFilter/SpeakerBeam TSE
+ Paraformer-zh / SenseVoice-small
+ 家居命令后处理
+ 融合拒识决策
```

### 决赛创新增强

```text
Joint Target-Speaker ASR
+ auxiliary target/non-target VAD head
+ ASR-aware TSE
+ teacher-student distillation
+ ONNX / TensorRT 部署
```
