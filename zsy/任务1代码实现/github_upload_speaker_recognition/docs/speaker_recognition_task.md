# 声纹识别子任务实现方案

## 任务定位

本项目的声纹识别模块负责回答一个问题：给定唤醒音频 `W` 和待识别音频 `Q`，`Q` 是否包含与 `W` 同一目标发音人的语音。它直接影响比赛里的非目标说话人拒识率，也会影响正样本是否被误拒。

在本比赛中，声纹模块不应只做普通 speaker verification，还需要面向远场、低信噪比、短唤醒音频、重叠说话和相似音色负样本做鲁棒判断。

## 已实现工程能力

当前实现位置：

- `speaker_model.py`：声纹 embedding backend，默认 ECAPA，已预留 WavLM。
- `speaker_gate.py`：多模型融合门控、多裁剪打分、质量感知动态阈值。
- `speaker_eval.py`：声纹评测和阈值校准，输出 EER、minDCF 和比赛权重下的推荐阈值。
- `configs/default.json`：声纹 backend、权重和动态阈值配置。

默认配置只启用 ECAPA，保证现有项目稳定运行。需要做高阶实验时，可以将：

```json
"embedding_backends": ["ecapa", "wavlm"]
```

WavLM 需要额外安装 `transformers` 并下载 Hugging Face 模型。

## 推荐方法路线

### 1. 稳健基线：ECAPA-TDNN

ECAPA-TDNN 是当前代码默认模型，适合先跑出稳定 baseline。它通过通道注意力、多尺度 TDNN、统计池化提取 speaker embedding，再用余弦相似度判断同一说话人。

优点：成熟、快、工程稳定。  
缺点：短音频、强噪声、重叠语音下 embedding 容易被污染。

### 2. 主推增强：CAM++ / ERes2NetV2

3D-Speaker 工具链提供 CAM++、ERes2Net、ERes2NetV2 等模型。它们是更适合中文、远场、多场景声纹实验的候选模型。建议后续作为第二路声纹模型加入集成。

建议用途：

- 与 ECAPA 做 score fusion；
- 在中文数据集 CN-Celeb / 3D-Speaker 上验证；
- 重点看相似音色负样本和噪声负样本的 false accept。

### 3. 创新增强：WavLM Speaker Verification

WavLM 是自监督语音大模型，可用 speaker verification checkpoint 提取 x-vector 风格 embedding。它对噪声、信道变化和弱监督场景有优势，适合作为“创新模型路线”汇报和实验。

建议用途：

- ECAPA + WavLM 融合；
- 对比短唤醒音频下的稳定性；
- 低 SNR 下观察是否降低误拒。

### 4. 质量感知动态阈值

固定阈值在比赛里风险很高：噪声 wake 可能让同人分数下降，低 SNR query 可能让非目标人分数抬高。当前工程已实现动态阈值：

- wake 质量低时，提高阈值，避免脏 enrollment 误接收；
- query SNR 低时，提高阈值，避免噪声/重叠导致误接收；
- 后续可用开发集搜索这些 boost 参数。

### 5. 分数归一化和 hard negative

拿到数据后建议继续做：

- adaptive s-norm：用 cohort speaker 分数归一化，降低跨说话人分布偏移；
- hard negative mining：同文本、同男女、相似音色、电视背景人声作为负样本；
- quality-aware calibration：按 wake 时长、SNR、query speech ratio 分桶校准阈值。

### 6. 2025/2026 可作为创新点的方法

这些方法可以作为你负责模块的“创新储备”，其中一部分更适合复现思想，不一定立即作为主提交模型：

- SSPS / Bootstrapped Positive Sampling（2025）：自监督 speaker verification 的正样本采样改进，核心是从 embedding 空间寻找同说话人但不同信道/录音条件的伪正样本，降低模型学习到信道信息的风险。适合我们后续在无标签或弱标签中文数据上做自监督微调。
- DAME（2026）：Duration-Aware Matryoshka Embedding，针对短语音声纹不稳的问题，把不同长度语音映射到不同层级的嵌套 embedding。比赛的唤醒音频可能很短，这个思想很贴合。
- ReDimNet2（2026）：强调参数量和计算量 Pareto front 的 speaker encoder，适合效率分占 20% 的比赛约束。若有公开权重，可作为 ECAPA/CAM++ 的替代或补充。
- Curry / Curriculum Ranking Loss（2026）：利用样本难度分层训练，让模型先学稳定身份特征，再学困难边界。适合我们后续做 hard negative fine-tuning。
- SpeakerCard-1M（2026）：把 speaker profile 和文本属性引入 speaker verification。比赛不一定需要跨模态，但它提供 hard-negative triplets 和 VoxCeleb/CN-Celeb 衍生资源，可作为困难负样本构造参考。

工程上当前已接入最稳妥的创新落地点：WavLM 自监督声纹 backend + 多模型分数融合 + 质量感知动态阈值。等公开权重和数据准备好，再逐步试 ReDimNet2/DAME/CAM++。

## 推荐公开数据集

### VoxCeleb1 / VoxCeleb2

用途：国际通用 speaker verification 基准，适合验证模型泛化能力和 EER。  
建议：先用 VoxCeleb1-O trials 做快速验证，再用 VoxCeleb2 做训练/扩展。

### CN-Celeb / CN-Celeb2

用途：中文说话人识别数据，场景和语言更接近本比赛。  
建议：重点用于中文声纹阈值、同语言负样本、跨场景鲁棒性验证。

### 3D-Speaker

用途：多设备、多距离、多场景中文声纹数据和模型工具链。  
建议：用于远场家居、设备差异、中文声纹模型验证。

### AISHELL-1 / AISHELL-4 / AliMeeting

用途：中文 ASR/会议/远场多说话人数据。  
建议：可以构造本比赛 episode：speaker A 的一句话作为 wake，speaker A/B 的另一句话作为 query，并合成 overlap 和噪声。

### MUSAN + RIRS_NOISES

用途：噪声、音乐、人声干扰、混响增强。  
建议：构造 -5 dB、0 dB、5 dB 噪声条件，验证低 SNR 下声纹误拒和误接收。

## 实验命令

比赛 meta 校准：

```bash
python speaker_eval.py --meta path/to/meta.jsonl --audio-root path/to/audio --output output/speaker_eval.json --score-dump output/speaker_scores.json
```

标准 trials 校准：

```bash
python speaker_eval.py --trials path/to/trials.csv --audio-root path/to/audio --output output/speaker_eval.json
```

trials 文件格式：

```csv
enroll_audio,test_audio,label
spk1/a.wav,spk1/b.wav,1
spk1/a.wav,spk2/c.wav,0
```

## 阶段目标

第一阶段：ECAPA baseline + 动态阈值，跑通 EER/minDCF/比赛阈值。  
第二阶段：加入 WavLM 或 CAM++ 第二路模型，做 score fusion。  
第三阶段：构造噪声、远场、重叠、人声干扰 episode，分桶调阈值。  
第四阶段：加入 cohort s-norm 和 hard negative 校准，形成最终声纹模块。

## 参考链接

- CAM++: https://arxiv.org/abs/2303.00332
- 3D-Speaker Toolkit: https://arxiv.org/abs/2403.19971
- WavLM: https://arxiv.org/abs/2110.13900
- WavLM speaker checkpoint: https://huggingface.co/microsoft/wavlm-base-plus-sv
- SSPS speaker verification: https://arxiv.org/abs/2505.14561
- ReDimNet2: https://arxiv.org/abs/2603.11841
- DAME: https://arxiv.org/abs/2601.13999
- Curry loss: https://arxiv.org/abs/2603.24432
- SpeakerCard-1M: https://arxiv.org/abs/2606.03283
- VoxCeleb: https://www.robots.ox.ac.uk/~vgg/data/voxceleb/
- MUSAN: https://www.openslr.org/17/
- RIRS_NOISES: https://www.openslr.org/28/
