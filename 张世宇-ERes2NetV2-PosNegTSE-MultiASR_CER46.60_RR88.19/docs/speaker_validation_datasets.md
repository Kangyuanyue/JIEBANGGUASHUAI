# 声纹识别验证数据集来源清单

本清单用于验证本项目的说话人识别模块，包括同人/非同人验证、中文场景、远场会议、噪声增强和混响增强。

## 优先级 1：直接验证声纹识别

### VoxCeleb1 / VoxCeleb2

- 来源：https://www.robots.ox.ac.uk/~vgg/data/voxceleb/
- 类型：大规模英文名人公开视频语音，text-independent speaker verification 经典基准。
- 规模：VoxCeleb1 超过 150,000 utterances / 1,251 speakers；VoxCeleb2 超过 1,000,000 utterances / 6,112 speakers。
- 适合验证：通用声纹模型 EER、minDCF、跨信道鲁棒性。
- 门槛：音频下载需要在官网填表获取密码。
- 建议用法：先用 VoxCeleb1-O trials 做快速验证；VoxCeleb2 用于训练或大规模增强。

### CN-Celeb / CN-Celeb2

- 来源：https://www.openslr.org/82/
- 类型：清华 CSLT 发布的中文说话人识别数据集。
- 规模：CN-Celeb1 超过 130,000 utterances / 1,000 speakers；CN-Celeb2 超过 520,000 utterances / 2,000 speakers。
- 适合验证：中文声纹、真实场景、多 genre 条件，和本比赛更接近。
- 许可证：CC BY-SA 4.0。
- 建议用法：构造同 speaker 正样本和异 speaker 负样本；重点看相似音色、同语种和真实噪声环境下的 false accept。

### 3D-Speaker

- 来源：https://github.com/modelscope/3D-Speaker
- 论文：https://arxiv.org/abs/2403.19971
- 类型：ModelScope 开源 speaker verification / diarization 工具链和数据资源。
- 适合验证：中文、多设备、多距离、多场景声纹；也适合接 CAM++ / ERes2NetV2 模型。
- 建议用法：作为本项目 ECAPA 之外的主增强路线，验证 CAM++ 和 ERes2Net 系列。

## 优先级 2：中文/远场/重叠场景验证

### AISHELL-1

- 来源：https://www.openslr.org/33/
- 类型：开源普通话语音语料。
- 规模：400 人录制，16 kHz，安静室内环境。
- 许可证：Apache 2.0。
- 适合验证：中文短语音 enrollment/query 构造、基础同人/非同人验证。
- 建议用法：每个 speaker 取一句作为 wake，另一句作为 query；异 speaker 构造拒识样本。

### AISHELL-4

- 来源：https://www.openslr.org/111/
- 类型：中文多通道会议语音，真实会议场景。
- 规模：211 场会议，4-8 speakers/session，总计约 120 小时。
- 许可证：CC BY-SA 4.0。
- 适合验证：远场、多说话人、重叠语音、speaker activity。
- 建议用法：从标注里提取同一 speaker 片段构造正样本，其他 speaker 片段构造负样本；重叠区域用于压力测试。

### AliMeeting

- 来源：https://www.openslr.org/119/
- 类型：阿里巴巴发布的中文多通道多方会议语料。
- 规模：118.75 小时；Train/Eval/Test 分别为 104.75/4/10 小时；每场 2-4 人。
- 许可证：CC BY-SA 4.0。
- 适合验证：远场、近场对比、真实重叠说话、多人会议。
- 建议用法：用近场 headset 片段构造干净 enrollment，用远场阵列片段构造 query，模拟家居远场。

## 优先级 3：噪声和混响增强

### MUSAN

- 来源：https://www.openslr.org/17/
- 类型：music / speech / noise corpus。
- 大小：约 11G。
- 许可证：CC BY 4.0。
- 适合验证：背景音乐、人声噪声、环境噪声下的 speaker verification。
- 建议用法：给 AISHELL / CN-Celeb query 混入 -5 dB、0 dB、5 dB 噪声，测试动态阈值和误接收率。

### RIRS_NOISES

- 来源：https://www.openslr.org/28/
- 类型：真实和模拟房间脉冲响应、各向同性噪声和点声源噪声。
- 大小：约 1.3G。
- 许可证：Apache 2.0。
- 适合验证：远场混响、房间声学退化。
- 建议用法：对 wake/query 分别卷积不同 RIR，模拟不同麦克风位置和家居远场。

## 推荐验证顺序

1. AISHELL-1：最快做中文同人/非同人 sanity check。
2. CN-Celeb：验证中文真实场景声纹能力。
3. MUSAN + RIRS_NOISES：构造比赛要求的 -5 dB 到 5 dB、远场混响压力测试。
4. AISHELL-4 / AliMeeting：验证远场、重叠语音、多说话人干扰。
5. VoxCeleb1-O：对齐国际 speaker verification 标准指标。

## 统一 trials 格式

本项目的 `speaker_eval.py` 支持以下 CSV：

```csv
enroll_audio,test_audio,label
spk001/wake.wav,spk001/query.wav,1
spk001/wake.wav,spk009/query.wav,0
```

运行：

```bash
python speaker_eval.py --trials path/to/trials.csv --audio-root path/to/audio --output output/speaker_eval.json --score-dump output/speaker_scores.json
```

输出包括：

- EER
- minDCF
- best competition threshold
- positive / negative score mean
- 每条 trial 的分数和 backend_scores
