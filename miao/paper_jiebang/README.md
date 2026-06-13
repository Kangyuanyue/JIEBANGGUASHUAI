# paper_jiebang：抗干扰语音指令识别论文库

本文件夹用于存放“复杂交互场景的抗干扰语音指令识别技术”相关论文，服务于 `miao` 方案中的算法设计、技术路线论证、模型选型、实验对比和答辩材料整理。

本题不是普通 ASR，而是 **基于唤醒音频的目标说话人条件化语音识别任务**：系统需要利用唤醒音频确定目标发音人，在远场、低信噪比、多说话人重叠、非人声噪声等复杂环境下识别目标人的控制指令，并拒识非目标说话人。

---

## 1. 论文库研究地图

| 方向 | 代表论文 | 对本项目的价值 |
|---|---|---|
| 目标语音提取综述 | Neural Target Speech Extraction: An overview | 理解 TSE 的整体脉络、常见结构和评价指标 |
| 目标说话人提取 | SpeakerBeam、TEnet、OR-TSE、Speaker-Aware Enhancement | 支撑“唤醒音频 embedding + 条件式目标语音提取”模块 |
| 目标说话人 ASR | Joint Target-Speaker ASR and Activity Detection、Streaming End-to-End TS-ASR、Conformer-based TS-ASR | 支撑“只识别唤醒人、不识别干扰人”的核心模型设计 |
| 重叠语音识别 | Improving End-to-End Single-Channel Multi-Talker Speech Recognition、Multi-channel Overlapped Speech Recognition | 解决目标人与干扰人同时说话的问题 |
| 远场与鸡尾酒会 | End-to-End Dereverberation, Beamforming, and Speech Recognition in a Cocktail Party | 支撑远场、混响、多干扰条件下的系统鲁棒性设计 |
| 高效 ASR 编码器 | Conformer、Zipformer | 支撑主干网络选择和推理效率优化 |
| 噪声鲁棒 ASR | Two-Step Joint Optimization with Auxiliary Loss Function for Noise-Robust Speech Recognition | 支撑低 SNR、空调噪声、非人声噪声下的模型优化 |

---

## 2. 已收录论文清单

### 2.1 目标说话人提取 / Target Speech Extraction

1. **Neural Target Speech Extraction: An overview**  
   用途：目标语音提取方向综述，适合作为 TSE 方向的入门和技术路线总览。

2. **SpeakerBeam: Speaker Aware Neural Network for Target Speaker Extraction in Speech Mixtures**  
   用途：非常贴合本项目“唤醒音频提供目标说话人信息，后续音频只提取目标说话人”的任务设定。

3. **Speaker-Aware Target Speaker Enhancement by Jointly Learning with Speaker Embedding Extraction**  
   用途：支撑 speaker embedding 与语音增强联合训练的思路。

4. **TEnet: target speaker extraction network with accumulated speaker embedding for automatic speech recognition**  
   用途：参考“目标说话人提取 + ASR”的串联系统，适合用于重叠样本的条件式增强路径。

5. **OR-TSE: An Overlap-Robust Speaker Encoder for Target Speech Extraction**  
   用途：解决重叠语音条件下 speaker embedding 被干扰人污染的问题。

---

### 2.2 目标说话人自动语音识别 / Target-Speaker ASR

6. **Joint Target-Speaker ASR and Activity Detection**  
   用途：对应本项目的核心思想：不仅要识别文本，还要判断目标说话人是否正在说话。

7. **Streaming End-to-End Target-Speaker Automatic Speech Recognition and Activity Detection**  
   用途：支撑低延迟、在线式智能家居语音控制场景。

8. **End-to-End Target Speaker speech recognition with voice activity detection fusion**  
   用途：支撑 pVAD / VAD 与 ASR 融合，降低非目标人被误识别为有效指令的风险。

9. **Conformer-based Target-Speaker Automatic Speech Recognition for Single-Channel Audio**  
   用途：直接对应单通道远场家居语音场景，可作为 Conformer / Paraformer / Zipformer 主干选择依据。

10. **Simultaneous Speech Recognition and Speaker Diarization for Monaural Dialogue Recordings with Target-Speaker Acoustic Models**  
    用途：参考“识别 + 说话人区分”联合建模思路。

---

### 2.3 重叠语音与多说话人识别

11. **Improving End-to-End Single-Channel Multi-Talker Speech Recognition**  
    用途：用于解决单通道多说话人同时说话导致 ASR 输出混乱的问题。

12. **Multi-channel overlapped speech recognition with location guided speech extraction network**  
    用途：其“先提取目标语音、再识别”的思路可迁移到 speaker embedding 引导的目标提取路径。

13. **Far-field location guided target speech extraction using end-to-end speech recognition objectives**  
    用途：支撑远场目标语音提取，并强调增强模块应面向 ASR 识别目标优化。

---

### 2.4 远场、去混响与鲁棒前端

14. **End-to-End Dereverberation, Beamforming, and Speech Recognition in a Cocktail Party**  
    用途：支撑远场、多干扰、混响条件下的“前端增强 + 后端 ASR”联合系统。

15. **基于计算听觉场景分析和语者模型信息的语音识别鲁棒性前端研究**  
    用途：中文鲁棒语音识别前端研究，可作为计算听觉场景分析和语者信息利用的理论参考。

---

### 2.5 高效 ASR 编码器与主干结构

16. **Conformer: Convolution-augmented Transformer for Speech Recognition**  
    用途：高性能 ASR encoder 参考，兼具 CNN 局部建模能力和 Transformer 全局建模能力。

17. **Zipformer: A faster and better encoder for automatic speech recognition**  
    用途：支撑推理效率优化，与比赛中的 batch=1 推理时间和内存占用指标对应。

---

### 2.6 噪声鲁棒语音识别

18. **Two-Step Joint Optimization with Auxiliary Loss Function for Noise-Robust Speech Recognition**  
    用途：支撑低 SNR、空调噪声、非人声噪声等复杂环境下的 ASR 微调与多目标损失设计。

---

### 2.7 待确认相关性论文

19. **A track-before-detect algorithm based on multi-coordinate system**  
    说明：该论文标题更偏目标检测 / 跟踪方向，与语音识别任务直接相关性较弱。建议确认是否误放；若作为交叉参考，可保留，否则建议移至 `paper_jiebang/other_reference/`。

---

## 3. 与比赛方案的对应关系

| 比赛关键问题 | 对应论文方向 | 推荐优先阅读 |
|---|---|---|
| 如何拒识非目标说话人 | Target-Speaker ASR、pVAD、Speaker Verification | Joint Target-Speaker ASR and Activity Detection；End-to-End TS-ASR with VAD Fusion |
| 如何处理目标与干扰人同时说话 | Target Speech Extraction、Overlapped ASR | SpeakerBeam；Neural Target Speech Extraction；OR-TSE |
| 如何降低目标人 CER | ASR encoder、ASR-aware enhancement | Conformer；Zipformer；Conformer-based TS-ASR |
| 如何适配远场和混响 | Dereverberation、Beamforming、Robust Front-end | End-to-End Dereverberation, Beamforming, and Speech Recognition in a Cocktail Party |
| 如何提高低 SNR 鲁棒性 | Noise-Robust ASR、辅助损失、多条件增强 | Two-Step Joint Optimization with Auxiliary Loss Function for Noise-Robust Speech Recognition |
| 如何兼顾推理效率 | 高效 encoder、条件式分支、轻量化部署 | Zipformer；Conformer；TEnet |

---

## 4. 推荐阅读顺序

### 第一阶段：理解任务本质

1. Neural Target Speech Extraction: An overview  
2. SpeakerBeam: Speaker Aware Neural Network for Target Speaker Extraction in Speech Mixtures  
3. Joint Target-Speaker ASR and Activity Detection  

目标：搞清楚为什么本题不能只做普通 ASR，而必须做“目标说话人条件化识别”。

### 第二阶段：支撑核心算法设计

4. Conformer-based Target-Speaker Automatic Speech Recognition for Single-Channel Audio  
5. Streaming End-to-End Target-Speaker Automatic Speech Recognition and Activity Detection  
6. Speaker-Aware Target Speaker Enhancement by Jointly Learning with Speaker Embedding Extraction  
7. OR-TSE: An Overlap-Robust Speaker Encoder for Target Speech Extraction  

目标：支撑 speaker embedding、pVAD、TSE、ASR 融合结构。

### 第三阶段：补充复杂场景鲁棒性

8. Improving End-to-End Single-Channel Multi-Talker Speech Recognition  
9. End-to-End Dereverberation, Beamforming, and Speech Recognition in a Cocktail Party  
10. Two-Step Joint Optimization with Auxiliary Loss Function for Noise-Robust Speech Recognition  

目标：解决重叠说话、远场混响和噪声条件下的稳定性问题。

### 第四阶段：工程效率与模型主干

11. Conformer: Convolution-augmented Transformer for Speech Recognition  
12. Zipformer: A faster and better encoder for automatic speech recognition  
13. TEnet: target speaker extraction network with accumulated speaker embedding for automatic speech recognition  

目标：为最终模型选择、压缩加速和部署优化提供依据。

---

## 5. 后续阅读笔记建议

建议在本目录下新增 `notes/` 文件夹，每读完一篇论文就补充一份阅读笔记：

```text
paper_jiebang/
  notes/
    01_neural_target_speech_extraction_overview.md
    02_speakerbeam.md
    03_joint_target_speaker_asr_activity_detection.md
```

每篇阅读笔记建议包含：

```markdown
# 论文标题

## 1. 这篇论文解决什么问题

## 2. 核心方法

## 3. 和本比赛的关系

## 4. 可以直接借鉴的模块

## 5. 不适合直接照搬的地方

## 6. 可用于 PPT / 答辩的一句话总结
```

---

## 6. 本论文库对最终方案的支撑

本论文库服务于 `miao` 技术方案中的六个核心模块：

1. Wake Speaker Enrollment：从唤醒音频中提取目标说话人 embedding；
2. Speaker Verification / Target Decision：判断识别音频是否来自目标发音人；
3. Target-Speaker pVAD：判断每一帧是目标人、非目标人还是静音；
4. Target Speaker Extraction：在目标人和干扰人重叠时提取目标语音；
5. Robust ASR：在低 SNR、远场、空调噪声等复杂环境下识别控制指令；
6. Efficiency Optimization：通过 Conformer / Zipformer / 条件式 TSE / 模型压缩兼顾推理速度和内存占用。

论文库不是简单堆论文，而是围绕比赛评分指标构建的技术证据链：

```text
目标发音人 CER 下降
  <- 目标说话人提取 + 鲁棒 ASR + 家居指令后处理

非目标拒识率 RR 提升
  <- Speaker Verification + pVAD + 融合拒识决策

推理效率提升
  <- 轻量前置门控 + 条件式 TSE + 高效 ASR encoder + 模型量化
```

---

## 7. 文件管理规范

建议后续新增论文时采用统一命名方式：

```text
年份_第一作者_论文关键词.pdf
```

示例：

```text
2023_Moriya_Streaming_Target_Speaker_ASR_AD.pdf
2023_Zhang_Conformer_Target_Speaker_ASR.pdf
2019_Zmolikova_SpeakerBeam_Target_Speaker_Extraction.pdf
```

这样方便按年份、方向和算法模块检索，也避免文件名过长导致 GitHub 页面显示被截断。

---

## 8. 使用说明

本文件夹中的论文主要用于比赛学习、算法调研和团队内部技术路线论证。若后续在报告、PPT、论文或开源项目中引用相关工作，应在正式材料中规范列出参考文献。
