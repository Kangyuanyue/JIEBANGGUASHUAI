## DiCoW / Target-Speaker ASR with Whisper

---

### 效果

* 目标说话人识别 + 双人重叠语音 + 非目标人抑制

---

### 代码与方法说明

* 上面是找到的代码，包含数据准备、训练、推理、评测和模型发布流程。通过逐帧说话人状态掩码，把 Silence、Target、Non-Target、Overlap 四类状态注入 Whisper 编码器，使 ASR 尽量只转写目标人

* 上述因为应用于中文智能家居指令有一些不太==稳定==，所以**FunASR**可以被关注，我也将其放到本项目中，FunASR 支持中文 ASR、VAD、说话人分离、流式推理和部署。https://github.com/modelscope/FunASR，只看看就行。

---

### 所以我建议后面项目可以这样整合：

```text
唤醒音频
   ↓
CAM++：提取目标人声纹
   ↓
待识别音频
   ↓
说话人分离 / p-VAD / overlap detection
   ↓
目标说话人帧级掩码 STNO
   ↓
DiCoW：仅识别目标人语音
   ↓
置信度 + 声纹一致性判断
   ↓
目标人 → 输出中文指令
非目标人 / 无目标人 → 输出空字符串
```

---

### DiCoW所用到的数据集：

* **NOTSOFAR-1**：真实会议场景，280 场会议，每场平均约 6 分钟，存在远场、混响、多人交叠等复杂条件

* **AMI Meeting Corpus**：真实会议语音数据，论文使用 AMI 的单远场麦克风测试集

* **Libri2Mix**：基于 LibriSpeech 合成的双说话人混合语音数据集
