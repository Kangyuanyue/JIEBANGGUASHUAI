# 7.10 后正式训练方案

本文档记录 2026-07-10 FAQ 更新后的正式训练路线。核心调整是：`datasetA` 只能作为开发验证和 sanity check，不能继续作为训练集或候选命令表来源，否则会对测试集过拟合。

## 1. 规则变化带来的结论

主办方 FAQ 明确了以下几点：

- 测试集 A 只是开发、调参和临时排行榜参考，不决定最终入围。
- 初赛最终依据测试集 B 成绩和提交脚本核查。
- 最终客观指标为 CER 40%、RR 40%、推理效率 20%。
- 负样本只统计 RR，不统计 CER。
- 正样本如果被错误拒识，按删除错误计入 CER。
- 提交 JSON 不需要额外增加 RR、内存占用等字段。
- 评测会统一 GPU 环境，当前预计为 L20-46G。

因此后续策略必须从“DatasetA 上追高分”调整为“用公开数据和合成场景训练泛化能力，用 DatasetA 做只读验证”。

## 2. 之前工作的重新定位

保留为最终工程骨架：

- `ERes2NetV2 + CAM++` 双声纹特征。
- Paraformer 主 ASR。
- SenseVoiceSmall 作为辅助 ASR 或困难样本诊断工具。
- `command_grammar.py` 中的命令语法特征。
- LogisticRegression 融合决策器框架。
- 缓存式推理流程：先缓存声纹和 ASR，再融合判断。

不能直接作为最终训练结果：

- `output/datasetA_fusion_model.pkl`
- 用 DatasetA 全量标签生成的候选命令表。
- 用 DatasetA 全量标签调出来的阈值。
- DatasetA 上的 63 分离线结果。

它们只能作为原型验证和代码可行性证明。

## 3. 数据集使用规划

### 3.1 AISHELL-WakeUp-1-sample

本地路径：

`AISHELL-WakeUp-1-sample\AISHELL-WakeUp-1-sample`

当前结构：

- `SPEECHDATA/speech/text/160.txt`
- `SPEECHDATA/speech/wav/160.rar`
- `AISHELL-WakeUp-1.pdf`

用途：

- 构造唤醒音频 enrollment。
- 构造同说话人正样本和异说话人负样本。
- 训练/校准短唤醒音频下的声纹门控。

注意：

- 当前音频在 `160.rar` 中，需要先解压。
- `160.txt` 已有音频名和文本标注。

### 3.2 OpenSLR 120 / HI-MIA-CW

链接：

- https://www.openslr.org/120/

用途：

- 该资源是 HI-MIA 唤醒词的混淆词负样本补充。
- 适合构造“相似唤醒词 / 相似发音 / 非目标唤醒”的困难负样本。
- 用于提升 RR，尤其是降低假接受。

### 3.3 `data`

当前结构：

- `data\16k_wav_file`
- 已扫描到 16343 条 wav。
- 文件名形如 `0001_M02_01_fast_0001.wav`，第一段可作为 speaker-like id。
- 初步统计约 35 个 speaker-like id。

用途：

- 如果有文本标签：可构造智能家居命令 ASR 训练/验证数据。
- 如果没有文本标签：优先用于声纹验证、负样本构造、不同语速/说话人鲁棒性验证。

### 3.4 AISHELL 智能家居/开放语音数据

链接：

- https://www.aishelltech.com/kysjcp

用途：

- 补充中文命令、家居控制、常用口语的 ASR 和文本先验。
- 扩展命令语法词表，减少对 DatasetA 标签的依赖。

### 3.5 WHAM! / 噪声数据

链接：

- http://wham.whisper.ai/

用途：

- 合成 noisy command audio。
- 构造低信噪比、背景干扰、多说话人混合场景。
- 配合 MUSAN 和 RIRS_NOISES 做鲁棒性验证。

### 3.6 M2MeT / AliMeeting 方向

论文：

- https://arxiv.org/abs/2110.07393

用途：

- 该方向关注多说话人、重叠语音、远场会议转写。
- 不直接替代比赛数据，但可作为重叠语音构造和目标说话人增强的参考。

## 4. 正式训练数据构造

我们需要把外部数据统一构造成比赛式 episode：

```json
{
  "id": "episode_xxx",
  "wake_audio": "path/to/wake.wav",
  "wake_text": "你好米雅",
  "cmd_audio": "path/to/command.wav",
  "label": "打开客厅空调",
  "is_positive": true,
  "source": "aishell_wakeup",
  "speaker_id": "0001",
  "noise_type": "clean"
}
```

正样本构造：

- 同一 speaker 的 wake audio 和 command audio 配对。
- 如果没有命令文本，则只用于声纹验证，不用于 CER 训练。
- 同一 speaker 不同语速/不同录音条件优先作为困难正样本。

负样本构造：

- wake audio 来自 speaker A，command audio 来自 speaker B。
- 相同/相似文本但不同 speaker，构造 hard negative。
- 相似唤醒词和混淆词作为 false wake negative。

噪声与重叠构造：

- clean command + WHAM/MUSAN 噪声。
- clean command + RIRS 混响。
- target command + interfering speaker speech。
- 生成多个 SNR：`10dB / 5dB / 0dB / -5dB`。

## 5. 模型训练与校准

### 5.1 声纹门控

主模型：

- ERes2NetV2
- CAM++

训练/校准内容：

- 不微调大模型，先使用预训练 embedding。
- 在外部 episode 上校准双模型分数分布。
- 输出推荐阈值、低阈值、高阈值、不确定区间。
- 分 clean、noise、overlap 三类分别统计 EER、FAR、FRR、RR。

### 5.2 ASR 与文本后处理

主 ASR：

- Paraformer

辅助 ASR：

- SenseVoiceSmall，只在困难样本诊断或不确定样本触发。

文本处理：

- 官方 CER 归一化：NFKC、小写、去所有 Unicode 标点和空白。
- 命令语法词表必须来自公开数据、人工规则和外部训练集。
- 不使用 DatasetA 标签作为最终候选命令表。

### 5.3 融合决策器

输入特征：

- ERes2NetV2 分数和 segment statistics。
- CAM++ 分数和 segment statistics。
- wake/query 音频质量。
- query SNR、speech ratio、duration。
- Paraformer 文本语法特征。
- SenseVoice 文本语法特征。
- 双 ASR 一致性。
- 目标命令候选距离。

训练方式：

- 使用外部 episode train/dev 划分。
- speaker-disjoint 或 source-disjoint 验证优先。
- DatasetA 只作为最终只读 dev，不参与训练和候选构建。

输出：

- `output/final_fusion_model.pkl`
- `output/final_fusion_model_summary.json`
- `configs/final_inference.json`

## 6. 推理效率策略

最终脚本不能过度依赖慢速多模型全量推理。

默认流程：

1. ERes2NetV2 + CAM++ 声纹打分。
2. 明确拒识样本直接输出空文本。
3. 明确接受样本跑 Paraformer。
4. 不确定样本再触发 SenseVoice 或增强。
5. 只在不确定区使用目标说话人增强。

这样能兼顾 CER、RR 和效率。

## 7. 近期任务清单

### P0：立刻完成

- [ ] 对齐官方 CER 归一化。
- [ ] 整理 AISHELL-WakeUp 和 `data` 目录 metadata。
- [ ] 明确 DatasetA 不再作为训练数据。
- [ ] 生成外部数据 inventory。

### P1：外部 episode 构造

- [ ] 解压 AISHELL-WakeUp 的 `160.rar`。
- [ ] 解析 wakeup 文本标注。
- [ ] 从 `data/16k_wav_file` 解析 speaker-like id。
- [ ] 构造 clean speaker verification episode。
- [ ] 构造 hard negative episode。

### P2：鲁棒性增强

- [ ] 接入 WHAM/MUSAN/RIRS 噪声。
- [ ] 合成 overlap command audio。
- [ ] 生成 clean/noise/reverb/overlap 四套 dev。

### P3：正式训练

- [ ] 在外部 episode 上训练融合模型。
- [ ] DatasetA 只读验证迁移效果。
- [ ] 选择稳健阈值，而不是 DatasetA 最高阈值。

### P4：最终推理

- [ ] 做一键推理脚本。
- [ ] 支持无标签测试集。
- [ ] 输出官方 JSON。
- [ ] 统计本地耗时和内存。

## 8. 当前下一步

本轮先执行：

1. 修改 `metrics_cer.py`，对齐官方 CER。
2. 新增外部数据 metadata 整理脚本。
3. 扫描本地 AISHELL-WakeUp 和 `data`，输出 inventory。
4. 如果 `160.rar` 暂时无法解压，则先解析 `160.txt`，等待解压工具或手动解压后继续。

## 9. 2026-07-10 已执行进展

已完成：

- `metrics_cer.py` 已对齐官方 CER 归一化：
  - NFKC；
  - 小写；
  - 去 Unicode 标点；
  - 去空白字符；
  - RR 的空输出判断也使用同一归一化。
- 新增 `scripts/prepare_external_training_metadata.py`。
- 已生成外部数据 inventory：
  - `output/external_training_metadata/inventory.json`
  - `output/external_training_metadata/aishell_wakeup_160.jsonl`
  - `output/external_training_metadata/local_data_16k_wav_inventory.jsonl`
- 已为 `scripts/make_speaker_trials.py` 增加 `filename_prefix` speaker id 解析模式。
- 已生成外部声纹 trials：
  - `output/external_training_metadata/local_data_speaker_trials.csv`
- 已完成 200 条外部声纹 smoke eval：
  - `output/external_training_metadata/local_data_speaker_eval_eres2netv2_smoke.json`
  - `output/external_training_metadata/local_data_speaker_scores_eres2netv2_smoke.json`

本轮数据整理结果：

| 数据 | 数量 | 状态 |
| --- | ---: | --- |
| AISHELL-WakeUp 文本标注 | 15520 | 已解析 |
| AISHELL-WakeUp 音频 | 15520 | 已解压并链接 |
| `data/16k_wav_file` wav | 16343 | 已扫描 |
| `data/16k_wav_file` speaker-like id | 35 | 已解析 |
| 外部声纹 trials | 3497 | 已生成 |
| 外部 target/non-target trials | 2000 | 已生成 |

外部声纹 smoke eval：

| 模型 | trials | EER | 推荐阈值 | 说明 |
| --- | ---: | ---: | ---: | --- |
| ERes2NetV2 | 200 | 5.01% | 0.2692 | 仅 smoke eval，证明外部 trials 和声纹流水线可用 |

短唤醒注册 target/non-target smoke eval：

| 模型 | trials | EER | 推荐阈值 | 说明 |
| --- | ---: | ---: | ---: | --- |
| ERes2NetV2 | 200 | 3.48% | 0.3388 | AISHELL-WakeUp 正样本 + local_data 负样本 |

当前阻塞：

- `data\16k_wav_file` 当前未发现文本标签，只能先用于声纹验证和负样本构造。
- AISHELL-WakeUp sample 只有一个 speaker-like id：`160`，不能单独做跨说话人训练；需要与 `data` 或 OpenSLR 120 补充数据组合使用。

下一步优先：

1. 基于 `local_data_speaker_trials.csv` 和 `external_target_speaker_trials.csv` 跑更大规模 ERes2NetV2 / CAM++ 外部声纹评估。
2. 用外部 trials 重新校准声纹阈值，替代 DatasetA 阈值。
3. 接入 OpenSLR 120 的混淆唤醒词，构造 hard negative。
4. 构造 clean/noise/overlap 三类外部 episode，为正式融合模型做训练数据。

