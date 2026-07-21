# 说话人识别模块阶段性工作总结

本文档整理本项目在“说话人识别 / 声纹识别”方向已经完成的工作，方便后续上传 GitHub 后让其他成员快速理解：我们为什么这样做、用了哪些数据和模型、目前效果如何，以及后续接入比赛开发集时应该怎么继续优化。

## 1. 任务理解

比赛题目是“复杂交互场景的抗干扰语音指令识别技术”。从系统角度看，它不是单纯做一句话的 ASR，而是要在复杂噪声、多人说话、远场混响、非目标用户干扰等条件下，判断语音指令是否来自目标用户，并尽可能正确识别指令内容。

我们当前负责的是其中的说话人识别部分，核心目标是：

- 判断当前语音是否来自已注册的目标说话人；
- 拒绝非目标说话人的语音指令，降低误触发风险；
- 在噪声、混响、短语音指令等场景下保持稳定；
- 为后续 ASR、命令理解和最终决策模块提供可靠的声纹分数与置信区间。

因此，本模块不是只输出“相似 / 不相似”，而是输出一个更适合比赛系统使用的门控结果：

```text
clear_reject
uncertain_below_threshold
accepted_uncertain_band
clear_accept
```

这样可以让系统在高置信样本上快速决策，在边界样本上结合 ASR、音频质量、命令先验等信息继续判断。

## 2. 已使用的数据集

当前还没有比赛官方开发集，因此我们先使用公开数据集构建可复现的验证流程。

| 数据集 | 用途 | 当前状态 | 来源 |
|---|---|---|---|
| CN-Celeb2 v2 | 中文说话人识别主验证集，用于模型对比和阈值校准 | 已下载、拼接、解压并完成全量评估 | [CN-Celeb / CN-Celeb2 官方项目页](http://www.openslr.org/resources/82/) |
| MUSAN | 噪声、音乐、人声干扰增强 | 已下载并用于噪声增强验证 | [OpenSLR MUSAN](https://www.openslr.org/17/) |
| RIRS_NOISES | 房间脉冲响应与混响增强 | 已下载并用于远场混响验证 | [OpenSLR RIRS_NOISES](https://www.openslr.org/28/) |

本地数据位置如下：

```text
CN-Celeb2:   CN-Celeb2_flac/data
MUSAN:       musan
RIRS_NOISES: RIRS_NOISES
```

注意：这些数据集体积较大，不建议直接上传到 GitHub。仓库中应保留下载说明、处理脚本和实验结果摘要。

## 3. 模型选型与判断

我们先以原有 ECAPA-TDNN 声纹模型作为基线，然后接入 3D-Speaker 中的中文预训练模型进行对比。

已接入的模型包括：

- ECAPA-TDNN：原始基线模型；
- CAM++：3D-Speaker 中文声纹模型；
- ERes2NetV2：3D-Speaker 中文声纹模型，目前作为主模型；
- WavLM：曾作为候选方向调研，但当前环境下加载成本和稳定性不理想，暂不作为主路线。

最终选择 ERes2NetV2 的原因：

- 在 CN-Celeb2 快速验证集上 EER 最低；
- 在全量 CN-Celeb2 trials 上明显优于 ECAPA；
- 中文说话人场景适配更好；
- 推理链路已经可以稳定接入当前工程；
- 相比简单模型融合，单独使用 ERes2NetV2 更稳定、更容易解释和校准。

当前默认配置已经切换为：

```json
"embedding_backends": ["eres2netv2"],
"aggregate": "topk_mean",
"top_k": 3
```

## 4. 工程实现内容

本阶段主要完成了以下工程改造：

| 文件 | 主要改动 |
|---|---|
| `speaker_model.py` | 接入 3D-Speaker 的 CAM++ 和 ERes2NetV2 后端，支持本地预训练权重加载 |
| `speaker_gate.py` | 支持多后端声纹打分、多片段聚合、二段阈值和动态阈值 |
| `config.py` / `configs/default.json` | 将默认声纹模型切换到 ERes2NetV2，并写入校准后的阈值 |
| `speaker_eval.py` | 新增声纹验证评估入口，支持 trials 评估、EER、minDCF、比赛加权分数和分数导出 |
| `scripts/make_speaker_trials.py` | 从 CN-Celeb2 构造正负样本 trials |
| `scripts/calibrate_speaker_thresholds.py` | 根据分数文件自动校准主阈值、拒绝阈值和高置信接受阈值 |
| `scripts/mine_speaker_hard_cases.py` | 挖掘高分负样本和低分正样本，构建困难样本集 |
| `scripts/build_augmented_speaker_trials.py` | 使用 MUSAN 和 RIRS_NOISES 构造噪声、混响和噪声+混响验证集 |

对应的阶段性文档已经放在：

```text
docs/cnceleb2_speaker_baseline.md
docs/speaker_hard_negative_calibration.md
docs/noise_reverb_speaker_eval.md
docs/speaker_validation_datasets.md
docs/speaker_recognition_task.md
```

## 5. CN-Celeb2 主验证结果

我们使用 CN-Celeb2 v2 构造了全量验证 trials：

```text
trials: output/cnceleb2_trials_full_2to20s.csv
总数: 7902
正样本: 3918
负样本: 3984
音频时长筛选: 2s <= duration <= 20s
```

### 5.1 模型对比结果

| 模型 | Trials | EER | 推荐阈值 | 比赛加权分数 | 正样本均值 | 负样本均值 |
|---|---:|---:|---:|---:|---:|---:|
| ECAPA topk_mean k=3 | 7902 | 13.38% | 0.2668 | 69.70 | 0.4755 | 0.1302 |
| ERes2NetV2 topk_mean k=3 | 7902 | 4.09% | 0.3371 | 76.95 | 0.6468 | 0.1251 |

结论：ERes2NetV2 相比原始 ECAPA 基线有明显提升，是当前最适合作为主声纹模型的方案。

## 6. 阈值校准结果

基于 ERes2NetV2 在 CN-Celeb2 全量 trials 上的分数，我们完成了阈值校准。

当前默认阈值：

```json
{
  "threshold": 0.3370734058066252,
  "speaker_reject_low": 0.23909343270879338,
  "speaker_accept_high": 0.37844391932749266
}
```

不同工作点如下：

| 工作点 | 阈值 | 目标人召回率 | 非目标拒识率 | FAR | FRR |
|---|---:|---:|---:|---:|---:|
| EER | 0.3017 | 95.92% | 95.91% | 4.09% | 4.08% |
| 比赛加权推荐 | 0.3371 | 94.44% | 97.94% | 2.06% | 5.56% |
| RR99 | 0.3784 | 92.70% | 99.02% | 0.98% | 7.30% |

由于比赛中“误接受非目标说话人”通常风险更高，因此当前主阈值选择 `0.3371`，比 EER 阈值更偏向降低误接受。

二段阈值解释：

```text
score < 0.2391       -> clear_reject
0.2391 ~ 0.3371      -> uncertain_below_threshold
0.3371 ~ 0.3784      -> accepted_uncertain_band
score >= 0.3784      -> clear_accept
```

这种设计的好处是：系统不仅能给出接收/拒绝，还能告诉后续模块“这个判断有多稳”。

## 7. 困难样本分析

我们从 CN-Celeb2 全量结果中挖掘了困难样本：

```text
false accept: 82
false reject: 218
uncertain band: 644
```

困难样本集中包含：

- 高分负样本：非目标人但声纹分数很高；
- 低分正样本：目标人但声纹分数很低；
- 边界样本：落入不确定区间的样本。

这些样本主要用于后续做压力测试和模型迭代，不代表自然验证集分布。困难集上的 EER 很高是正常现象，因为它故意选取了最难区分的一批样本。

困难样本的意义在于：

- 找到模型最容易误接收的非目标人；
- 找到模型最容易误拒的目标人；
- 为后续融合、重校准、微调提供针对性样本；
- 在没有比赛开发集前，构造一个更接近“危险边界”的内部测试集。

## 8. 噪声与混响验证

为了贴近比赛中的复杂交互场景，我们使用 MUSAN 和 RIRS_NOISES 对 query 音频进行增强，保持 enrollment 为干净原音。

验证场景包括：

- MUSAN noise 5dB；
- MUSAN noise 0dB；
- MUSAN noise -5dB；
- RIRS 混响；
- RIRS + MUSAN noise 0dB。

实验结果：

| 场景 | EER | 推荐阈值 | 比赛加权分数 | 正样本均值 | 负样本均值 |
|---|---:|---:|---:|---:|---:|
| clean 1000 | 3.30% | 0.3468 | 77.39 | 0.6428 | 0.1227 |
| noise 5dB | 3.90% | 0.3130 | 76.97 | 0.6061 | 0.1257 |
| noise 0dB | 4.30% | 0.2989 | 76.65 | 0.5828 | 0.1271 |
| noise -5dB | 6.50% | 0.2957 | 75.64 | 0.5410 | 0.1215 |
| RIR | 4.70% | 0.3296 | 76.91 | 0.5908 | 0.1277 |
| RIR + noise 0dB | 7.10% | 0.3153 | 74.78 | 0.5186 | 0.1309 |

核心发现：

- 噪声和混响主要会压低目标说话人的分数；
- 负样本均值变化相对较小；
- 因此低 SNR 下不应简单提高声纹阈值，否则会误拒更多目标用户；
- 更合理的策略是质量感知动态阈值：低 SNR 时适当降低声纹阈值，同时结合 ASR、命令先验和不确定区间进行保守决策。

当前配置中已经加入：

```json
"dynamic_threshold": true,
"low_snr_threshold_boost": -0.035
```

推荐解释：

```text
clean / 高 SNR       使用 0.3371 左右
普通噪声 / 0-5dB     使用约 0.300 ~ 0.315
低 SNR / -5dB        使用约 0.295
RIR + noise          使用约 0.315
```

## 9. 当前阶段结论

在没有比赛官方开发集的情况下，当前版本可以认为是“公开数据可验证条件下的阶段性最优基线”。

当前系统已经具备：

- 中文声纹主模型 ERes2NetV2；
- 多片段 top-k 聚合打分；
- CN-Celeb2 全量验证；
- 主阈值、拒绝阈值、高置信接受阈值校准；
- 困难样本挖掘；
- MUSAN 噪声验证；
- RIRS 混响验证；
- 低 SNR 动态阈值策略。

但当前结果不能直接等价于最终比赛成绩，因为比赛开发集可能在以下方面与 CN-Celeb2 不同：

- 语音更短，更像真实指令；
- 远场录音比例更高；
- 家电噪声、环境噪声类型不同；
- 可能存在目标人和非目标人同时说话；
- 说话内容可能固定在指令域，而不是开放域说话人数据。

因此，当前版本适合作为强基线和汇报版本。拿到比赛开发集后，需要重新做阈值校准和场景化验证。

## 10. 后续计划

优先级从高到低如下：

1. 接入比赛开发集后重新校准阈值
   - 不直接沿用 CN-Celeb2 阈值；
   - 分析比赛开发集中的正负样本分数分布；
   - 重新确定 `threshold`、`speaker_reject_low`、`speaker_accept_high`。

2. 构造更接近比赛的人声干扰测试集
   - 目标说话人 + 非目标说话人混合；
   - 非目标人单独发出指令；
   - 多人重叠说话；
   - 近场 enrollment + 远场 query。

3. 做分场景动态阈值
   - 根据 SNR、混响、VAD 稳定性、语音时长选择阈值；
   - 对短语音和低质量语音使用更保守的决策策略。

4. 尝试更高级的融合策略
   - CAM++ + ERes2NetV2 分数归一化融合；
   - 不再使用简单平均；
   - 用开发集学习融合权重。

5. 结合最终指令识别链路做联合决策
   - 声纹分数；
   - ASR 置信度；
   - 命令合法性；
   - 音频质量；
   - 非目标人干扰概率。

## 11. 复现实验示例

构造 CN-Celeb2 trials：

```powershell
D:\Python39\python.exe scripts\make_speaker_trials.py `
  --root CN-Celeb2_flac\data `
  --output output\cnceleb2_trials_full_2to20s.csv `
  --min-duration-sec 2 `
  --max-duration-sec 20
```

运行 ERes2NetV2 全量评估：

```powershell
D:\Python39\python.exe speaker_eval.py `
  --trials output\cnceleb2_trials_full_2to20s.csv `
  --audio-root CN-Celeb2_flac\data `
  --output output\speaker_eval_eres2netv2_full_2to20s.json `
  --score-dump output\speaker_scores_eres2netv2_full_2to20s.json
```

校准阈值：

```powershell
D:\Python39\python.exe scripts\calibrate_speaker_thresholds.py `
  --score-dump output\speaker_scores_eres2netv2_full_2to20s.json `
  --output output\speaker_threshold_calibration_eres2netv2_cnceleb2.json
```

构造噪声增强验证集：

```powershell
D:\Python39\python.exe scripts\build_augmented_speaker_trials.py `
  --trials output\cnceleb2_trials_full_2to20s.csv `
  --audio-root CN-Celeb2_flac\data `
  --musan-root musan `
  --rirs-root RIRS_NOISES `
  --output-root output\augmented_speaker_audio `
  --output-trials output\aug_trials_noise0_1000.csv `
  --metadata output\aug_trials_noise0_1000_meta.json `
  --limit 1000 `
  --mode noise `
  --snr-db 0 `
  --noise-kind noise `
  --augment test
```

## 12. 一句话总结

我们目前已经把原始“能识别说话人”的功能，升级成了一个可评估、可校准、可抗噪声验证、可解释置信区间的声纹门控模块。当前主方案是 ERes2NetV2 + CN-Celeb2 阈值校准 + MUSAN/RIRS 鲁棒性验证；后续真正提升的关键，是接入比赛开发集后做场景化重校准和人声干扰测试。
