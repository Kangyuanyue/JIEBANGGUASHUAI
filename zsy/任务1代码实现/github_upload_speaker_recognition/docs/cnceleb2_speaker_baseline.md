# CN-Celeb2 声纹识别基线实验

## 数据

- 数据集：CN-Celeb2 v2
- 本地路径：`G:\Midea_Task\CN-Celeb2_flac\data`
- 说话人数：1995
- 原始音频数：524786
- 试验对：7902
- 正样本：3918
- 负样本：3984
- 音频筛选：2s <= duration <= 20s

trials 文件：

```text
output/cnceleb2_trials_full_2to20s.csv
```

## 模型

- Speaker encoder：SpeechBrain ECAPA-TDNN VoxCeleb
- Enrollment：多裁剪平均 embedding
- Query：3s segment，多裁剪打分

## 实验结果

| 聚合策略 | EER | 推荐阈值 | 比赛权重分数 | 正样本均值 | 负样本均值 |
|---|---:|---:|---:|---:|---:|
| max | 13.78% | 0.3021 | 69.47 | 0.4989 | 0.1500 |
| topk_mean, k=3 | 13.38% | 0.2668 | 69.70 | 0.4755 | 0.1302 |

## 3D-Speaker 模型接入结果

已接入 3D-Speaker 官方预训练模型：

- CAM++：`iic/speech_campplus_sv_zh-cn_16k-common`
- ERes2NetV2：`iic/speech_eres2netv2_sv_zh-cn_16k-common`

模型权重已下载到：

```text
pretrained/speech_campplus_sv_zh-cn_16k-common
pretrained/speech_eres2netv2_sv_zh-cn_16k-common
```

### 1000 条快速对比

| 模型 | EER | 推荐阈值 | 比赛权重分数 | 正样本均值 | 负样本均值 |
|---|---:|---:|---:|---:|---:|
| ECAPA | 13.20% | 0.2837 | 70.48 | 0.4696 | 0.1301 |
| CAM++ | 4.90% | 0.3014 | 76.76 | 0.5967 | 0.0614 |
| ERes2NetV2 | 3.30% | 0.3468 | 77.39 | 0.6428 | 0.1227 |
| CAM++ + ERes2NetV2 | 3.70% | 0.3170 | 77.63 | 0.6198 | 0.0920 |

### ERes2NetV2 全量结果

| 模型 | Trials | EER | 推荐阈值 | 比赛权重分数 | 正样本均值 | 负样本均值 |
|---|---:|---:|---:|---:|---:|---:|
| ECAPA topk_mean k=3 | 7902 | 13.38% | 0.2668 | 69.70 | 0.4755 | 0.1302 |
| ERes2NetV2 topk_mean k=3 | 7902 | 4.09% | 0.3371 | 76.95 | 0.6468 | 0.1251 |

当前默认声纹 backend 已切换为：

```json
"embedding_backends": ["eres2netv2"]
```

## ERes2NetV2 阈值校准

校准文件：

```text
output/speaker_threshold_calibration_eres2netv2_cnceleb2.json
```

当前默认配置采用 CN-Celeb2 全量 trials 上的推荐 operating point：

```json
"gate": {
  "threshold": 0.3370734058066252
},
"decision": {
  "speaker_reject_low": 0.23909343270879338,
  "speaker_accept_high": 0.37844391932749266
}
```

关键 operating points：

| 工作点 | 阈值 | 接收召回 | 拒识率 | FAR | FRR |
|---|---:|---:|---:|---:|---:|
| EER | 0.3017 | 95.92% | 95.91% | 4.09% | 4.08% |
| 比赛均衡推荐 | 0.3371 | 94.44% | 97.94% | 2.06% | 5.56% |
| RR 99 | 0.3784 | 92.70% | 99.02% | 0.98% | 7.30% |

比赛拒识率权重很高，因此当前主阈值选择 `0.3371`，比 EER 阈值更偏向降低误接收。低阈值 `0.2391` 用于明显非目标人的快速拒识，高阈值 `0.3784` 用于高置信目标人判断。

## 当前结论

`topk_mean, k=3` 是当前更稳的分段聚合策略。接入 ERes2NetV2 后，声纹识别能力大幅优于 ECAPA baseline，说明优先级 1 的“更强 speaker encoder”是正确方向。

当前默认配置已更新为：

```json
"aggregate": "topk_mean",
"top_k": 3
```

## 下一步优先级

1. 接入 CAM++ / ERes2NetV2 作为第二路 embedding，做 score fusion。
2. 用 CN-Celeb2 构造 hard negative：同 genre、同文本风格、相近时长、相似分数负样本。
3. 加入 MUSAN + RIRS_NOISES 生成低 SNR 和远场混响 trials，验证比赛要求的 -5 dB 到 5 dB 鲁棒性。
4. 在比赛开发集到手后重新校准阈值；CN-Celeb2 阈值不可直接当最终比赛阈值。

## 错误分析

在推荐阈值 `0.2668` 下：

```text
FP: 388
FN: 627
```

误接收主要集中在：

- vlog
- live_broadcast
- speech
- entertainment
- recitation

误拒主要集中在：

- entertainment
- singing
- interview
- live_broadcast
- recitation

这说明当前瓶颈不是数据量不够，而是场景和语体差异导致的 speaker embedding 分布变化。单一路 ECAPA 在唱歌、娱乐节目、采访、直播等强域偏移场景上仍然不够稳。

## 关于是否继续下载数据

短期优先级不是继续下载更大的普通声纹数据，而是：

1. 先接入更强的预训练 speaker encoder，例如 CAM++ / ERes2NetV2；
2. 再用现有 CN-Celeb2 做 hard negative 校准；
3. 然后下载 MUSAN / RIRS_NOISES 做噪声和混响增强验证；
4. 最后考虑 AISHELL-4 / AliMeeting，用于多人重叠和远场压力测试。

原因是 CN-Celeb2 已经有 1995 个说话人和 52 万条音频，足够用于第一阶段验证。继续堆普通 clean speaker 数据，收益会小于模型融合和噪声/远场针对性增强。
