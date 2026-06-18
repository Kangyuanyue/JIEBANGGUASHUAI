# ERes2NetV2 Hard Negative 校准记录

## 输入

- 模型：ERes2NetV2
- 分数文件：`output/speaker_scores_eres2netv2_full_2to20s.json`
- 主阈值：`0.3370734058066252`
- 低阈值：`0.23909343270879338`
- 高阈值：`0.37844391932749266`

## 产物

```text
output/speaker_hard_cases_eres2netv2_cnceleb2.json
output/speaker_hard_trials_eres2netv2_cnceleb2.csv
output/speaker_eval_hard_eres2netv2_cnceleb2.json
```

## 全量集错误统计

在 CN-Celeb2 全量 7902 条 trials 上：

```text
false accept: 82
false reject: 218
uncertain band: 644
```

误接收主要集中在：

- vlog
- live_broadcast
- entertainment
- speech
- singing
- interview
- recitation

误拒主要集中在：

- entertainment
- interview
- singing
- drama
- live_broadcast

这说明最危险的非目标样本往往来自自然说话、直播、访谈和相同 genre 的音频；最容易误拒的目标样本往往跨娱乐、唱歌、访谈等强风格变化场景。

## 困难集说明

困难集不是自然分布验证集，而是压力测试集：

- 取 top 300 高分负样本；
- 取 bottom 300 低分正样本；
- 共 600 条。

因此困难集上 EER 很高是正常现象。它的用途是后续比较模型和校准策略有没有改善最坏情况，而不是作为最终阈值来源。

当前困难集结果：

```text
EER: 53.67%
positive_mean: 0.2772
negative_mean: 0.3200
```

这个结果说明 hard set 故意选出了“负样本比正样本更像目标人”的极端边界样本，后续要靠更强策略改善。

## 已落地二段阈值策略

当前声纹门控会输出：

```text
score < 0.2391       -> clear_reject
0.2391 ~ 0.3371     -> uncertain_below_threshold
0.3371 ~ 0.3784     -> accepted_uncertain_band
score >= 0.3784     -> clear_accept
```

推荐解释：

- `clear_reject`：明显非目标人，直接拒识；
- `uncertain_below_threshold`：低置信不确定区，默认拒识，但可在比赛开发集上结合 pVAD/ASR 再判断；
- `accepted_uncertain_band`：达到主阈值但未达到高置信，需要结合 ASR、命令先验、音频质量；
- `clear_accept`：高置信目标人。

## 下一步

1. 在比赛开发集到手后，只用开发集重新标定这三个阈值。
2. 对 hard negative 集尝试 CAM++ + ERes2NetV2 的分数归一化融合，而不是简单均值。
3. 加入 MUSAN / RIRS_NOISES 构造噪声和远场 hard set。
