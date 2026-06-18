# ERes2NetV2 噪声与远场鲁棒性评估

## 数据与设置

- 主声纹模型：ERes2NetV2
- 原始 trials：`output/cnceleb2_trials_full_2to20s.csv`
- 每组增强 trials：1000 条
- Enrollment：保持干净原音频
- Test/query：加入噪声或混响
- 噪声：MUSAN noise
- 混响：RIRS_NOISES

## 产物

增强音频目录：

```text
output/augmented_speaker_audio
```

增强 trials：

```text
output/aug_trials_noise5_1000.csv
output/aug_trials_noise0_1000.csv
output/aug_trials_noise-5_1000.csv
output/aug_trials_rir_1000.csv
output/aug_trials_rir_noise0_1000.csv
```

## 评估结果

| 场景 | EER | 推荐阈值 | 比赛权重分数 | 正样本均值 | 负样本均值 |
|---|---:|---:|---:|---:|---:|
| clean 1000 | 3.30% | 0.3468 | 77.39 | 0.6428 | 0.1227 |
| noise 5dB | 3.90% | 0.3130 | 76.97 | 0.6061 | 0.1257 |
| noise 0dB | 4.30% | 0.2989 | 76.65 | 0.5828 | 0.1271 |
| noise -5dB | 6.50% | 0.2957 | 75.64 | 0.5410 | 0.1215 |
| RIR | 4.70% | 0.3296 | 76.91 | 0.5908 | 0.1277 |
| RIR + noise 0dB | 7.10% | 0.3153 | 74.78 | 0.5186 | 0.1309 |

## 结论

1. ERes2NetV2 在常规噪声下表现稳定，5dB 和 0dB 仅小幅退化。
2. -5dB 和 RIR+noise 是明显压力场景，EER 分别升至 6.50% 和 7.10%。
3. 噪声主要压低正样本分数，负样本均值变化较小。因此固定使用 clean 阈值 `0.3371` 会增加目标人误拒风险。
4. 增强场景推荐阈值集中在 `0.295 ~ 0.315`，低于 clean 推荐阈值。

## 下一步策略

推荐加入质量感知动态阈值：

```text
clean / 高 SNR      使用 0.3371
普通噪声 / 0-5dB    使用约 0.300 ~ 0.315
低 SNR / -5dB       使用约 0.295
RIR + noise         使用约 0.315
```

工程上不建议简单“低 SNR 提高阈值”。从实验看，噪声主要降低目标人得分，若提高阈值会误拒更多目标人。更合理的是：

- 声纹阈值略降低，减少目标人误拒；
- 同时依靠二段阈值和 ASR/命令先验防止非目标误接收；
- 对 `accepted_uncertain_band` 样本保守处理。

后续可将 SNR 估计接入 `speaker_gate`，按 query quality 动态选择阈值。
