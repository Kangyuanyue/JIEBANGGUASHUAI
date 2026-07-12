# 双模型声纹融合鲁棒性实验记录（2026-07-11）

本轮目标：在不继续依赖 DatasetA 标签训练的前提下，验证 ERes2NetV2 + CAM++ 双模型声纹融合在公开/外部数据、噪声、混响场景下是否稳定，并确定下一版稳健推理配置。

## 1. 本轮新增文件

新增脚本：

- `scripts/build_fused_speaker_scores.py`
  - 将 ERes2NetV2 与 CAM++ 两份声纹分数融合成 DatasetA 风格 score dump；
  - 用于复用现有离线阈值扫描脚本。
- `scripts/run_robustness_fusion_batch.ps1`
  - 顺序跑 clean/noise/RIR 增强场景的双模型声纹评估。

新增配置：

- `configs/final_robust_speaker_fusion.json`
  - ERes2NetV2 权重：0.54；
  - CAM++ 权重：0.46；
  - gate threshold：0.274；
  - reject_low：0.254；
  - accept_high：0.320；
  - dynamic_threshold：false。

主要输出：

- `output/datasetA_speaker_gate_scores_external_fusion_054_046.json`
- `output/datasetA_external_fusion_054_046_exhaustive_sweep.json`
- `output/datasetA_external_fusion_054_046_key_thresholds.json`
- `output/robustness_fusion_20260711/*_fusion_1000_eval.json`
- `output/robustness_fusion_20260711/*_fusion_1000_scores.json`

## 2. DatasetA 只读迁移验证

使用外部数据得到的融合权重：

```text
fused_score = 0.54 * ERes2NetV2 + 0.46 * CAM++
```

然后只在 DatasetA 上做离线阈值观察，不把 DatasetA 作为最终训练依据。

| 阈值 | CER | RR | 正样本接受率 | 负样本误接 | proxy 40/40 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.254 | 44.52 | 88.61 | 77.42 | 54 | 57.63 |
| 0.265 | 46.00 | 89.66 | 75.22 | 49 | 57.46 |
| 0.274 | 47.57 | 90.72 | 72.95 | 44 | 57.26 |
| 0.285 | 48.99 | 91.77 | 70.75 | 39 | 57.11 |
| 0.292 | 50.35 | 92.83 | 68.77 | 34 | 56.99 |
| 0.320 | 54.59 | 95.78 | 62.32 | 20 | 56.48 |

DatasetA 上低阈值更好，核心原因是正样本误拒会显著拉高 CER。但 DatasetA 旧融合模型的 full tuned 结果属于开发集调参结果，有过拟合风险：

| 方案 | CER | RR | proxy 40/40 | 说明 |
| --- | ---: | ---: | ---: | --- |
| DatasetA full tuned fusion | 37.13 | 96.20 | 63.63 | 开发集全量调参，隐藏集风险高 |
| DatasetA CV median fusion | 37.07 | 95.78 | 63.48 | 仍依赖 DatasetA 候选命令与标签 |
| 外部权重融合，DatasetA 最优阈值 | 44.48 | 88.61 | 57.65 | 不训练 DatasetA，仅作迁移观察 |

结论：DatasetA 可以说明“低误拒很重要”，但不应单独决定隐藏测试最终阈值。

## 3. 噪声与混响鲁棒性

数据：CN-Celeb2 clean trials 1000 条，以及 MUSAN/RIRS 增强后的 1000 条 trials。

配置：`configs/external_speaker_fusion.json` 的双模型融合权重。

| 场景 | EER | 推荐阈值 | 门控分 | 正样本均值 | 负样本均值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 3.70% | 0.3204 | 77.55 | 0.6216 | 0.0945 |
| noise 5dB | 3.90% | 0.2851 | 77.14 | 0.5898 | 0.0982 |
| noise 0dB | 4.10% | 0.2836 | 76.98 | 0.5676 | 0.1009 |
| noise -5dB | 5.50% | 0.2542 | 75.95 | 0.5263 | 0.0962 |
| RIR | 5.30% | 0.2915 | 76.99 | 0.5679 | 0.0991 |
| RIR + noise 0dB | 6.70% | 0.2736 | 74.93 | 0.4982 | 0.1044 |

观察：

- clean 场景可用更高阈值，约 0.320；
- 5dB、0dB 噪声与 RIR 场景都集中在 0.284 到 0.292；
- -5dB 强噪声会把最佳阈值拉低到约 0.254；
- RIR + noise 0dB 是当前最难场景，EER 到 6.70%。

## 4. 阈值策略对比

在六个鲁棒场景上比较固定阈值与当前简单动态阈值：

| 策略 | 平均门控分 | 最低场景门控分 | 判断 |
| --- | ---: | ---: | --- |
| fixed 0.254 | 75.98 | 74.26 | 误接偏高 |
| fixed 0.265 | 76.25 | 74.68 | 较稳 |
| fixed 0.274 | 76.41 | 74.85 | 平均最好 |
| fixed 0.285 | 76.41 | 74.70 | 与 0.274 接近，但强干扰误拒更多 |
| fixed 0.292 | 76.27 | 74.87 | 最坏场景略高，但平均下降 |
| current dynamic | 76.39 | 74.78 | 没有优于简单固定阈值 |

因此本轮最终选择：

```text
threshold = 0.274
reject_low = 0.254
accept_high = 0.320
dynamic_threshold = false
```

这个选择不是 DatasetA 分数最高的阈值，而是面向隐藏测试的稳健折中。

## 5. 当前最终判断

下一版正式推理优先使用：

```text
configs/final_robust_speaker_fusion.json
```

理由：

- 双模型融合在 local_data 多说话人验证中将 EER 从单模型 5.15% 降到 3.95%；
- 在 noise 5dB、noise 0dB、RIR 中推荐阈值均接近 0.274 到 0.292；
- 0.274 在六个鲁棒场景上平均门控分最高；
- 相比 0.285，0.274 对强噪声和 RIR+noise 的正样本误拒更友好；
- 相比 0.254，0.274 对负样本误接更稳。

## 6. 下一步

继续推进的优先级：

1. 用 `configs/final_robust_speaker_fusion.json` 跑 DatasetA 小规模真实推理，确认实际 pipeline 决策逻辑没有被配置破坏；
2. 做级联推理优化：ERes2NetV2 先跑，只有 0.254 到 0.320 的不确定区间再跑 CAM++，降低耗时；
3. 将噪声/混响结果写入总方案文档；
4. 如需要冲开发集榜单，可单独保留 DatasetA tuned 方案，但不要作为隐藏测试最终方案。

## 7. 真实推理冒烟与命令先验修复

使用 `configs/final_robust_speaker_fusion.json` 跑 DatasetA 小样本真实推理时，发现两个工程问题：

1. ASR `model_dir` 为空时，FunASR 会尝试联网下载 Paraformer，在本机出现 SSL 失败；
2. `command_postprocess.py` 与 `command_grammar.py` 中的中文命令词表存在乱码，导致真实家电命令和非命令文本的 `command_prior_score` 都为 0。

已修复：

- `configs/final_robust_speaker_fusion.json` 已指向本地 Paraformer：
  - `pretrained/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`
- `command_postprocess.py` 已恢复正常 UTF-8 中文命令词表；
- `command_grammar.py` 已恢复正常 UTF-8 中文设备、动作、属性、场景词表；
- `decision.py` 增加低命令先验拒识：
  - ASR 已经输出文本；
  - `command_prior_score < 0.1`；
  - `speaker_similarity < 0.55`；
  - 则拒识，原因记为 `low_command_prior`。

修复效果：

| 冒烟样本 | 修复前 | 修复后 |
| --- | --- | --- |
| DatasetA neg 前 5 条 | RR 80%，其中 1 条高声纹非命令文本误接 | RR 100%，误接样本被拒 |
| DatasetA pos 前 3 条 | 第 2、3 条真实命令通过 | 第 2、3 条真实命令仍通过 |

这次修复对隐藏测试很重要：它不依赖 DatasetA 标签训练，而是修正了线上推理中真实存在的命令语义判断缺陷。

## 8. DatasetA 融合模型重训对照

修复命令词表后，重新跑 DatasetA 交叉验证融合：

| 方案 | CER | RR | proxy 40/40 | 说明 |
| --- | ---: | ---: | ---: | --- |
| 修复前 DatasetA CV fusion | 51.71 | 93.88 | 56.87 | 旧乱码命令特征 |
| 修复后 DatasetA CV fusion | 50.90 | 93.67 | 57.11 | 命令语义特征恢复 |
| 修复后 full-tuned | 37.60 | 95.99 | 63.36 | 开发集调参结果 |
| 修复后 CV-median | 38.24 | 96.20 | 63.18 | 较保守开发集结果 |

新产物：

- `output/datasetA_cv_fusion_eres2netv2_campplus_grammar_fixed.json`
- `output/datasetA_fusion_model_grammar_fixed.pkl`
- `output/datasetA_fusion_model_grammar_fixed_summary.json`
- `output/datasetA_fusion_submission_grammar_fixed_cv_median.json`
- `output/datasetA_fusion_submission_grammar_fixed_full_tuned.json`

注意：DatasetA 融合模型仍属于开发集方案，不建议作为隐藏测试最终方案直接使用。

