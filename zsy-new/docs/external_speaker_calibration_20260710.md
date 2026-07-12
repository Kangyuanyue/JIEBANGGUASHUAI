# 外部数据声纹校准记录（2026-07-10）

本文记录在不使用 DatasetA 标签训练的前提下，基于外部/公开数据对声纹门控进行的校准实验。

## 1. 本轮目标

比赛最终隐藏测试集不可见，DatasetA 只适合作为开发观察集，不能作为最终训练依据。

因此本轮目标是：

- 用 AISHELL-WakeUp sample 与本地 `data/16k_wav_file` 构造 target/non-target 验证；
- 用 `data/16k_wav_file` 的 speaker-like id 构造多说话人验证；
- 对 ERes2NetV2 与 CAM++ 做外部校准；
- 判断是否采用双模型声纹融合。

## 2. 已生成数据与脚本

数据清单：

- `output/external_training_metadata/external_target_speaker_trials.csv`
- `output/external_training_metadata/local_data_speaker_trials.csv`

新增脚本：

- `scripts/evaluate_speaker_trials_cached.py`
  - 带 embedding 缓存的声纹 trial 评估脚本；
  - 复现现有门控逻辑：预处理、分段、top-k 聚合、阈值搜索；
  - 避免同一音频在大量 trial 中反复提声纹。
- `scripts/fuse_speaker_score_dumps.py`
  - 对两个模型的 score dump 做分数级融合权重搜索。

新增候选配置：

- `configs/external_speaker_fusion.json`
  - ERes2NetV2 权重 0.54；
  - CAM++ 权重 0.46；
  - base threshold 0.285；
  - reject_low 0.265；
  - accept_high 0.334。

## 3. 单模型完整结果

| 数据集 | 模型 | trials | EER | 推荐阈值 | 正样本均值 | 负样本均值 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| external target/non-target | ERes2NetV2 | 2000 | 2.10% | 0.3388 | 0.5625 | 0.0969 |
| external target/non-target | CAM++ | 2000 | 0.60% | 0.2972 | 0.5565 | 0.0541 |
| local_data speaker trials | ERes2NetV2 | 3497 | 5.15% | 0.2858 | 0.5281 | 0.1052 |
| local_data speaker trials | CAM++ | 3497 | 5.95% | 0.2572 | 0.4577 | 0.0971 |

结论：

- CAM++ 在 AISHELL-WakeUp target/non-target 场景明显更强，负样本分数压得更低；
- ERes2NetV2 在 local_data 多说话人验证上更稳；
- 两者错误分布不完全一致，值得做分数级融合。

## 4. 双模型融合结果

使用 raw score 加权：

`fused_score = 0.54 * ERes2NetV2 + 0.46 * CAM++`

| 数据集 | 融合 EER | 推荐阈值 | 比赛式门控分 |
| --- | ---: | ---: | ---: |
| external target/non-target | 0.70% | 0.3341 | 79.56 |
| local_data speaker trials | 3.95% | 0.2655 | 76.96 |

在 local_data 上，融合将 EER 从 ERes2NetV2 的 5.15% 降到 3.95%，说明 CAM++ 对困难样本有互补价值。

## 5. 候选阈值对比

双模型融合权重固定为 0.54 / 0.46。

### external target/non-target

| 阈值 | FAR | FRR | 接受召回 | 拒识率 | 门控分 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.265 | 2.50% | 0.30% | 99.70% | 97.50% | 78.88 |
| 0.285 | 1.70% | 0.30% | 99.70% | 98.30% | 79.20 |
| 0.300 | 1.00% | 0.60% | 99.40% | 99.00% | 79.36 |
| 0.320 | 0.50% | 0.90% | 99.10% | 99.50% | 79.44 |
| 0.334 | 0.20% | 1.00% | 99.00% | 99.80% | 79.52 |

### local_data speaker trials

| 阈值 | FAR | FRR | 接受召回 | 拒识率 | 门控分 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.265 | 4.40% | 3.21% | 96.79% | 95.60% | 76.96 |
| 0.285 | 3.20% | 5.21% | 94.79% | 96.80% | 76.64 |
| 0.300 | 2.23% | 6.64% | 93.36% | 97.77% | 76.45 |
| 0.320 | 1.43% | 8.82% | 91.18% | 98.57% | 75.90 |
| 0.334 | 1.26% | 10.76% | 89.24% | 98.74% | 75.19 |

## 6. 当前判断

建议下一阶段采用：

- 主声纹模型：ERes2NetV2；
- 辅助声纹模型：CAM++；
- 融合权重：ERes2NetV2 0.54，CAM++ 0.46；
- 基础阈值：0.285；
- 低阈值：0.265；
- 高阈值：0.334。

理由：

- 0.334 在 target/non-target 上最好，但在 local_data 上误拒偏高；
- 0.265 在 local_data 上最好，但对拒识压力稍弱；
- 0.285 是当前更稳的折中点，后续可在噪声、混响、重叠说话 episode 上继续修正。

## 7. 下一步

优先级从高到低：

1. 用 `configs/external_speaker_fusion.json` 跑 DatasetA 只读验证，观察迁移效果，但不反向训练；
2. 构造 MUSAN/RIRS 噪声与混响版本的 speaker trials；
3. 对 clean/noise/reverb 三类数据分别统计阈值漂移；
4. 如果双模型全量推理耗时过高，再实现级联门控：ERes2NetV2 先判定，只有不确定样本再调用 CAM++；
5. 继续接入 OpenSLR 120 / HI-MIA-CW hard negative，增强唤醒词相近负样本。

