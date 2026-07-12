# 汇总结果说明

本目录只保留不含音频、逐条标签和声纹 embedding 的聚合实验结果。

- `clean/noise*/rir*_fusion_1000_eval.json`：CN-Celeb2 trials 在 MUSAN/RIRS_NOISES 条件下的双模型鲁棒性结果。
- `external_target_speaker_*`：AISHELL-WakeUp 与外部非目标说话人试验结果。
- `local_data_speaker_*`：本地多说话人公开/外部数据验证结果。
- `datasetA_final_robust_cascade_no_prior_full_stats.json`：正式外部校准方案在原始 DatasetA 上的只读汇总。

DatasetA 汇总仅用于迁移检查。DatasetA 没有参与正式模型训练、融合权重训练或最终阈值选择。
