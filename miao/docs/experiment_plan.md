# 实验计划与消融设计

## 1. 实验目标

通过逐步增加模块，量化每个模块对 CER、RR、推理时间和内存占用的贡献，最终确定最优提交配置。

## 2. 实验版本

| 版本 | 模块 | 目的 |
|---|---|---|
| V0 | ASR only | 获得普通 ASR baseline，观察拒识问题 |
| V1 | VAD + ASR | 减少空音频误触发 |
| V2 | VAD + SV + ASR | 验证 speaker verification 对 RR 的提升 |
| V3 | VAD + SV + Command Postprocess | 验证指令后处理对 CER 的提升 |
| V4 | VAD + SV + pVAD + ASR | 验证 target-speaker pVAD 对重叠和拒识的提升 |
| V5 | VAD + SV + pVAD + Conditional TSE + ASR | 验证 TSE 对重叠 CER 的提升 |
| V6 | V5 + Fusion Decision | 验证多证据融合决策 |
| V7 | V6 + ONNX / FP16 / INT8 | 验证部署效率优化 |

## 3. 指标记录模板

| 版本 | 正样本 CER | 负样本 RR | -5 dB CER | 100% overlap CER | 相似音色 RR | 平均时延 | 总 duration | 峰值内存 | 备注 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| V0 |  |  |  |  |  |  |  |  |  |
| V1 |  |  |  |  |  |  |  |  |  |
| V2 |  |  |  |  |  |  |  |  |  |
| V3 |  |  |  |  |  |  |  |  |  |
| V4 |  |  |  |  |  |  |  |  |  |
| V5 |  |  |  |  |  |  |  |  |  |
| V6 |  |  |  |  |  |  |  |  |  |
| V7 |  |  |  |  |  |  |  |  |  |

## 4. 阈值搜索

建议搜索范围：

```yaml
speaker_accept_high: [0.65, 0.70, 0.75, 0.80]
speaker_reject_low: [0.35, 0.40, 0.45, 0.50]
min_target_ratio: [0.10, 0.15, 0.20, 0.25]
non_target_ratio_limit: [0.50, 0.60, 0.70]
fusion_accept_threshold: [0.40, 0.50, 0.60, 0.70]
```

每次搜索不仅记录平均分，还要记录错误类型：

- false reject：目标样本被拒识；
- false accept：非目标样本被接收；
- wrong speaker transcription：输出干扰人内容；
- noisy ASR error：低 SNR 下识别错；
- TSE artifact error：TSE 后识别变差。

## 5. 分桶评估

| 分桶 | 需要回答的问题 |
|---|---|
| SNR -5 / 0 / 5 dB | 噪声越大 CER 是否明显变坏 |
| overlap 0 / 25 / 50 / 75 / 100% | 重叠越高目标识别是否稳定 |
| target-only / non-target-only / mixed | 接收和拒识是否平衡 |
| short wake / normal wake | 短唤醒音频 embedding 是否不稳 |
| similar-speaker negative | 音色相似非目标是否误接收 |
| air-conditioner noise | 稳态家居噪声是否造成误触发 |
| TSE triggered / not triggered | 条件式 TSE 是否真的提升疑难样本 |

## 6. 最终提交标准

满足以下条件才进入最终提交：

- 本地正样本 CER 明显优于 ASR only；
- 本地 RR 明显优于 SV only；
- -5 dB 和 100% overlap 分桶没有灾难性下降；
- TSE 触发比例合理，不能拖慢整体 duration；
- JSON 可通过 schema 校验；
- 干净环境完整复现成功；
- 技术报告和测试报告能解释每个模块的贡献。
