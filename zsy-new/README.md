# 抗干扰目标说话人识别：ERes2NetV2 + CAM++ 级联融合

本目录整理了我们在美的 XH-202615 复杂交互场景语音指令识别任务中完成的目标说话人识别工作。正式方案不使用 DatasetA 标签训练模型或选择最终阈值；DatasetA 只用于只读迁移检查。

## 1. 当前正式方案

- 主声纹模型：ERes2NetV2
- 辅助声纹模型：CAM++
- 分数融合：`0.54 * ERes2NetV2 + 0.46 * CAM++`
- 融合阈值：`0.274`
- 级联区间：ERes2NetV2 分数低于 `0.22` 直接拒绝，高于等于 `0.38` 直接接受，中间样本再调用 CAM++
- ASR：FunASR Paraformer-large
- 最终配置：`configs/final_robust_speaker_fusion.json`

级联策略在 DatasetA 只读检查中避免了约 68% 的 CAM++ 调用。正式阈值和融合权重来自外部说话人数据及噪声、混响试验，不来自 DatasetA 标签拟合。

## 2. 主要实验结果

### 外部多说话人验证

| 方案 | EER |
| --- | ---: |
| ERes2NetV2 | 5.15% |
| CAM++ | 5.95% |
| ERes2NetV2 + CAM++ | 3.95% |

### 噪声与混响压力测试

| 场景 | EER |
| --- | ---: |
| clean | 3.70% |
| MUSAN noise 5 dB | 3.90% |
| MUSAN noise 0 dB | 4.10% |
| MUSAN noise -5 dB | 5.50% |
| RIRS_NOISES RIR | 5.30% |
| RIR + noise 0 dB | 6.70% |

### DatasetA 只读迁移检查

| 指标 | 结果 |
| --- | ---: |
| 样本数 | 1838 |
| CER | 53.06% |
| RR | 90.93% |
| 正样本接受率 | 72.43% |
| CAM++ 调用率 | 31.94% |
| 平均耗时 | 0.397 秒/条 |

这不是隐藏测试集成绩，也不是在 DatasetA 上训练后的结果。隐藏测试集 B 不可获得，最终泛化性能只能由主办方评测。

## 3. 目录结构

```text
.
|-- configs/                    正式配置与外部校准配置
|-- docs/                       任务分析、训练方案和实验记录
|-- results/                    不含音频和标签的汇总结果
|-- scripts/                    数据清单、trial 构造、增强、评测和校准脚本
|-- tests/                      不依赖模型权重的指标测试
|-- infer.py                    端到端推理入口
|-- speaker_model.py            声纹模型后端
|-- speaker_gate.py             多裁剪、融合与级联门控
|-- metrics_cer.py              官方口径 CER/RR 计算
`-- requirements.txt            Python 依赖
```

## 4. 环境准备

推荐使用 Python 3.9，并根据本机 CUDA 版本先安装 PyTorch，然后执行：

```powershell
python -m pip install -r requirements.txt
git clone https://github.com/modelscope/3D-Speaker.git external_3D-Speaker
```

3D-Speaker 如提示缺少依赖，请按其官方仓库的安装说明补充安装。当前 GitHub 配置将 `asr.model_dir` 留空，因此 Paraformer 会通过 FunASR/ModelScope 获取；离线部署时可把该字段改成本地模型目录。

首次运行时，ModelScope 可以下载以下公开预训练权重：

- `iic/speech_eres2netv2_sv_zh-cn_16k-common`
- `iic/speech_campplus_sv_zh-cn_16k-common`
- `iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch`

也可以提前下载到 `pretrained/`。模型权重、第三方仓库和数据集不包含在本上传包中。

## 5. 推理

输入 JSONL 每行至少包含 `id`、`唤醒音频` 和 `识别音频`。音频路径相对于 `--audio-root`。

```powershell
python infer.py `
  --meta path\to\meta.jsonl `
  --audio-root path\to\audio `
  --config configs\final_robust_speaker_fusion.json `
  --output output\result.json `
  --stats output\result_stats.json
```

仅生成通用声纹分数：

```powershell
python scripts\score_speaker_pairs.py `
  --meta path\to\meta.jsonl `
  --audio-root path\to\audio `
  --config configs\final_robust_speaker_fusion.json `
  --output output\speaker_eval.json `
  --score-dump output\speaker_scores.json
```

## 6. 外部数据校准流程

1. 按 `docs/speaker_validation_datasets.md` 准备公开数据。
2. 使用 `prepare_external_training_metadata.py` 建立外部数据清单。
3. 使用 `build_external_target_trials.py` 或 `make_speaker_trials.py` 构造同人/异人 trials。
4. 使用 `build_augmented_speaker_trials.py` 加入 MUSAN 与 RIRS_NOISES 干扰。
5. 使用 `evaluate_speaker_trials_cached.py` 分别提取两个模型的分数。
6. 使用 `fuse_speaker_score_dumps.py` 搜索外部数据上的融合权重和工作点。

批量鲁棒性评测脚本：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_robustness_fusion_batch.ps1
```

## 7. 数据与合规边界

本仓库不包含：

- DatasetA 音频、标签、逐条预测或由其训练得到的分类器；
- CN-Celeb2、AISHELL-WakeUp、MUSAN、RIRS_NOISES 原始数据；
- ERes2NetV2、CAM++、Paraformer 模型权重；
- 本机缓存、绝对路径和临时增强音频。

历史上做过的 DatasetA 融合和阈值实验仅用于认识开发集特征，相关模型不属于正式隐藏集方案。最新技术结论优先阅读：

- `docs/formal_training_plan_20260710.md`
- `docs/external_speaker_calibration_20260710.md`
- `docs/robust_speaker_fusion_results_20260711.md`
- `docs/cascade_and_fusion_update_20260711.md`

## 8. 复现检查

不加载模型即可先检查 CER/RR 实现：

```powershell
python tests\test_metrics.py
python -m compileall -q .
```
