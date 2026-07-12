# 级联声纹门控与 DatasetA 融合方案更新（2026-07-11）

本文记录本轮“下一步优化”实际落地的内容，方便后续继续实施时快速接上。

## 1. 当前结论

我们现在保留两条路线：

1. **隐藏测试稳健路线**
   - 配置：`configs/final_robust_speaker_fusion.json`
   - 模型：ERes2NetV2 主模型 + CAM++ 辅助模型
   - 策略：ERes2NetV2 先判，只有不确定样本才调用 CAM++ 融合
   - 目标：减少过拟合 DatasetA，优先保证未知场景鲁棒性

2. **DatasetA 开发集最强路线**
   - 一键脚本：`scripts/run_datasetA_fusion_stack.ps1`
   - 方法：ERes2NetV2/CAM++ 声纹分数 + Paraformer/SenseVoice ASR + 文本语法特征 + LogisticRegression 融合判别
   - 目标：充分利用 DatasetA 标签和缓存结果，冲开发集分数
   - 注意：该路线对 DatasetA 有明显调参成分，不能等价视为隐藏测试集真实效果

## 2. 本轮新增/修改文件

新增：

- `scripts/score_speaker_pairs.py`
  - 通用声纹打分脚本，不再依赖 DatasetA 的 `pos.jsonl` / `neg.jsonl` 固定结构
  - 输入任意比赛 meta 文件，输出离线融合脚本可直接使用的 score dump

- `scripts/run_datasetA_fusion_stack.ps1`
  - DatasetA 离线融合一键流程
  - 已有缓存会自动跳过，缺失或加 `-Recompute` 才重跑

- `docs/cascade_and_fusion_update_20260711.md`
  - 本文档

修改：

- `config.py`
  - 新增 `cascade_enabled`
  - 新增 `cascade_primary_backend`
  - 新增 `cascade_low`
  - 新增 `cascade_high`

- `speaker_gate.py`
  - 接入级联门控
  - enrollment 阶段先只计算主模型 embedding
  - 只有命中不确定区间时才懒加载 CAM++ embedding 和 query 分数

- `configs/final_robust_speaker_fusion.json`
  - 开启级联
  - 主模型：`eres2netv2`
  - `cascade_low = 0.22`
  - `cascade_high = 0.38`
  - 关闭硬性低命令先验拒绝：`min_command_prior_for_accept = 0.0`

## 3. 级联门控策略

离线搜索后采用：

```text
primary_score = ERes2NetV2(wake, cmd)

primary_score < 0.22:
    直接拒绝，不调用 CAM++

primary_score >= 0.38:
    直接接受，不调用 CAM++

0.22 <= primary_score < 0.38:
    调用 CAM++
    fused_score = 0.54 * ERes2NetV2 + 0.46 * CAM++
    threshold = 0.274
```

这不是为了追求 DatasetA 单点最优，而是为了在鲁棒性和推理成本之间做折中。

## 4. 实测结果

### 4.1 DatasetA 小样本冒烟

8 条样本：

- 6 条只调用 ERes2NetV2
- 2 条进入 ERes2NetV2 + CAM++ 融合
- 说明级联路径真实生效

### 4.2 DatasetA 全量声纹门控

```text
n_total              : 1838
n_positive           : 1364
n_rejection          : 474
positive accept rate : 72.43%
negative reject rate : 90.93%
CAM++ call rate      : 31.94%
primary-only rate    : 68.06%
avg sec/sample       : 0.355
```

级联后的声纹门控性能与离线估计基本一致，同时避免了约 68% 样本的 CAM++ 调用。

### 4.3 命令先验调整

端到端测试发现：DatasetA 正样本不全是家电控制命令，也包含开放式语音，例如歌曲、节日、生活场景等。因此不能用“是否像家电命令”作为硬性拒绝条件。

对比：

| 配置 | CER | RR | 说明 |
|---|---:|---:|---|
| 硬低命令先验拒绝 | 67.46 | 95.99 | 误拒大量正样本 |
| 关闭硬低命令先验拒绝 | 53.06 | 90.93 | 更符合 DatasetA 标注形态 |

因此当前最终稳健配置保留命令语义特征，但不再一票否决。

## 5. DatasetA 离线融合结果

当前最强的 DatasetA 方案不是纯规则端到端，而是离线融合模型。

交叉验证结果：

```text
CV aggregate:
CER          : 50.90
RR           : 93.67
proxy 40/40  : 57.11
pos accept   : 78.52%
neg false acc: 30
```

全量 DatasetA 训练后的提交结果：

| 阈值来源 | CER | RR | proxy 40/40 | 说明 |
|---|---:|---:|---:|---|
| CV median | 38.24 | 96.20 | 63.18 | 相对保守 |
| full tuned | 37.60 | 95.99 | 63.36 | 开发集最优，但更乐观 |

输出文件：

- `output/datasetA_fusion_submission_grammar_fixed_cv_median.json`
- `output/datasetA_fusion_submission_grammar_fixed_full_tuned.json`
- `output/datasetA_fusion_model_grammar_fixed.pkl`
- `output/datasetA_fusion_model_grammar_fixed_summary.json`

## 6. 推荐使用方式

### 6.1 跑 DatasetA 最强开发集方案

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_datasetA_fusion_stack.ps1
```

如果需要全部重算：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_datasetA_fusion_stack.ps1 -Recompute
```

### 6.2 跑在线稳健端到端方案

```powershell
python infer.py `
  --meta output\datasetA_all.jsonl `
  --audio-root datasetA `
  --config configs\final_robust_speaker_fusion.json `
  --output output\datasetA_final_robust_cascade_no_prior_full.json `
  --stats output\datasetA_final_robust_cascade_no_prior_full_stats.json
```

### 6.3 给隐藏集生成通用声纹分数

```powershell
python scripts\score_speaker_pairs.py `
  --meta path\to\hidden_meta.jsonl `
  --audio-root path\to\hidden_audio_root `
  --config configs\datasetA_speaker_tuned.json `
  --output output\hidden_eres2netv2_eval.json `
  --score-dump output\hidden_eres2netv2_scores.json
```

CAM++ 同理把配置换成：

```text
configs/datasetA_campplus.json
```

## 7. 下一步建议

1. 隐藏测试集到来后，先跑 `configs/final_robust_speaker_fusion.json` 生成稳健基线。
2. 如果允许使用 DatasetA 训练出的二级融合模型，再用通用 score dump + ASR cache 复用离线融合流程生成高分候选。
3. 后续真正提升上限的方向不是继续手调单个阈值，而是：
   - 更强的目标说话人活动检测（target-speaker VAD）
   - 只对不确定样本做目标说话人增强/TSE
   - 用比赛风格数据做 ERes2NetV2/CAM++ 分数校准
   - 加入更多真实多说话人、远场、噪声、混响样本做压力验证

