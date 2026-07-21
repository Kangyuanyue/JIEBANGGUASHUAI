# Midea XH-202615 Target-Speaker ASR

本目录是“复杂交互场景的抗干扰语音指令识别”项目的可上传代码包。主线由 ERes2NetV2/CAM++ 声纹门控、Paraformer/SenseVoice 三路无标签共识和选择性 Pos/Neg TSE 组成。

## 当前结论

- 推荐外部校准版：ERes2NetV2+CAM++，DatasetA 开发评测 CER 46.11%、RR 87.55%；
- 高拒识开发版：CER 47.95%、RR 90.72%；
- CER 优先开发版：CER 42.79%、RR 73.84%；
- 三路 ASR 正样本 CER：38.34%；
- 选择性 TSE 在 200 条均匀正样本上将三路共识 CER 从 35.24%降至 33.71%。
- 选择性 TSE 全量端到端：CER 46.60%、RR 88.19%、平均 0.4714 秒/条。

完整实验解释、边界和架构见 `docs/competition_speaker_asr_optimization_2026.md`。

## 快速运行

普通融合版：

```powershell
D:\Python39\python.exe infer.py `
  --meta output\datasetA_all.jsonl `
  --audio-root datasetA `
  --config configs\competition_release_fusion_external.json `
  --output output\release_result.json `
  --stats output\release_stats.json
```

选择性 TSE 版需要独立环境：

```powershell
D:\Python39\python.exe -m venv .venv_tse --system-site-packages
.\.venv_tse\Scripts\python.exe -m pip install -r requirements-tse.txt
.\.venv_tse\Scripts\python.exe scripts\setup_tse_posneg.py
```

`V4` 还需要 `pretrained/tse_posneg_cnceleb_stage2.pt`。该中文适配权重应由团队在确认第三方再分发条件后放到 GitHub Release/LFS；也可以按主实验文档中的 CN-Celeb2 命令重新训练。

本仓库默认不上传数据集和大型第三方权重。模型目录要求和许可证注意事项见 `docs/third_party_and_data_notice.md`。

## 目录说明

- 根目录 Python 文件：端到端推理、声纹、ASR、TSE 和决策逻辑；
- `configs/`：外部校准、CER 优先、平衡、高拒识和选择性 TSE 版本；
- `scripts/`：训练、校准、缓存、评测和结果汇总；
- `docs/`：完整架构、实验记录和数据说明；
- `results/`：可复查的小型 JSON 结果，不含原始音频与数据集；
- `tests/`：CER 与路由规则测试。

## 数据使用声明

DatasetA 没有用于任何模型的梯度训练。它只用于开发评测和开发工作点比较；发布包另含只用 CN-Celeb2 说话人试验校准阈值的配置。
