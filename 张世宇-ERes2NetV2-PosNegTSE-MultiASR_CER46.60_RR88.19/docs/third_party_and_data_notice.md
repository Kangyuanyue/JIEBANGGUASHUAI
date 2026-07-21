# 第三方模型、代码与数据说明

## 仓库不包含的内容

以下内容体积大或受各自许可证/数据协议约束，不应直接提交到本仓库：

- `datasetA/`、`CN-Celeb2_flac/`、`musan/`、`RIRS_NOISES/`；
- `pretrained/` 下的 Paraformer、SenseVoice、ERes2NetV2、CAM++ 模型；
- `external_TSE_PosNeg/` 下的第三方完整源码和官方检查点；
- `.venv_tse/`、缓存音频和全量中间特征。

`scripts/setup_tse_posneg.py` 只负责从官方地址下载 Pos/Neg TSE 源码和基线检查点。正式公开前，应由团队负责人再次确认所有第三方模型、代码、数据以及衍生权重的许可证和再分发条件。

## 主要来源

| 项目 | 用途 | 来源 |
|---|---|---|
| ERes2NetV2 | 主声纹模型 | <https://arxiv.org/abs/2406.02167> |
| CAM++ | 声纹融合模型 | FunASR/ModelScope 模型库 |
| Paraformer | 主中文 ASR | <https://github.com/modelscope/FunASR> |
| SenseVoice | 第二中文 ASR | <https://github.com/modelscope/FunASR> |
| Pos/Neg TSE | 重叠目标说话人提取 | <https://github.com/xu-shitong/TSE-through-Positive-Negative-Enroll> |
| CN-Celeb2 | 中文说话人域适配 | CN-Celeb 官方发布页/协议 |
| MUSAN | 噪声增强 | OpenSLR 17 |
| RIRS_NOISES | 混响增强 | OpenSLR 28 |

## 本项目生成的权重

本地生成但默认不上传的文件：

- `pretrained/tse_posneg_cnceleb_stage2.pt`：800 名训练说话人，主 V4 使用；
- `pretrained/tse_posneg_cnceleb_stage3.pt`：1500 名训练说话人，目标身份准确率更高；
- `pretrained/tse_posneg_cnceleb_robust.pt`：MUSAN+RIRS 实验性鲁棒版本。

如确认可以发布，建议使用 Git LFS 或 GitHub Release，并同时提供 SHA-256、训练配置、基础检查点来源和许可证说明。
