# miao：复杂交互场景的抗干扰语音指令识别方案

本目录用于存放“XH-202615 复杂交互场景的抗干扰语音指令识别技术”比赛的解决方案、算法设计、训练策略、推理脚本骨架和本地评测工具。

## 1. 任务一句话总结

本题不是普通中文 ASR，而是 **基于唤醒音频注册目标说话人身份的远场家居指令识别系统**：系统需要识别唤醒人的后续控制指令，同时拒识非唤醒说话人的语音，并在低信噪比、双人重叠、空调等非人声噪声场景下保持高准确率和高推理效率。

## 2. 核心目标

初赛客观评分围绕三个指标展开：

| 指标 | 权重 | 方案目标 |
|---|---:|---|
| 目标发音人 CER | 40% | 目标人说话时识别文本尽可能准确 |
| 非目标发音人拒识率 RR | 40% | 非目标人说话时输出空字符串，避免误触发 |
| 模型推理效率 | 20% | batch=1 推理快，内存占用低 |

因此，本方案采用 **目标说话人条件化的级联系统**：

```text
Wake Audio -> Target Speaker Embedding
Query Audio -> VAD / SNR / Speaker Similarity / target-speaker pVAD
           -> accept / reject / uncertain gate
           -> Direct ASR or Conditional TSE + ASR
           -> Command Postprocess
           -> Fusion Decision
           -> JSON Result
```

## 3. 推荐最终技术路线

主提交路线选择“高分竞赛方案”：

1. **音频预处理**：16 kHz 单声道、响度归一化、VAD、SNR 估计、必要时轻量降噪。
2. **唤醒音频建模**：从唤醒音频提取目标 speaker embedding，建议 CAM++ / ERes2Net / ECAPA 多裁剪融合。
3. **目标说话人判别**：通过 speaker similarity 和质量感知动态阈值判断 query 是否包含目标人。
4. **target-speaker pVAD**：预测 silence / target / non-target 帧比例，解决整段 embedding 在重叠场景被污染的问题。
5. **条件式目标说话人提取 TSE**：仅在重叠或不确定样本触发 VoiceFilter / SpeakerBeam 风格 TSE，避免所有样本都跑大模型。
6. **中文 ASR**：主模型建议 Paraformer-zh / SenseVoice-small / Zipformer-Transducer 等快速中文 ASR。
7. **指令后处理**：基于家居命令词表、热词、拼音相似和数字规范化降低短指令 CER。
8. **融合拒识决策**：联合 speaker similarity、target ratio、non-target ratio、ASR confidence、SNR、wake quality 做 accept / reject。
9. **推理加速**：优先 ONNX Runtime / TensorRT / FP16 / INT8；TSE 走条件触发；speaker embedding 缓存。

## 4. 目录结构

```text
miao/
  README.md                         # 本说明
  SOLUTION.md                       # 完整比赛方案与特等奖路线
  ALGORITHM.md                      # 算法流程、决策逻辑与伪代码
  requirements.txt                  # Python 依赖建议
  run_infer.py                      # 官方测试集推理入口骨架
  config/
    config.yaml                     # 模型、路径、推理配置
    thresholds.yaml                 # 拒识、接收、融合阈值
  docs/
    requirements_matrix.md          # 比赛规则需求矩阵
    training_augmentation.md        # 训练与数据增强策略
    evaluation_submission.md        # 本地评测与提交说明
    open_source_references.md       # 开源代码、论文、数据资源清单
    self_check.md                   # 自检与修正记录
    experiment_plan.md              # 消融实验计划
  src/
    __init__.py
    preprocess.py                   # 音频预处理工具
    decision.py                     # 融合拒识决策逻辑
    text_norm.py                    # 文本规范化与 CER 计算
    scorer.py                       # 本地 CER / RR / duration 评测
    schema.py                       # JSON 结果生成与校验
    pipeline.py                     # 推理 pipeline 骨架
  scripts/
    push_to_github.sh               # 提交到 GitHub 的命令模板
  models/
    .gitkeep                        # 模型权重不建议直接提交到 GitHub
  outputs/
    .gitkeep
  logs/
    .gitkeep
```

## 5. 快速开始

### 5.1 安装依赖

```bash
cd miao
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 5.2 运行推理骨架

当前 `run_infer.py` 是工程骨架，默认 `--mock` 模式不会加载真实模型，仅用于检查数据流、输出格式和计时逻辑。

```bash
python run_infer.py \
  --input_manifest /path/to/test_manifest.json \
  --output_json outputs/result.json \
  --config config/config.yaml \
  --thresholds config/thresholds.yaml \
  --mock
```

接入真实模型后，去掉 `--mock`，并在 `src/pipeline.py` 中实现：

- `extract_wake_embedding`
- `extract_query_evidence`
- `run_target_speaker_extraction`
- `run_asr`
- `postprocess_command`

### 5.3 本地评测

```bash
python -m src.scorer \
  --result_json outputs/result.json \
  --report_json outputs/metrics.json
```

## 6. GitHub 提交流程

从仓库根目录执行：

```bash
git clone https://github.com/Kangyuanyue/JIEBANGGUASHUAI.git
cd JIEBANGGUASHUAI
mkdir -p miao
# 将本目录所有文件复制到仓库 miao/ 下
git add miao
git commit -m "Add miao solution for anti-interference speech command recognition"
git push origin main
```

如果没有 push 权限，请先在 GitHub 仓库页面确认登录账号是否为仓库所有者，或使用 fork + pull request 的方式提交。

## 7. 关键原则

- **先拒识，后识别**：非目标语音不能因为 ASR 输出合法命令而被接收。
- **先轻量门控，后复杂模型**：只有疑难样本触发 TSE，保护效率分。
- **按比赛指标调参**：阈值目标不是单纯 EER，而是综合 CER、RR、推理时间和内存。
- **防止过拟合测试 A**：必须构造 speaker-disjoint、noise-disjoint、command-template-disjoint 的本地验证集。
- **文档和代码必须可复现**：最终提交前做干净环境复现。
