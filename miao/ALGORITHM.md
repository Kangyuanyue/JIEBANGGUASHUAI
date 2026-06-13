# 算法设计：目标说话人条件化抗干扰语音指令识别

## 1. 总体算法目标

给定一段唤醒音频 `W` 和一段待识别音频 `Q`，系统需要输出：

```text
f(W, Q) = target_command_text 或 ""
```

其中：

- 如果 `Q` 中包含与 `W` 同一发音人的有效家居指令，则输出目标发音人的指令文本；
- 如果 `Q` 中没有目标发音人，或目标证据不足，则输出空字符串 `""` 完成拒识。

## 2. 输入输出定义

### 2.1 输入

```python
sample = {
    "id": "识别音频 ID",
    "wake_audio": "唤醒音频路径",
    "wake_text": "唤醒文本",
    "query_audio": "待识别音频路径",
    "label": "待识别文本标签，拒识样本为空"
}
```

### 2.2 输出

```python
prediction = {
    "id": sample["id"],
    "content": "识别文本或空字符串",
    "label": sample.get("label", ""),
    "cer": "当前样本 CER"
}
```

## 3. Pipeline 伪代码

```python
def infer_one(sample):
    wake = load_audio(sample.wake_audio)
    query = load_audio(sample.query_audio)

    wake_proc, wake_quality = preprocess_wake(wake)
    query_proc, query_quality = preprocess_query(query)

    if query_quality.no_speech:
        return reject(sample.id)

    target_embedding = speaker_encoder(wake_proc)
    query_embedding = speaker_encoder(query_proc)
    speaker_similarity = cosine(target_embedding, query_embedding)

    pvad_out = target_speaker_pvad(query_proc, target_embedding)
    target_ratio = pvad_out.target_frame_ratio
    non_target_ratio = pvad_out.non_target_frame_ratio
    overlap_prob = pvad_out.overlap_probability

    if should_reject_early(speaker_similarity, target_ratio, query_quality):
        return reject(sample.id)

    if should_use_tse(speaker_similarity, target_ratio, non_target_ratio, overlap_prob, query_quality):
        asr_audio = target_speaker_extraction(query_proc, target_embedding)
        tse_used = True
    else:
        asr_audio = query_proc
        tse_used = False

    asr_text, asr_confidence = asr_model(asr_audio)
    command_text, command_score = command_postprocess(asr_text)

    evidence = {
        "speaker_similarity": speaker_similarity,
        "target_frame_ratio": target_ratio,
        "non_target_frame_ratio": non_target_ratio,
        "overlap_probability": overlap_prob,
        "asr_confidence": asr_confidence,
        "command_prior_score": command_score,
        "wake_quality": wake_quality.score,
        "query_snr": query_quality.snr,
        "tse_used": tse_used,
    }

    if fusion_decision(evidence) == "accept":
        return accept(sample.id, command_text)
    else:
        return reject(sample.id)
```

## 4. 三段式门控

第一层门控用于保护效率和拒识率：

```text
similarity >= accept_high 且 target_ratio >= min_target_ratio
    -> 高置信目标，进入 ASR

similarity <= reject_low 且 target_ratio <= max_target_ratio_for_reject
    -> 高置信非目标，直接 reject

其他情况
    -> 进入 pVAD / TSE / 融合决策
```

建议初始阈值写入 `config/thresholds.yaml`，后续通过本地验证集网格搜索。初始值只作为占位，不代表最终最优：

```yaml
speaker_accept_high: 0.72
speaker_reject_low: 0.45
min_target_ratio: 0.18
max_non_target_ratio: 0.65
fusion_accept_threshold: 0.50
```

## 5. 融合决策函数

### 5.1 线性融合 baseline

```text
score =
    1.20 * speaker_similarity
  + 0.90 * target_frame_ratio
  - 0.70 * non_target_frame_ratio
  + 0.40 * asr_confidence
  + 0.25 * command_prior_score
  - 0.30 * enrollment_bad_quality
  - 0.20 * query_noise_penalty
```

决策：

```text
if score >= fusion_accept_threshold:
    accept
else:
    reject
```

### 5.2 学习型融合器

当有足够本地验证数据后，推荐使用 Logistic Regression / LightGBM / 小 MLP 学习融合器。

输入特征：

| 特征 | 含义 |
|---|---|
| `speaker_similarity` | wake 和 query 的说话人相似度 |
| `speaker_score_margin` | 与 accept / reject 阈值的距离 |
| `target_frame_ratio` | pVAD 预测的目标帧比例 |
| `non_target_frame_ratio` | pVAD 预测的非目标帧比例 |
| `overlap_probability` | 重叠概率 |
| `asr_confidence` | ASR 平均置信度 |
| `command_prior_score` | 是否符合家居命令语法 |
| `wake_speech_duration` | wake 有效语音时长 |
| `wake_snr` | wake 信噪比估计 |
| `query_snr` | query 信噪比估计 |
| `speech_ratio` | query 中有效语音比例 |
| `tse_used` | 是否触发 TSE |

训练目标：

```text
positive sample -> accept
negative sample -> reject
```

但阈值选择不应只优化准确率，而应最大化本地近似比赛分数。

## 6. Target-speaker pVAD 设计

### 6.1 输入

- Query log-mel feature：`T x F`；
- Target speaker embedding：`D`。

### 6.2 结构建议

```text
log-mel -> TDNN / tiny Conformer encoder -> frame representation
speaker embedding -> projection -> FiLM / concat / attention conditioning
conditioned frame representation -> softmax(silence, target, non-target)
```

### 6.3 损失函数

```text
Loss = CE(frame_label, frame_prediction)
     + lambda1 * BCE(target_presence)
     + lambda2 * contrastive_speaker_loss
```

其中帧级标签可由合成数据自动生成：

- 目标人有语音的帧：target；
- 干扰人有语音且目标人无语音的帧：non-target；
- 两人同时说话的帧：target + overlap 标记，主标签可设为 target，同时额外训练 overlap head；
- 静音帧：silence。

## 7. 条件式 TSE 设计

### 7.1 触发条件

```text
use_tse = (
    overlap_probability > overlap_threshold
    or (speaker_reject_low < speaker_similarity < speaker_accept_high)
    or (target_frame_ratio > min_target_ratio and non_target_frame_ratio > non_target_threshold)
    or (query_snr < low_snr_threshold and target_frame_ratio > target_low_snr_ratio)
)
```

### 7.2 TSE 模型选择

可采用 VoiceFilter / SpeakerBeam 风格：

```text
query log-mel + target speaker embedding -> mask estimator -> target spectrogram -> waveform reconstruction
```

训练损失：

```text
Loss = SI-SDR loss
     + alpha * mel reconstruction loss
     + beta * ASR CTC / CE auxiliary loss
```

ASR-aware 辅助损失很重要，因为目标不是获得听感最好的增强音频，而是降低目标发音人指令的 CER。

## 8. ASR 与命令后处理

### 8.1 ASR 推荐

主模型：Paraformer-zh 或 SenseVoice-small。  
候选增强：Zipformer / WeNet Conformer。  
不建议主推理使用 Whisper large，因为推理效率压力较大，可作为 teacher 或离线伪标签模型。

### 8.2 指令命令语法

可定义家居命令模板：

```text
[动作] + [房间/设备] + [参数]

动作：打开、关闭、调高、调低、设置、切换、暂停、继续
房间：客厅、卧室、厨房、书房、阳台
设备：空调、灯、电视、窗帘、净化器、风扇
参数：温度、风速、模式、亮度、音量
```

后处理流程：

1. 文本规范化；
2. 删除标点、空格和无意义语气词；
3. 纠正常见同音错误；
4. 统一数字格式；
5. 仅当身份接收后进行强命令纠错。

## 9. CER / RR 联合优化

### 9.1 错误类型

| 错误类型 | 对应指标 | 处理 |
|---|---|---|
| 正样本识别错 | CER | ASR 微调、TSE、后处理 |
| 正样本被拒识 | CER / 有效样本丢失 | 降低拒识阈值、提高 target evidence 权重 |
| 负样本被接收 | RR | 提高拒识阈值、hard negative 训练、pVAD |
| 重叠样本输出干扰人文本 | CER | TSE、target-speaker pVAD、ASR rerank |
| 无语音输出文本 | RR / 误触发 | VAD、ASR confidence gate |

### 9.2 阈值搜索目标

```python
score = 0.4 * positive_score + 0.4 * rr_score + 0.1 * time_score + 0.1 * memory_score
```

如果官方没有给出效率分归一化细则，先在本地使用相对排名或归一化指标，最终以官方说明为准。

## 10. 复杂度控制

为避免效率分下降，推理路径分为三类：

| 路径 | 条件 | 模块 | 目标 |
|---|---|---|---|
| Reject Fast Path | 明显非目标 / 无语音 | VAD + SV + pVAD | 快速拒识 |
| Direct ASR Path | 明显目标且无重叠 | VAD + SV + ASR | 快速识别 |
| Hard Case Path | 重叠 / 低 SNR / 不确定 | VAD + SV + pVAD + TSE + ASR | 提升疑难样本 CER |

## 11. 最终算法伪代码摘要

```python
for sample in test_set:
    timer.start()

    wake, query = load(sample)
    wake_feat = preprocess(wake)
    query_feat = preprocess(query)

    if query_feat.no_speech:
        content = ""
    else:
        target_emb = encode_speaker(wake_feat.audio)
        sv_score = score_speaker(target_emb, query_feat.audio)
        pvad = run_pvad(query_feat.audio, target_emb)

        if is_clear_reject(sv_score, pvad, query_feat):
            content = ""
        else:
            if need_tse(sv_score, pvad, query_feat):
                asr_audio = run_tse(query_feat.audio, target_emb)
            else:
                asr_audio = query_feat.audio

            raw_text, asr_conf = run_asr(asr_audio)
            fixed_text, cmd_score = postprocess(raw_text)
            accept = final_decision(sv_score, pvad, asr_conf, cmd_score, wake_feat, query_feat)
            content = fixed_text if accept else ""

    timer.stop()
    write_result(sample.id, content, sample.label)
```
