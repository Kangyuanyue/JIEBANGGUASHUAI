# 本地评测与提交说明

## 1. 官方提交 JSON 结构

推荐结果结构：

```json
{
  "result": {
    "results": [
      {
        "id": "id1",
        "content": "打开客厅空调",
        "label": "打开客厅空调",
        "cer": "0.0"
      },
      {
        "id": "id2",
        "content": "",
        "label": "",
        "cer": "0.0"
      }
    ],
    "final_cer": "0.0",
    "duration": "123.45"
  }
}
```

其中：

- `id`：测试音频名字；
- `content`：系统推理结果，拒识样本应为空字符串；
- `label`：开发 / 测试 A 标签，测试 B 若无标签可为空；
- `cer`：单条样本 CER；
- `final_cer`：正样本平均 CER；
- `duration`：batch=1 推理整个测试集的总耗时，单位秒。

## 2. 本地指标定义

### 2.1 CER

```text
CER = edit_distance(hyp_chars, ref_chars) / len(ref_chars)
```

只在正样本上统计平均 CER：

```text
label != "" -> 正样本
```

### 2.2 RR

```text
RR = correctly_rejected_negative_samples / total_negative_samples
```

拒识样本定义：

```text
label == ""
```

正确拒识：

```text
label == "" and content == ""
```

拒识失败：

```text
label == "" and content != ""
```

## 3. 评测脚本

运行：

```bash
python -m src.scorer \
  --result_json outputs/result.json \
  --report_json outputs/metrics.json
```

输出：

```json
{
  "positive_count": 100,
  "negative_count": 50,
  "final_cer": 0.123,
  "rr": 0.94,
  "negative_false_accept": 3,
  "duration": 120.5
}
```

## 4. 推理计时原则

建议：

1. 模型预加载；
2. warmup 若干条；
3. 开始计时；
4. 对测试集逐条 batch=1 推理；
5. 结束计时；
6. 写入 `duration`。

如果官方明确要求包含模型加载时间，则以官方说明为准。

## 5. 分桶评估

每次实验应至少输出：

| 分桶 | 指标 |
|---|---|
| 正样本全量 | CER |
| 负样本全量 | RR |
| SNR = -5 / 0 / 5 dB | CER / RR |
| overlap = 0 / 25 / 50 / 75 / 100% | CER / RR |
| 短 wake | CER / RR |
| 相似音色负样本 | RR |
| 空调噪声 | CER / RR |
| TSE 触发样本 | CER / 耗时 |
| Direct ASR 样本 | CER / 耗时 |

## 6. 提交前检查清单

- [ ] `run_infer.py` 可在干净环境运行；
- [ ] `requirements.txt` 不缺依赖；
- [ ] JSON 能通过 schema 校验；
- [ ] 拒识样本 content 为空字符串；
- [ ] `duration` 为 batch=1 总推理时间；
- [ ] 模型路径不依赖本机绝对路径；
- [ ] 不提交超大临时文件；
- [ ] 不写针对测试 A 文件名的规则；
- [ ] 代码固定随机种子；
- [ ] 提供 README、技术设计报告、测试报告、使用说明。
