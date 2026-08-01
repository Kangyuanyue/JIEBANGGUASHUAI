# 模型信息

## 推荐模型：ERes2NetV2

- **ModelScope ID**: `iic/speech_eres2netv2_sv_zh-cn_16k-common`
- **ModelScope 链接**: https://www.modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common
- **框架**: 3D-Speaker / FunASR
- **输入**: 16kHz 单声道音频
- **输出**: 192 维说话人 embedding
- **参数量**: 17.8M

### 推理代码

```python
from funasr import AutoModel
import numpy as np

# 加载模型（自动下载或使用缓存）
model = AutoModel(
    model="iic/speech_eres2netv2_sv_zh-cn_16k-common",
    device="cuda",
    disable_update=True
)

# 提取声纹
res_kws = model.generate(input="kws.wav")
res_cmd = model.generate(input="cmd.wav")

emb_kws = res_kws[0]["spk_embedding"].squeeze().cpu().numpy()  # [192]
emb_cmd = res_cmd[0]["spk_embedding"].squeeze().cpu().numpy()  # [192]

# 余弦相似度
sim = np.dot(emb_kws, emb_cmd) / (np.linalg.norm(emb_kws) * np.linalg.norm(emb_cmd))

# 拒识判断
threshold = 0.27
if sim >= threshold:
    print(f"接受 (sim={sim:.4f})")
else:
    print(f"拒绝 (sim={sim:.4f})")
```

## 备选模型：CAM++

- **ModelScope ID**: `iic/speech_campplus_sv_zh-cn_16k-common`
- **ModelScope 链接**: https://www.modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common
- **参数量**: 7.2M
- **用法**: 替换上述代码中的 model 参数为 `"cam++"` 即可
