# 第三方项目与数据说明

本目录只包含苗桐郡编写或整理的实验程序、配置和报告，不重新分发下列模型权重或第三方源代码。

| 组件 | 用途 | 来源 | 许可或使用说明 |
|---|---|---|---|
| CosyVoice | 多说话人TTS | https://github.com/QwenAudio/CosyVoice | 仓库代码Apache-2.0；模型权重以模型卡为准 |
| SenseVoice-Small | ASR与轻量微调 | https://github.com/QwenAudio/SenseVoice | 仓库代码与模型权重分别遵循其LICENSE和模型卡，使用时保留SenseVoice/FunASR归属 |
| SpeechBrain | 分离与说话人编码 | https://github.com/speechbrain/speechbrain | Apache-2.0 |
| SepFormer WHAMR 16k | 双路语音分离 | https://huggingface.co/speechbrain/sepformer-whamr16k | 模型卡标注Apache-2.0 |
| ECAPA VoxCeleb | 说话人相似度 | https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb | 以模型卡和SpeechBrain许可为准 |
| HI-MIA-CW | 公开说话人参考 | https://www.openslr.org/120/ | CC BY-SA 4.0 |

试听样例的具体许可和引用见`samples/LICENSE.md`。上传前未发现上述项目的权重、完整源码或原始数据被误打包。
