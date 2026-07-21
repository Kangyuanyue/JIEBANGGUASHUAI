ClearVoice + MossFormer2_SS_16K 语音分离项目

基于阿里巴巴ClearVoice框架与MossFormer2_SS_16K模型的智能语音分离解决方案。

1. 项目组件介绍

1.1 ClearVoice

ClearVoice 是阿里巴巴开源的一站式语音处理工具包，对各类语音模型进行上层封装，提供极简调用接口，开发者无需手动编写模型前向推理、音频预处理等底层逻辑。

核心功能：语音降噪、回声消除、多人重叠语音分离

模型生态：内置多款阿里自研预训练语音模型，MossFormer2 系列是其中用于语音分离的模型

底层依赖：基于 PyTorch 深度学习框架，运行环境必须正确安装 Torch 相关依赖库

1.2 MossFormer2_SS_16K

MossFormer2 是阿里巴巴自研基于 Transformer 架构的语音分离模型，发表于 2024 语音领域顶会，针对多人重叠混叠语音设计。依靠学习不同说话人声学频谱特征实现声源分离，分离效果远优于传统维纳滤波等信号处理算法。

字段释义

SS	Speech Separation	任务类型为语音分离
16K	16 kHz	模型固定输入采样率为 16000 Hz

⚠️ 重要：输入音频必须预处理为 16kHz、单声道 wav 格式，否则推理异常。

模型适用场景
MossFormer2_SS_16K 面向双人重叠对话语音分离，适配 3 秒左右短中文重叠音频；训练数据集包含大量中文混合对话，可以有效区分男女声、音色相近的双人说话人。

2. ClearVoice 与 MossFormer2_SS_16K 二者关系

ClearVoice = 上层调用封装工具（提供统一API、音频预处理、设备管理等功能）

MossFormer2_SS_16K = ClearVoice 内置可加载的预训练语音分离模型（实际执行分离任务的核心算法）