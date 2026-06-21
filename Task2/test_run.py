from funasr import AutoModel

model_dir = "iic/SenseVoiceSmall"

model = AutoModel(
    model=model_dir,
    trust_remote_code=True,
    remote_code="./model.py",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cpu",
)

res = model.generate(
    input=f"{model.model_path}/example/en.mp3",
    cache={},
    language="auto",
    use_itn=True,
    batch_size_s=60,
    merge_vad=True,
    merge_length_s=15,
)

# 直接输出原始结果
print(res[0]["text"])