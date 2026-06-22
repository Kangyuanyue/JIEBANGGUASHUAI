import os
import time
import glob
import numpy as np
import soundfile as sf
from funasr import AutoModel
import audiomentations as AA

# 路径设置
AUDIO_DIR = "data/data_aishell/wav"
TRANSCRIPT_FILE = "data/data_aishell/transcript/aishell_transcript_v0.8.txt"
NOISY_DIR = "data/data_aishell/wav_noisy"
os.makedirs(NOISY_DIR, exist_ok=True)

# 加载标注文本
print("加载标注文本...")
transcripts = {}
with open(TRANSCRIPT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            key = parts[0]
            text = "".join(parts[1:])
            transcripts[key] = text

# 定义加噪方案（模拟真实噪声场景）
augment = AA.Compose([
    AA.AddGaussianNoise(min_amplitude=0.01, max_amplitude=0.05, p=1.0),
    AA.AddBackgroundNoise(
        sounds_path=None,  # 无背景音文件，只用高斯噪声
        p=0.0
    ) if False else AA.TimeStretch(min_rate=0.9, max_rate=1.1, p=0.3),
])

# 简单加噪函数（高斯噪声，SNR约15dB）
def add_noise(audio, snr_db=15):
    signal_power = np.mean(audio ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
    return (audio + noise).astype(np.float32)

# 加载模型
print("加载模型...")
model = AutoModel(
    model="iic/SenseVoiceSmall",
    trust_remote_code=True,
    remote_code="./model.py",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cpu",
)

# 找测试音频
wav_files = glob.glob(os.path.join(AUDIO_DIR, "**/*.wav"), recursive=True)[:100]
print(f"共找到 {len(wav_files)} 条测试音频")

# 计算CER
def calc_cer(ref, hyp):
    ref = ref.replace(" ", "")
    hyp = hyp.replace(" ", "")
    import difflib
    matcher = difflib.SequenceMatcher(None, ref, hyp)
    distance = len(ref) + len(hyp) - 2 * sum(t.size for t in matcher.get_matching_blocks())
    return distance / max(len(ref), 1)

# 测试不同噪声强度
for snr_db in [20, 10, 5]:
    print(f"\n===== 噪声强度 SNR={snr_db}dB =====")
    total_cer = 0
    total_time = 0
    count = 0

    for wav_path in wav_files:
        key = os.path.splitext(os.path.basename(wav_path))[0]
        if key not in transcripts:
            continue

        ref_text = transcripts[key]

        # 加噪并保存临时文件
        audio, sr = sf.read(wav_path)
        noisy_audio = add_noise(audio, snr_db=snr_db)
        noisy_path = os.path.join(NOISY_DIR, f"{key}_snr{snr_db}.wav")
        sf.write(noisy_path, noisy_audio, sr)

        # 识别
        start = time.time()
        res = model.generate(
            input=noisy_path,
            cache={},
            language="zh",
            use_itn=False,
            batch_size_s=60,
        )
        elapsed = time.time() - start

        # 清理标签
        hyp_text = res[0]["text"]
        for tag in ["<|zh|>", "<|NEUTRAL|>", "<|SAD|>", "<|HAPPY|>", "<|ANGRY|>",
                    "<|Speech|>", "<|withitn|>", "<|woitn|>", "<|BGM|>", "<|Laughter|>"]:
            hyp_text = hyp_text.replace(tag, "")
        hyp_text = hyp_text.strip()

        cer = calc_cer(ref_text, hyp_text)
        total_cer += cer
        total_time += elapsed
        count += 1

    print(f"测试条数: {count}")
    print(f"平均 CER: {total_cer/count:.2%}")
    print(f"平均耗时: {total_time/count:.2f}s/条")

print("\n========== 汇总对比 ==========")
print("干净音频 CER:       4.12%  (之前测的结果)")
print("以上为不同噪声强度下的 CER，SNR越小噪声越大")