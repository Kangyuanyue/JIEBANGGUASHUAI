import os
import time
import glob
from funasr import AutoModel

# 路径设置
AUDIO_DIR = "data/data_aishell/wav/test"
TRANSCRIPT_FILE = "data/data_aishell/transcript/aishell_transcript_v0.8.txt"

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
wav_files = glob.glob(os.path.join(AUDIO_DIR, "**/*.wav"), recursive=True)[:500]
print(f"共找到 {len(wav_files)} 条测试音频")

# 计算CER
def calc_cer(ref, hyp):
    ref = ref.replace(" ", "")
    hyp = hyp.replace(" ", "")
    import difflib
    matcher = difflib.SequenceMatcher(None, ref, hyp)
    distance = len(ref) + len(hyp) - 2 * sum(t.size for t in matcher.get_matching_blocks())
    return distance / max(len(ref), 1)

# 开始评测
total_cer = 0
total_time = 0
count = 0

for wav_path in wav_files:
    key = os.path.splitext(os.path.basename(wav_path))[0]
    if key not in transcripts:
        continue

    ref_text = transcripts[key]

    start = time.time()
    res = model.generate(
        input=wav_path,
        cache={},
        language="zh",
        use_itn=False,
        batch_size_s=60,
    )
    elapsed = time.time() - start

    # 提取识别结果（去掉标签）
    hyp_text = res[0]["text"]
    for tag in ["<|zh|>", "<|NEUTRAL|>", "<|SAD|>", "<|HAPPY|>", "<|ANGRY|>",
                "<|Speech|>", "<|withitn|>", "<|woitn|>", "<|BGM|>", "<|Laughter|>"]:
        hyp_text = hyp_text.replace(tag, "")
    hyp_text = hyp_text.strip()

    cer = calc_cer(ref_text, hyp_text)
    total_cer += cer
    total_time += elapsed
    count += 1

    print(f"[{count}] {key}")
    print(f"  参考: {ref_text}")
    print(f"  识别: {hyp_text}")
    print(f"  CER: {cer:.2%}  耗时: {elapsed:.2f}s")

print("\n========== 评测结果 ==========")
print(f"测试条数: {count}")
print(f"平均 CER: {total_cer/count:.2%}")
print(f"平均耗时: {total_time/count:.2f}s/条")
print(f"总耗时:   {total_time:.1f}s")