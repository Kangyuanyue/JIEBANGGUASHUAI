import os
import json
import time

import soundfile as sf
import noisereduce as nr
import editdistance

from funasr import AutoModel


# ==================================================
# 热词库
# ==================================================

HOTWORDS = [
    # 空调控制
    "空调开到制热", "制冷", "除湿", "送风",
    "风速加大", "风速减小", "风速调自动",
    "风速五十", "风速九十", "风速十",
    "高速风", "低速风", "中速风", "无风感",
    "向左吹风", "向右吹风", "风往下吹", "风往上吹",
    "防直吹", "开防直吹", "开左右风",
    "节能模式", "关闭节能模式", "开启节能模式",
    "关闭ECO模式", "开启ECO模式",
    "加热故障", "开机空调", "关机空调", "把空调关上",
    "开到二十度", "温度调高", "温度调低",
    "风大一点", "风小一点",
    "开屏幕", "关屏幕",

    # 灯光
    "关灯", "开灯",
    "灯光亮度调到百分之三十",
    "所有灯光调到百分之五十",

    # 屏幕
    "打开显示屏", "关闭显示屏",

    # 音乐
    "播放", "暂停", "下一首", "上一首",
    "播放苦命人", "播放小飞鱼二", "播放吐槽大会",

    # 洗碗机
    "洗碗机暂停工作",

    # 通用指令
    "打开", "关闭", "关掉", "恢复",
    "你好科慕", "hi colmo",
]


# ==================================================
# 热词匹配后处理
# ==================================================

def match_hotword(text, hotwords, threshold=0.45):
    """
    用编辑距离匹配热词库
    返回：(纠正后文本, 匹配到的热词或None, 相似度)
    """
    if not text:
        return text, None, 0.0

    best_match = None
    best_ratio = float('inf')

    for hw in hotwords:
        dist = editdistance.eval(text, hw)
        ratio = dist / max(len(text), len(hw))
        if ratio < best_ratio:
            best_ratio = ratio
            best_match = hw

    if best_ratio <= threshold:
        return best_match, best_match, best_ratio
    else:
        return text, None, best_ratio


# ==================================================
# CER（比赛标准）
# ==================================================

def calc_cer(ref, hyp):
    ref = ref.replace(" ", "")
    hyp = hyp.replace(" ", "")
    dist = editdistance.eval(ref, hyp)
    return dist / max(len(ref), 1)


# ==================================================
# 配置
# ==================================================

JSONL_FILE = "../data/datasetA/pos.jsonl"
DATA_ROOT = "../data/datasetA"

# 调试阶段先跑100条
#MAX_SAMPLES = 100

# 正式测试改为None
MAX_SAMPLES = None

# 热词匹配阈值（越小越严格）
HOTWORD_THRESHOLD = 0.6


# ==================================================
# 清理SenseVoice标签
# ==================================================

def clean_text(text):
    tags = [
        "<|zh|>", "<|en|>",
        "<|Speech|>", "<|BGM|>",
        "<|Applause|>", "<|Laughter|>",
        "<|Cry|>", "<|Sneeze|>",
        "<|Breath|>", "<|Cough|>",
        "<|Sing|>", "<|Speech_Noise|>",
        "<|withitn|>", "<|woitn|>",
        "<|HAPPY|>", "<|SAD|>",
        "<|ANGRY|>", "<|NEUTRAL|>",
        "<|EMO_UNKNOWN|>",
    ]
    for tag in tags:
        text = text.replace(tag, "")
    return text.strip()


# ==================================================
# 加载模型
# ==================================================

print("=" * 60)
print("加载 SenseVoiceSmall")
print("=" * 60)

model = AutoModel(
    model="iic/SenseVoiceSmall",
    trust_remote_code=True,
    remote_code="./model.py",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cpu"
)

print("模型加载完成\n")


# ==================================================
# 读取数据集
# ==================================================

samples = []

with open(JSONL_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            samples.append(json.loads(line))

if MAX_SAMPLES is not None:
    samples = samples[:MAX_SAMPLES]

print(f"测试数量: {len(samples)}\n")


# ==================================================
# 统计变量
# ==================================================

total_time = 0.0
total_audio_time = 0.0

total_cer_original = 0.0
total_cer_corrected = 0.0

exact_match_original = 0
exact_match_corrected = 0

hotword_triggered = 0
hotword_improved = 0
hotword_worsened = 0

failed_count = 0
results = []


# ==================================================
# 开始测试
# ==================================================

for idx, sample in enumerate(samples):

    wav_path = os.path.join(DATA_ROOT, sample["识别音频"])
    ref_text = sample["识别文本"]

    if not os.path.exists(wav_path):
        failed_count += 1
        print(f"文件不存在: {wav_path}")
        continue

    try:
        # ----------------------
        # 降噪
        # ----------------------
        audio, sr = sf.read(wav_path)
        audio = nr.reduce_noise(y=audio, sr=sr)
        duration = len(audio) / sr
        total_audio_time += duration

        tmp_path = wav_path.replace(".wav", "_denoised.wav")
        sf.write(tmp_path, audio, sr)

        # ----------------------
        # 推理
        # ----------------------
        start = time.time()

        res = model.generate(
            input=tmp_path,
            cache={},
            language="zh",
            use_itn=False,      # 中文CER评测关闭ITN
            batch_size_s=60,
            merge_vad=False,    # 短指令关闭合并
        )

        elapsed = time.time() - start
        total_time += elapsed

        # ----------------------
        # 清理标签
        # ----------------------
        raw_text = res[0]["text"]
        hyp_text = clean_text(raw_text)

        # ----------------------
        # 热词后处理
        # ----------------------
        hyp_corrected, matched_hw, similarity = match_hotword(
            hyp_text,
            HOTWORDS,
            threshold=HOTWORD_THRESHOLD
        )

        if matched_hw is not None:
            hotword_triggered += 1

        # ----------------------
        # 计算CER
        # ----------------------
        cer_original = calc_cer(ref_text, hyp_text)
        cer_corrected = calc_cer(ref_text, hyp_corrected)

        total_cer_original += cer_original
        total_cer_corrected += cer_corrected

        # 统计热词纠正效果
        if matched_hw is not None:
            if cer_corrected < cer_original:
                hotword_improved += 1
            elif cer_corrected > cer_original:
                hotword_worsened += 1

        if hyp_text == ref_text:
            exact_match_original += 1
        if hyp_corrected == ref_text:
            exact_match_corrected += 1

        results.append({
            "audio": sample["识别音频"],
            "reference": ref_text,
            "prediction_raw": hyp_text,
            "prediction_corrected": hyp_corrected,
            "matched_hotword": matched_hw,
            "similarity": round(similarity, 4),
            "cer_original": round(cer_original, 4),
            "cer_corrected": round(cer_corrected, 4),
            "time": round(elapsed, 3),
            "duration": round(duration, 3),
        })

        print(
            f"[{idx+1}/{len(samples)}] "
            f"原始CER={cer_original:.2%} → "
            f"纠正CER={cer_corrected:.2%} "
            f"{'✓热词:' + str(matched_hw) if matched_hw else ''}"
        )

    except Exception as e:
        failed_count += 1
        print(f"\n失败: {wav_path}")
        print(e)


# ==================================================
# 汇总
# ==================================================

count = len(results)

avg_cer_original = total_cer_original / count
avg_cer_corrected = total_cer_corrected / count
avg_time = total_time / count
rtf = total_time / total_audio_time

# 分类统计
cmd_early = [r for r in results if int(r['audio'].split('cmd_')[1].split('.')[0]) < 1000]
cmd_late = [r for r in results if int(r['audio'].split('cmd_')[1].split('.')[0]) >= 2000]

print("\n")
print("=" * 60)
print("最终结果")
print("=" * 60)

print(f"成功样本数         : {count}")
print(f"失败样本数         : {failed_count}")
print()
print(f"原始平均CER        : {avg_cer_original:.2%}")
print(f"热词纠正后CER      : {avg_cer_corrected:.2%}")
print(f"CER变化            : {(avg_cer_corrected - avg_cer_original):+.2%}")
print()
print(f"Exact Match(原始)  : {exact_match_original/count:.2%}")
print(f"Exact Match(纠正)  : {exact_match_corrected/count:.2%}")
print()
print(f"热词触发次数       : {hotword_triggered}/{count}")
print(f"热词改善次数       : {hotword_improved}")
print(f"热词变差次数       : {hotword_worsened}")
print()
print(f"平均推理时间       : {avg_time:.3f}s")
print(f"RTF                : {rtf:.4f}")
print()

if cmd_early:
    e_orig = sum(r['cer_original'] for r in cmd_early) / len(cmd_early)
    e_corr = sum(r['cer_corrected'] for r in cmd_early) / len(cmd_early)
    print(f"cmd_0~999  原始CER: {e_orig:.2%} → 纠正后: {e_corr:.2%}")

if cmd_late:
    l_orig = sum(r['cer_original'] for r in cmd_late) / len(cmd_late)
    l_corr = sum(r['cer_corrected'] for r in cmd_late) / len(cmd_late)
    print(f"cmd_2000+  原始CER: {l_orig:.2%} → 纠正后: {l_corr:.2%}")

print("=" * 60)


# ==================================================
# 保存结果
# ==================================================

summary = {
    "model": "SenseVoiceSmall",
    "dataset": "datasetA_pos",
    "note": "datasetA only used for evaluation, not training",
    "hotword_threshold": HOTWORD_THRESHOLD,
    "avg_cer_original": avg_cer_original,
    "avg_cer_corrected": avg_cer_corrected,
    "cer_improvement": avg_cer_original - avg_cer_corrected,
    "exact_match_original": exact_match_original / count,
    "exact_match_corrected": exact_match_corrected / count,
    "hotword_triggered": hotword_triggered,
    "hotword_improved": hotword_improved,
    "hotword_worsened": hotword_worsened,
    "avg_time": avg_time,
    "rtf": rtf,
    "results": results,
}

with open("benchmark_result_sensevoice_pos.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print()
print("结果保存完成: benchmark_result_sensevoice_pos.json")