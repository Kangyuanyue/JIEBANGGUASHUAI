"""
从 AISHELL-1 生成 TSE 训练数据

AISHELL-1: 400 说话人, 178h, 16kHz, 中文普通话

生成策略 (内存友好):
  1. 读取说话人-文件映射 (不加载音频)
  2. 随机配对: 目标说话人 + 干扰说话人
  3. 逐条生成: 读一个 → 混合 → 保存 → 释放

输出 (data/tse_aishell/):
  train/  val/ 各含:
    enroll/   注册音频 (目标说话人的 1s 片段)
    mixture/  混合音频 (目标 + 干扰)
    target/   干净目标音频
  train.jsonl / val.jsonl

用法:
  python scripts/prepare_tse_aishell.py          # 默认 2400 条
  python scripts/prepare_tse_aishell.py --limit 1200
"""

import os, sys, json, random, argparse
from pathlib import Path
import numpy as np

try:
    import soundfile as sf
    HAS_SF = True
except ImportError:
    print("[错误] 需要 soundfile: pip install soundfile")
    sys.exit(1)

BASE = Path(__file__).resolve().parent.parent
AISHELL_DIR = BASE / "dataset" / "data_aishell"
TSE_DIR = BASE / "data" / "tse_aishell"

SAMPLE_RATE = 16000
MAX_ENROLL_SEC = 1.2      # 注册音频最大长度
MAX_TARGET_SEC = 4.0      # 目标音频最大长度
SNR_LEVELS = [15, 10, 5, 0, -5]


def load_transcript(path: Path) -> dict:
    """读取 AISHELL-1 转录文本: utterance_id → text"""
    mapping = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                mapping[parts[0]] = parts[1].replace(" ", "")
    return mapping


def load_audio(path: Path) -> np.ndarray:
    """加载音频, 归一化到 [-1,1]"""
    audio, sr = sf.read(str(path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    # AISHELL-1 是 16kHz WAV, 已经是 PCM16
    peak = np.abs(audio).max()
    if peak > 1.0:
        audio = audio / peak
    return audio.astype(np.float32)


def save_audio(audio: np.ndarray, path: Path):
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(str(path), audio, SAMPLE_RATE)


def make_mixture(target: np.ndarray, interference: np.ndarray, snr_db: float) -> np.ndarray:
    """按 SNR 混合两段音频"""
    if len(interference) < len(target):
        repeats = len(target) // len(interference) + 1
        interference = np.tile(interference, repeats)
    interference = interference[:len(target)]

    target_pow = np.mean(target ** 2) + 1e-10
    interf_pow = np.mean(interference ** 2) + 1e-10
    scale = np.sqrt(target_pow / (interf_pow * (10 ** (snr_db / 10))))

    mixture = target + interference * scale
    return np.clip(mixture, -1.0, 1.0)


def collect_speaker_files(transcript: dict, wav_dir: Path) -> dict:
    """
    按说话人收集音频文件: speaker_id → [(utt_id, wav_path, text), ...]
    只保留转录文本存在的文件
    """
    speakers = {}
    for wav_path in wav_dir.rglob("*.wav"):
        utt_id = wav_path.stem  # BAC009S0002W0122
        if utt_id not in transcript:
            continue
        # 提取说话人 ID (Sxxxx)
        speaker_id = utt_id.split("S")[1].split("W")[0]
        speaker_id = f"S{speaker_id}"
        if speaker_id not in speakers:
            speakers[speaker_id] = []
        speakers[speaker_id].append((utt_id, wav_path, transcript[utt_id]))
    return speakers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=2400,
                        help="生成样本数 (默认 2400)")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 50)
    print("  TSE 数据生成: AISHELL-1")
    print("=" * 50)

    # 1. 读取转录文本
    trans_path = AISHELL_DIR / "transcript" / "aishell_transcript_v0.8.txt"
    transcript = load_transcript(trans_path)
    print(f"  转录文本: {len(transcript)} 条")

    # 2. 收集说话人文件 (只用 train 集)
    train_dir = AISHELL_DIR / "wav" / "train"
    speakers = collect_speaker_files(transcript, train_dir)
    print(f"  说话人: {len(speakers)} 个")
    print(f"  总文件: {sum(len(v) for v in speakers.values())} 条")

    # 过滤掉文件数 < 3 的说话人 (至少 2 条做 mix, 1 条做 enroll)
    speakers = {k: v for k, v in speakers.items() if len(v) >= 3}
    spk_ids = list(speakers.keys())
    print(f"  有效说话人: {len(spk_ids)} 个")

    # 3. 生成 TSE 配对
    TSE_DIR.mkdir(parents=True, exist_ok=True)
    for sub in ["train", "val"]:
        for typ in ["enroll", "mixture", "target"]:
            (TSE_DIR / sub / typ).mkdir(parents=True, exist_ok=True)

    records = []
    total = args.limit
    val_size = int(total * args.val_ratio)

    print(f"\n  生成 {total} 条 ({total - val_size} train + {val_size} val) ...")
    print(f"  (逐条生成, 内存友好)")

    for i in range(total):
        # 随机选目标说话人
        target_spk = random.choice(spk_ids)
        target_files = speakers[target_spk]

        # 选 2 条不同文件: 一条做 target, 一条做 enrollment
        t1, t2 = random.sample(target_files, 2)
        utt_id1, wav_path1, text1 = t1  # target utterance
        utt_id2, wav_path2, _ = t2      # enrollment utterance

        # 加载目标音频
        target_audio = load_audio(wav_path1)
        if len(target_audio) > int(MAX_TARGET_SEC * SAMPLE_RATE):
            target_audio = target_audio[:int(MAX_TARGET_SEC * SAMPLE_RATE)]

        # 加载注册音频 (取前 ~1s)
        enroll_audio = load_audio(wav_path2)
        max_enroll = int(MAX_ENROLL_SEC * SAMPLE_RATE)
        if len(enroll_audio) > max_enroll:
            enroll_audio = enroll_audio[:max_enroll]

        # 选干扰 (不同说话人, 50% 概率用高斯噪声)
        if random.random() < 0.5:
            inter_spk = random.choice([s for s in spk_ids if s != target_spk])
            inter_file = random.choice(speakers[inter_spk])
            interference = load_audio(inter_file[1])
        else:
            noise_len = len(target_audio)
            interference = np.random.randn(noise_len).astype(np.float32) * 0.03

        # 随机 SNR
        snr = random.choice(SNR_LEVELS)
        mixture = make_mixture(target_audio, interference, snr)

        # 保存
        is_val = i >= (total - val_size)
        split = "val" if is_val else "train"
        uid = f"aishell_tse_{i:06d}"
        uid2 = f"aishell_tse_{i:06d}"

        enroll_path = TSE_DIR / split / "enroll" / f"{uid}.wav"
        mixture_path = TSE_DIR / split / "mixture" / f"{uid}.wav"
        target_path = TSE_DIR / split / "target" / f"{uid}.wav"

        save_audio(enroll_audio, enroll_path)
        save_audio(mixture, mixture_path)
        save_audio(target_audio, target_path)

        records.append({
            "key": uid,
            "enroll_path": str(enroll_path.resolve()),
            "mixture_path": str(mixture_path.resolve()),
            "target_path": str(target_path.resolve()),
            "target_text": text1.replace(" ", ""),
            "snr": snr,
            "speaker_id": target_spk,
            "split": split,
        })

        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{total}]", flush=True)

    # 保存 JSONL
    for split_name in ["train", "val"]:
        split_records = [r for r in records if r["split"] == split_name]
        jsonl_path = TSE_DIR / f"{split_name}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for r in split_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  → {jsonl_path} ({len(split_records)} 条)")

    # 统计
    print(f"\n{'='*50}")
    print(f"  生成完成!")
    print(f"  总样本: {len(records)}")
    print(f"  Train:  {total - val_size}")
    print(f"  Val:    {val_size}")
    print(f"  目录:   {TSE_DIR}")
    print(f"  磁盘占用: ~{(total * 3 * 100 * 1024) / 1024**3:.1f}GB (估计)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
