"""
TSE + SV + ASR 完整测试: datasetA/pos 全部 1364 条

用法:
  python scripts/eval_tse_full.py

流程:
  1. 加载训练好的 TSE 模型 (outputs/tse_model)
  2. 对每条 kws → enroll → extract → enhanced
  3. CAM++ SV (kws vs enhanced)
  4. Paraformer-Large ASR (enhanced)
  5. 联合判决 → CER
"""

import os, sys, json, io, unicodedata, string, time
from pathlib import Path

if os.name == "nt":
    _seen = set()
    _o = os.add_dll_directory
    os.add_dll_directory = lambda p: _o(p) if p not in _seen and not _seen.add(p) else type(
        "_D", (), {"close": lambda _: None, "__enter__": lambda s: s, "__exit__": lambda *_: None})()
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import torch, soundfile as sf

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

# ── 模型依赖 ──
from train_tse import TSEModel

# ── 路径 ──
TSE_CKPT = Path("D:/funasr/outputs/tse_model/model.pt")
DATASET_DIR = Path("D:/funasr/dataset/datasetA")
POS_JSONL = DATASET_DIR / "pos.jsonl"

# ── CER 工具 ──
def norm_text(t):
    if t is None: return ""
    t = unicodedata.normalize("NFKC", str(t)).lower().strip()
    return "".join(c for c in t if c not in string.whitespace and not unicodedata.category(c).startswith("P"))

def levenshtein(s1, s2):
    if len(s1) < len(s2): s1, s2 = s2, s1
    prev = list(range(len(s2)+1))
    for i,c1 in enumerate(s1,1):
        cur = [i]
        for j,c2 in enumerate(s2,1):
            cur.append(min(prev[j]+1, cur[j-1]+1, prev[j-1]+(0 if c1==c2 else 1)))
        prev = cur
    return prev[-1]

def cosine_sim(a,b):
    return float(np.dot(a.flatten(),b.flatten())/(np.linalg.norm(a)*np.linalg.norm(b)+1e-10))

def denoise(audio, sr=16000):
    try:
        import noisereduce as nr
        noise = audio[:min(len(audio), sr // 2)]
        return nr.reduce_noise(y=audio, sr=sr, y_noise=noise,
                                prop_decrease=0.85, stationary=True)
    except ImportError:
        return audio


# ── 加载 TSE 模型 ──
print("="*60)
print("  TSE + SV + ASR 完整测试 (1364 条)")
print("="*60)

if not TSE_CKPT.exists():
    print(f"[错误] TSE 模型不存在: {TSE_CKPT}")
    print("  请先运行: python scripts/train_tse.py --light --epochs 10")
    sys.exit(1)

print(f"[TSE] 加载模型: {TSE_CKPT}")
model = TSEModel(light=True)
state = torch.load(TSE_CKPT, map_location="cpu")
if "model" in state: state = state["model"]
model.load_state_dict(state)
model.eval()
print(f"  [OK] 参数: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# ── 加载数据 ──
with open(POS_JSONL, "r", encoding="utf-8") as f:
    samples = [json.loads(l) for l in f if l.strip()]
print(f"[数据] {len(samples)} 条")

# ── 加载 SV + ASR ──
print("[SV] CAM++ ...")
from funasr import AutoModel
sv = AutoModel(model="iic/speech_campplus_sv_zh-cn_16k-common")

print("[ASR] Paraformer-Large ...")
asr = AutoModel(
    model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    vad_model="fsmn-vad",
)

# ── 测试 ──
print(f"\n{'测试中 ...':^60}\n")

total_errs, total_chars = 0, 0
asr_only_errs, asr_only_chars = 0, 0
tse_times, sv_times, asr_times = [], [], []
details = []
sv_scores = []

for i, item in enumerate(samples):
    kws_path = str(DATASET_DIR / item["唤醒音频"])
    cmd_path = str(DATASET_DIR / item["识别音频"])
    ref = item["识别文本"]
    uid = item["id"]

    # 1. TSE
    t0 = time.time()
    kws_a, _ = sf.read(kws_path)
    cmd_a, _ = sf.read(cmd_path)
    for a in [kws_a, cmd_a]:
        if a.ndim > 1: a = a.mean(axis=1)

    # 降噪
    kws_a = denoise(kws_a.astype(np.float32))
    cmd_a = denoise(cmd_a.astype(np.float32))

    kws_t = torch.from_numpy(kws_a.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    cmd_t = torch.from_numpy(cmd_a.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        embed = model.enroll(kws_t)
        enhanced = model.extract(cmd_t, embed)
    enh_np = enhanced.squeeze().numpy()
    tse_times.append(time.time() - t0)

    # 临时文件
    sf.write("_tse_tmp.wav", enh_np, 16000)
    sf.write("_kws_tmp.wav", kws_a.astype(np.float32), 16000)

    # 2. SV
    t0 = time.time()
    ke = sv.generate(input="_kws_tmp.wav")
    ce = sv.generate(input="_tse_tmp.wav")
    sv_score = cosine_sim(np.array(ke[0]["spk_embedding"]), np.array(ce[0]["spk_embedding"]))
    sv_times.append(time.time() - t0)
    sv_scores.append(sv_score)

    # 3. ASR (on TSE enhanced)
    t0 = time.time()
    ar = asr.generate(input="_tse_tmp.wav")
    hyp = ar[0]["text"] if ar and len(ar) > 0 else ""
    asr_times.append(time.time() - t0)

    os.remove("_tse_tmp.wav"); os.remove("_kws_tmp.wav")

    # 4. 联合判决
    nr = norm_text(ref)
    nh = norm_text(hyp)
    asr_conf = len(nh) / max(len(nr), 1) if nr else (0.0 if nh else 1.0)

    if sv_score >= 0.3:
        final_hyp = hyp
    elif asr_conf >= 0.8:
        final_hyp = hyp
    else:
        final_hyp = ""

    cer, errs, cc = 0, 0, 0
    np_, nt = norm_text(final_hyp), norm_text(ref)
    if nt:
        errs = levenshtein(np_, nt)
        cc = len(nt)
    else:
        errs = 0 if not np_ else len(np_)

    total_errs += errs
    total_chars += cc

    # ASR only CER (for comparison)
    cer_a, errs_a, cc_a = 0, 0, 0
    if nt:
        errs_a = levenshtein(nh, nt)
        cc_a = len(nt)
    asr_only_errs += errs_a
    asr_only_chars += cc_a

    details.append({
        "id": uid, "ref": ref, "hyp": final_hyp,
        "raw_asr_hyp": hyp, "sv_score": round(sv_score, 4),
        "cer": round(errs/max(cc,1), 4),
    })

    if (i+1) % 100 == 0:
        cer_now = total_errs/max(total_chars,1)
        print(f"  [{i+1}/{len(samples)}] CER={cer_now:.4f}", flush=True)

# ── 结果 ──
overall_cer = total_errs / total_chars if total_chars > 0 else 0
asr_only_cer = asr_only_errs / asr_only_chars if asr_only_chars > 0 else 0
avg_tse = np.mean(tse_times)
avg_sv_ = np.mean(sv_times)
avg_asr = np.mean(asr_times)
avg_sv_score = np.mean(sv_scores)

print(f"\n{'='*60}")
print(f"  测试结果")
print(f"{'='*60}")
print(f"  样本:          {len(samples)}")
print(f"  TSE CER:       {overall_cer:.4f} ({overall_cer*100:.2f}%)")
print(f"  ASR-only CER:  {asr_only_cer:.4f} ({asr_only_cer*100:.2f}%)  ← 无联合判决")
print(f"  {'─'*50}")
print(f"  总错误/字符:   {total_errs}/{total_chars}")
print(f"  平均 SV 分数:  {avg_sv_score:.4f}")
print(f"  TSE 耗时:      {avg_tse*1000:.0f}ms")
print(f"  SV 耗时:       {avg_sv_*1000:.0f}ms")
print(f"  ASR 耗时:      {avg_asr*1000:.0f}ms")

# 按子集分析
id_0_363 = [d for d in details if d["id"] <= 363]
id_2000 = [d for d in details if 2000 <= d["id"] <= 2999]
if id_0_363:
    e0 = sum(levenshtein(norm_text(d["hyp"]),norm_text(d["ref"])) for d in id_0_363)
    c0 = sum(len(norm_text(d["ref"])) for d in id_0_363)
    print(f"\n  id 0~363 ({len(id_0_363)}条): CER={e0/max(c0,1):.4f}")
if id_2000:
    e2 = sum(levenshtein(norm_text(d["hyp"]),norm_text(d["ref"])) for d in id_2000)
    c2 = sum(len(norm_text(d["ref"])) for d in id_2000)
    print(f"  id 2000~2999 ({len(id_2000)}条): CER={e2/max(c2,1):.4f}")

# 保存
out_path = Path("D:/funasr/outputs/tse_full_result_with_denoise.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump({
        "config": {"total": len(samples), "model": "light_tse_trained"},
        "tse_cer": round(overall_cer, 6),
        "asr_only_cer": round(asr_only_cer, 6),
        "avg_sv_score": round(float(avg_sv_score), 4),
        "total_errors": total_errs,
        "total_chars": total_chars,
        "details": details,
    }, f, ensure_ascii=False, indent=2)
print(f"\n[保存] {out_path}")
print("[完成]")
