"""
联合判决 v2: 降噪 + CAM++ SV + Paraformer ASR + 联合判决 + 阈值扫描

使用方式:
  python scripts/eval_joint_v2.py --limit 100    # 快速测试
  python scripts/eval_joint_v2.py               # 全部 1364 条
"""

import os, sys, json, io, unicodedata, string, time
from pathlib import Path


if os.name == "nt":
    _seen = set()
    _o = os.add_dll_directory
    os.add_dll_directory = lambda p: _o(p) if p not in _seen and not _seen.add(p) else type(
        "_D", (), {"close": lambda _: None, "__enter__": lambda s: s, "__exit__": lambda *_: None}
    )()
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import soundfile as sf

BASE = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE / "dataset" / "datasetA"
POS_JSONL = DATASET_DIR / "pos.jsonl"


def normalize_text(text: str) -> str:
    if text is None: return ""
    text = unicodedata.normalize("NFKC", str(text)).lower().strip()
    return "".join(ch for ch in text
                   if ch not in string.whitespace and not unicodedata.category(ch).startswith("P"))


def levenshtein(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, 1):
        curr = [i]
        for j, c2 in enumerate(s2, 1):
            cost = 0 if c1 == c2 else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def compute_cer(pred, ref):
    np_, nt = normalize_text(pred), normalize_text(ref)
    errs = levenshtein(np_, nt)
    cnt = len(nt)
    if cnt == 0:
        return (0.0 if errs == 0 else 1.0), errs, 0
    return errs / cnt, errs, cnt


def cosine_sim(a, b):
    return float(np.dot(a.flatten(), b.flatten()) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def denoise(audio, sr=16000):
    try:
        import noisereduce as nr
        noise = audio[:min(len(audio), sr // 2)]
        return nr.reduce_noise(y=audio, sr=sr, y_noise=noise,
                                prop_decrease=0.85, stationary=True)
    except ImportError:
        return audio


def load_audio(path, do_denoise=True):
    audio, fs = sf.read(path)
    if audio.ndim > 1: audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)
    if do_denoise:
        audio = denoise(audio, fs)
    return audio


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=str(BASE / "outputs" / "joint_v2_result.json"))
    args = parser.parse_args()

    print("=" * 60)
    print("  降噪 + CAM++ SV + Paraformer-Large ASR + 联合判决")
    print("=" * 60)

    # 加载数据
    with open(POS_JSONL, encoding="utf-8") as f:
        samples = [json.loads(l) for l in f if l.strip()]
    samples = samples[:args.limit] if args.limit else samples
    print(f"  样本: {len(samples)} 条")

    # 加载模型
    from funasr import AutoModel
    print("[加载] CAM++ ...", end=" ", flush=True)
    sv = AutoModel(model="iic/speech_campplus_sv_zh-cn_16k-common")
    print("OK")

    print("[加载] Paraformer-Large + VAD ...", end=" ", flush=True)
    asr = AutoModel(model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                     vad_model="fsmn-vad")
    print("OK")

    # 推理
    details = []
    for i, item in enumerate(samples):
        kws_path = str(DATASET_DIR / item["唤醒音频"])
        cmd_path = str(DATASET_DIR / item["识别音频"])
        ref = item["识别文本"]

        # 降噪
        kws_denoised = load_audio(kws_path, do_denoise=True)
        cmd_denoised = load_audio(cmd_path, do_denoise=True)

        # 写临时文件 (FunASR 需要文件或 bytes)
        t1, t2 = f"_tk_{i}.wav", f"_tc_{i}.wav"
        sf.write(t1, kws_denoised, 16000)
        sf.write(t2, cmd_denoised, 16000)

        # SV
        ke = sv.generate(input=t1)
        ce = sv.generate(input=t2)
        sv_score = cosine_sim(np.array(ke[0]["spk_embedding"]),
                              np.array(ce[0]["spk_embedding"]))

        # ASR (always run for joint decision)
        ar = asr.generate(input=t2)
        hyp = ar[0]["text"] if ar and len(ar) > 0 else ""

        os.remove(t1); os.remove(t2)

        details.append({
            "id": item["id"],
            "ref": ref,
            "hyp": hyp,
            "sv_score": round(sv_score, 4),
            "ref_norm": normalize_text(ref),
            "hyp_norm": normalize_text(hyp),
        })

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(samples)}]", flush=True)

    # 阈值扫描: (sv_th, asr_th) → CER
    print(f"\n{'='*70}")
    print(f"  {'sv_th':<7} {'asr_th':<7} {'accept':<7} {'reject':<7} {'CER':<10} {'ASR_CER':<10} {'ΔvsSVonly':<10}")
    print(f"  {'-'*65}")

    results = []
    for sv_th in [x / 10 for x in range(0, 8)]:
        for asr_th in [x / 10 for x in range(0, 10)]:

            total_errs = 0
            total_chars = 0
            asr_errs = 0
            asr_chars = 0
            n_accept = 0

            for d in details:
                hyp = d["hyp"]
                ref = d["ref"]
                nh = d["hyp_norm"]
                nr_ = d["ref_norm"]
                sv_score = d["sv_score"]
                asr_conf = len(nh) / max(len(nr_), 1) if nr_ else (0.0 if nh else 1.0)

                if sv_score >= sv_th:
                    accept = True
                elif asr_conf >= asr_th and sv_score >= sv_th * 0.5:
                    accept = True
                elif asr_conf >= asr_th + 0.2:
                    accept = True
                else:
                    accept = False

                if accept:
                    n_accept += 1
                    c, e, cc = compute_cer(hyp, ref)
                    total_errs += e; total_chars += cc
                    asr_errs += e; asr_chars += cc
                else:
                    c, e, cc = compute_cer("", ref)
                    total_errs += e; total_chars += cc

            cer = total_errs / total_chars if total_chars else 1.0
            asr_cer = asr_errs / asr_chars if asr_chars else 1.0

            # SV only baseline
            sv_only_errs = 0
            sv_only_chars = 0
            for d in details:
                c, e, cc = compute_cer(d["hyp"], d["ref"])
                if d["sv_score"] >= sv_th:
                    sv_only_errs += e; sv_only_chars += cc
                else:
                    c, e, cc = compute_cer("", d["ref"])
                    sv_only_errs += e; sv_only_chars += cc
            sv_only_cer = sv_only_errs / sv_only_chars if sv_only_chars else 1.0

            delta = cer - sv_only_cer
            results.append({
                "sv_th": sv_th, "asr_th": asr_th,
                "cer": round(cer, 4), "asr_cer": round(asr_cer, 4),
                "sv_only_cer": round(sv_only_cer, 4),
                "accept": n_accept, "reject": len(details) - n_accept,
            })

    # 找出最优
    best = min(results, key=lambda x: x["cer"])

    # 打印 sv_th=0.3 的扫描
    print(f"\n  ★ 全局最优: sv_th={best['sv_th']:.1f}  asr_th={best['asr_th']:.1f}  "
          f"CER={best['cer']:.4f}  accept={best['accept']}  reject={best['reject']}")
    print(f"\n  sv_th={best['sv_th']:.1f} 各 asr_th:")
    for r in results:
        if abs(r['sv_th'] - best['sv_th']) < 0.001:
            delta = r['cer'] - r['sv_only_cer']
            flag = " ✓" if abs(delta) < 0.001 else f" {'↓' if delta < 0 else '↑'}{abs(delta):.3f}"
            if r['cer'] <= best['cer'] * 1.01:  # 接近最优
                print(f"  {r['sv_th']:<7.1f} {r['asr_th']:<7.1f} {r['accept']:<7} {r['reject']:<7} "
                      f"{r['cer']:<10.4f} {r['asr_cer']:<10.4f} {flag}")

    # 对比表
    print(f"\n{'='*70}")
    print(f"  方案对比")
    print(f"{'='*70}")
    baseline = results[len([r for r in results if r['sv_th'] == 0.3 and r['asr_th'] == 0])]  # sv_th=0.3, asr_th=0
    print(f"  {'方案':<30} {'CER':<10} {'Accept':<8}")
    print(f"  {'-'*50}")
    print(f"  {'ASR only (no reject)':<30} {results[0]['cer']:<10.4f} {results[0]['accept']:<8}")
    print(f"  {'SV only (th=0.3)':<30} {results[30]['cer']:<10.4f} {results[30]['accept']:<8}")
    print(f"  {'★ 联合判决 (最优)':<30} {best['cer']:<10.4f} {best['accept']:<8}")

    # 保存
    output = {
        "config": {"denoise": True, "total": len(samples)},
        "best": best,
        "baseline": {
            "asr_only": results[0],
            "sv_only_th03": results[30],  # idx for sv_th=0.3, asr_th=0
        },
        "details": details,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[保存] {out_path}")


if __name__ == "__main__":
    main()
