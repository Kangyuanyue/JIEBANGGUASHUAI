#!/usr/bin/env python3
"""Build noisy/reverberant speaker verification trials from existing trials."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal


def read_trials(path: Path, limit: int) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit > 0 else rows


def collect_wavs(root: Path, subdir: str = "") -> list[Path]:
    base = root / subdir if subdir else root
    return sorted(p for p in base.rglob("*.wav") if p.is_file())


def load_audio(path: Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        import torchaudio
        import torch

        t = torch.from_numpy(wav)
        wav = torchaudio.functional.resample(t.unsqueeze(0), sr, target_sr).squeeze(0).numpy().astype(np.float32)
        sr = target_sr
    return wav, sr


def peak_norm(wav: np.ndarray) -> np.ndarray:
    peak = max(float(np.max(np.abs(wav))), 1e-6)
    return (wav / peak * 0.95).astype(np.float32)


def match_len(noise: np.ndarray, length: int, rng: random.Random) -> np.ndarray:
    if noise.size >= length:
        start = rng.randint(0, noise.size - length)
        return noise[start : start + length]
    reps = int(np.ceil(length / max(1, noise.size)))
    return np.tile(noise, reps)[:length]


def add_noise(clean: np.ndarray, noise: np.ndarray, snr_db: float, rng: random.Random) -> np.ndarray:
    noise = match_len(noise, clean.size, rng)
    clean_power = float(np.mean(clean * clean) + 1e-12)
    noise_power = float(np.mean(noise * noise) + 1e-12)
    scale = np.sqrt(clean_power / (noise_power * (10.0 ** (snr_db / 10.0))))
    return peak_norm(clean + noise * scale)


def apply_rir(clean: np.ndarray, rir: np.ndarray) -> np.ndarray:
    if rir.size == 0:
        return clean
    rir = rir.astype(np.float32)
    rir = rir / max(float(np.sqrt(np.sum(rir * rir))), 1e-6)
    out = signal.fftconvolve(clean, rir, mode="full")[: clean.size]
    return peak_norm(out.astype(np.float32))


def write_audio(path: Path, wav: np.ndarray, sr: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav.astype(np.float32), sr)


def augment_file(
    src_path: Path,
    out_path: Path,
    mode: str,
    snr_db: float,
    noise_files: list[Path],
    rir_files: list[Path],
    rng: random.Random,
) -> dict:
    clean, sr = load_audio(src_path)
    wav = clean
    meta = {"mode": mode, "snr_db": snr_db, "noise": "", "rir": ""}
    if "rir" in mode:
        rir_file = rng.choice(rir_files)
        rir, _ = load_audio(rir_file)
        wav = apply_rir(wav, rir)
        meta["rir"] = str(rir_file)
    if "noise" in mode:
        noise_file = rng.choice(noise_files)
        noise, _ = load_audio(noise_file)
        wav = add_noise(wav, noise, snr_db, rng)
        meta["noise"] = str(noise_file)
    write_audio(out_path, wav, sr=16000)
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Build augmented speaker trials")
    parser.add_argument("--trials", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--musan-root", required=True)
    parser.add_argument("--rirs-root", required=True)
    parser.add_argument("--output-root", default="output/augmented_speaker_audio")
    parser.add_argument("--output-trials", default="output/augmented_speaker_trials.csv")
    parser.add_argument("--metadata", default="output/augmented_speaker_trials_meta.json")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--mode", choices=["noise", "rir", "rir_noise"], default="noise")
    parser.add_argument("--snr-db", type=float, default=0.0)
    parser.add_argument("--noise-kind", choices=["noise", "speech", "music", "all"], default="noise")
    parser.add_argument("--augment", choices=["test", "both"], default="test")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = read_trials(Path(args.trials), args.limit)
    audio_root = Path(args.audio_root)
    output_root = Path(args.output_root)
    musan_root = Path(args.musan_root)
    rirs_root = Path(args.rirs_root)

    if args.noise_kind == "all":
        noise_files = collect_wavs(musan_root)
    else:
        noise_files = collect_wavs(musan_root, args.noise_kind)
    rir_files = collect_wavs(rirs_root)
    if "noise" in args.mode and not noise_files:
        raise SystemExit("No MUSAN noise files found.")
    if "rir" in args.mode and not rir_files:
        raise SystemExit("No RIR files found.")

    out_rows = []
    meta_rows = []
    for i, row in enumerate(rows):
        enroll_rel = row["enroll_audio"]
        test_rel = row["test_audio"]
        label = row["label"]
        enroll_src = audio_root / enroll_rel
        test_src = audio_root / test_rel

        enroll_out_rel = enroll_rel
        test_out_rel = f"{args.mode}_snr{args.snr_db:g}/{i:06d}_test.wav"
        test_meta = augment_file(
            test_src,
            output_root / test_out_rel,
            args.mode,
            args.snr_db,
            noise_files,
            rir_files,
            rng,
        )
        if args.augment == "both":
            enroll_out_rel = f"{args.mode}_snr{args.snr_db:g}/{i:06d}_enroll.wav"
            enroll_meta = augment_file(
                enroll_src,
                output_root / enroll_out_rel,
                args.mode,
                args.snr_db,
                noise_files,
                rir_files,
                rng,
            )
        else:
            enroll_meta = {"mode": "clean", "snr_db": None, "noise": "", "rir": ""}

        out_rows.append(
            {
                "enroll_audio": enroll_out_rel if args.augment == "both" else str(enroll_src),
                "test_audio": test_out_rel,
                "label": label,
            }
        )
        meta_rows.append(
            {
                "index": i,
                "label": label,
                "source_enroll": str(enroll_src),
                "source_test": str(test_src),
                "output_enroll": out_rows[-1]["enroll_audio"],
                "output_test": out_rows[-1]["test_audio"],
                "enroll_aug": enroll_meta,
                "test_aug": test_meta,
            }
        )

    out_trials = Path(args.output_trials)
    out_trials.parent.mkdir(parents=True, exist_ok=True)
    with open(out_trials, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["enroll_audio", "test_audio", "label"])
        writer.writeheader()
        writer.writerows(out_rows)
    Path(args.metadata).write_text(json.dumps(meta_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(out_rows)} augmented trials -> {out_trials}")
    print(f"Audio root for augmented test files -> {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
