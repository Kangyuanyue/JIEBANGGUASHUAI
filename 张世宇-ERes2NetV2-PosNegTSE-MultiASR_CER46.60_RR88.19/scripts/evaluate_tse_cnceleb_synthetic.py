#!/usr/bin/env python3
"""Evaluate Pos/Neg TSE on deterministic Chinese two-speaker mixtures."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from speaker_model import cosine_similarity, get_speaker_backend  # noqa: E402
from tse_model import PositiveNegativeTSE  # noqa: E402


SR = 16000


def load_16k(path: Path) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if int(sr) != SR:
        wav = (
            torchaudio.functional.resample(torch.from_numpy(wav)[None], int(sr), SR)
            .squeeze(0)
            .numpy()
        )
    wav = wav - float(np.mean(wav))
    return wav.astype(np.float32)


def repeat_crop(wav: np.ndarray, length: int, offset_ratio: float = 0.0) -> np.ndarray:
    if wav.size < length:
        wav = np.tile(wav, int(np.ceil(length / max(1, wav.size))))
    max_start = max(0, wav.size - length)
    start = int(round(max_start * min(1.0, max(0.0, offset_ratio))))
    return wav[start : start + length].astype(np.float32).copy()


def rms(wav: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(wav, dtype=np.float64) ** 2) + 1e-12))


def normalize_rms(wav: np.ndarray, target: float = 0.05) -> np.ndarray:
    return (wav * (target / max(rms(wav), 1e-6))).astype(np.float32)


def mix_at_tir(target: np.ndarray, interferer: np.ndarray, tir_db: float) -> np.ndarray:
    target = normalize_rms(target)
    interferer = normalize_rms(interferer)
    target_scale = 10.0 ** (tir_db / 20.0)
    return (target_scale * target + interferer).astype(np.float32)


def si_snr(reference: np.ndarray, estimate: np.ndarray) -> float:
    n = min(reference.size, estimate.size)
    ref = reference[:n].astype(np.float64)
    est = estimate[:n].astype(np.float64)
    ref -= np.mean(ref)
    est -= np.mean(est)
    target = np.dot(est, ref) * ref / max(np.dot(ref, ref), 1e-12)
    noise = est - target
    return float(10.0 * np.log10((np.dot(target, target) + 1e-12) / (np.dot(noise, noise) + 1e-12)))


def eligible_speakers(
    root: Path, speaker_start: int = 0, speaker_count: int = 300
) -> list[tuple[str, list[Path]]]:
    speakers = []
    directories = sorted(p for p in root.iterdir() if p.is_dir())
    selected = directories[speaker_start : speaker_start + speaker_count]
    for speaker_dir in selected:
        files = sorted(speaker_dir.glob("*.flac"))
        if len(files) >= 4:
            speakers.append((speaker_dir.name, files))
    return speakers


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="CN-Celeb2_flac/data")
    parser.add_argument("--cases", type=int, default=30)
    parser.add_argument("--duration-sec", type=float, default=3.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--speaker-start", type=int, default=0)
    parser.add_argument("--speaker-count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument(
        "--enrollment-mode",
        choices=("hybrid_partial", "negative_interferer_full", "clean_pseudo"),
        default="hybrid_partial",
    )
    parser.add_argument("--save-audio", type=int, default=4)
    parser.add_argument("--audio-dir", default="output/tse_cnceleb_examples")
    parser.add_argument("--output", default="output/tse_cnceleb_synthetic_eval.json")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    data_root = Path(args.data_root)
    speakers = eligible_speakers(data_root, args.speaker_start, args.speaker_count)
    if len(speakers) < 2:
        raise RuntimeError(f"Not enough speakers under {data_root}")

    length = int(round(args.duration_sec * SR))
    tse_kwargs = {"device": args.device}
    if args.checkpoint:
        tse_kwargs["checkpoint"] = args.checkpoint
    tse = PositiveNegativeTSE(**tse_kwargs)
    sv = get_speaker_backend("eres2netv2", local_model_dir="pretrained")
    audio_dir = Path(args.audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for idx in range(args.cases):
        target_info, interferer_info = rng.sample(speakers, 2)
        target_id, target_files = target_info
        interferer_id, interferer_files = interferer_info
        target_paths = rng.sample(target_files, 2)
        interferer_paths = rng.sample(interferer_files, 3)

        target_enroll = normalize_rms(repeat_crop(load_16k(target_paths[0]), length, 0.0))
        target_query = normalize_rms(repeat_crop(load_16k(target_paths[1]), length, 0.35))
        pos_interferer = normalize_rms(repeat_crop(load_16k(interferer_paths[0]), length, 0.0))
        neg_interferer = normalize_rms(repeat_crop(load_16k(interferer_paths[1]), length, 0.2))
        query_interferer = normalize_rms(repeat_crop(load_16k(interferer_paths[2]), length, 0.45))

        if args.enrollment_mode == "hybrid_partial":
            partial = np.zeros_like(pos_interferer)
            start = length // 4
            end = start + length // 2
            partial[start:end] = pos_interferer[start:end]
            positive = target_enroll + partial
            negative = neg_interferer
        elif args.enrollment_mode == "negative_interferer_full":
            positive = target_enroll + pos_interferer
            negative = neg_interferer
        else:
            rng_np = np.random.default_rng(args.seed + idx)
            pseudo_noise = rng_np.normal(0.0, 0.0025, size=length).astype(np.float32)
            positive = target_enroll + pseudo_noise
            negative = pseudo_noise
        tir_db = (-10.0, -5.0, 0.0, 5.0)[idx % 4]
        mixture = mix_at_tir(target_query, query_interferer, tir_db)

        output = tse.extract(mixture, SR, positive, SR, negative, SR)
        input_si_snr = si_snr(target_query, mixture)
        output_si_snr = si_snr(target_query, output.waveform)

        target_emb = sv.encode(target_query, SR)
        interferer_emb = sv.encode(query_interferer, SR)
        mix_emb = sv.encode(mixture, SR)
        out_emb = sv.encode(output.waveform, SR)
        mix_target_similarity = cosine_similarity(target_emb, mix_emb)
        output_target_similarity = cosine_similarity(target_emb, out_emb)
        output_interferer_similarity = cosine_similarity(interferer_emb, out_emb)

        item = {
            "case": idx,
            "target_id": target_id,
            "interferer_id": interferer_id,
            "tir_db": tir_db,
            "input_si_snr_db": input_si_snr,
            "output_si_snr_db": output_si_snr,
            "si_snri_db": output_si_snr - input_si_snr,
            "mix_target_similarity": mix_target_similarity,
            "output_target_similarity": output_target_similarity,
            "output_interferer_similarity": output_interferer_similarity,
            "identity_correct": output_target_similarity > output_interferer_similarity,
            "target_similarity_gain": output_target_similarity - mix_target_similarity,
            "elapsed_sec": output.elapsed_sec,
        }
        results.append(item)
        print(json.dumps(item, ensure_ascii=False))

        if idx < args.save_audio:
            sf.write(audio_dir / f"case{idx:03d}_mix.wav", mixture, SR)
            sf.write(audio_dir / f"case{idx:03d}_target.wav", target_query, SR)
            sf.write(audio_dir / f"case{idx:03d}_interferer.wav", query_interferer, SR)
            sf.write(audio_dir / f"case{idx:03d}_positive.wav", positive, SR)
            sf.write(audio_dir / f"case{idx:03d}_negative.wav", negative, SR)
            sf.write(audio_dir / f"case{idx:03d}_output.wav", output.waveform, SR)

    by_tir = {}
    for tir in sorted({r["tir_db"] for r in results}):
        group = [r for r in results if r["tir_db"] == tir]
        by_tir[str(tir)] = {
            "n": len(group),
            "mean_si_snri_db": float(np.mean([r["si_snri_db"] for r in group])),
            "identity_accuracy": float(np.mean([r["identity_correct"] for r in group])),
            "mean_target_similarity_gain": float(
                np.mean([r["target_similarity_gain"] for r in group])
            ),
        }

    summary = {
        "dataset": str(data_root.resolve()),
        "checkpoint": str(tse.checkpoint),
        "speaker_start": args.speaker_start,
        "speaker_count": args.speaker_count,
        "n_cases": len(results),
        "duration_sec": args.duration_sec,
        "enrollment_mode": args.enrollment_mode,
        "mean_si_snri_db": float(np.mean([r["si_snri_db"] for r in results])),
        "median_si_snri_db": float(np.median([r["si_snri_db"] for r in results])),
        "improved_ratio": float(np.mean([r["si_snri_db"] > 0.0 for r in results])),
        "identity_accuracy": float(np.mean([r["identity_correct"] for r in results])),
        "mean_target_similarity_gain": float(
            np.mean([r["target_similarity_gain"] for r in results])
        ),
        "mean_elapsed_sec": float(np.mean([r["elapsed_sec"] for r in results])),
        "by_tir": by_tir,
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
