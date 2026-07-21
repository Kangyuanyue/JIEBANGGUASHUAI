#!/usr/bin/env python3
"""Stage-2 Chinese adaptation for the positive/negative enrollment TSE model.

CN-Celeb2 speakers are split before sampling so validation identities never
appear in training.  DatasetA is deliberately not read by this script.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from scipy.signal import fftconvolve

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tse_model import PositiveNegativeTSE  # noqa: E402


SR = 16000


@lru_cache(maxsize=128)
def load_16k_cached(path_text: str) -> np.ndarray:
    wav, sr = sf.read(path_text, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if int(sr) != SR:
        wav = (
            torchaudio.functional.resample(torch.from_numpy(wav)[None], int(sr), SR)
            .squeeze(0)
            .numpy()
        )
    if wav.size:
        wav = wav - float(np.mean(wav))
    return wav.astype(np.float32)


def random_crop(wav: np.ndarray, length: int, rng: random.Random) -> np.ndarray:
    if wav.size == 0:
        return np.zeros(length, dtype=np.float32)
    if wav.size < length:
        wav = np.tile(wav, int(math.ceil(length / wav.size)))
    start = rng.randint(0, max(0, wav.size - length))
    return wav[start : start + length].astype(np.float32).copy()


def normalize_rms(wav: np.ndarray, target: float = 0.05) -> np.ndarray:
    value = float(np.sqrt(np.mean(np.asarray(wav, dtype=np.float64) ** 2) + 1e-12))
    return (wav * (target / max(value, 1e-6))).astype(np.float32)


def scale_at_ratio(signal: np.ndarray, ratio_db: float) -> np.ndarray:
    return signal * np.float32(10.0 ** (ratio_db / 20.0))


def peak_limit(*signals: np.ndarray, limit: float = 0.95) -> tuple[np.ndarray, ...]:
    peak = max(float(np.max(np.abs(x))) for x in signals)
    scale = min(1.0, limit / max(peak, 1e-6))
    return tuple((x * scale).astype(np.float32) for x in signals)


def eligible_speakers(root: Path) -> list[tuple[str, list[Path]]]:
    speakers = []
    for speaker_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(speaker_dir.glob("*.flac"))
        if len(files) >= 4:
            speakers.append((speaker_dir.name, files))
    return speakers


class AcousticAugmenter:
    def __init__(
        self,
        noise_root: str,
        rir_root: str,
        noise_probability: float,
        rir_probability: float,
        noise_snr_min: float,
        noise_snr_max: float,
    ) -> None:
        self.noise_paths = sorted(Path(noise_root).rglob("*.wav")) if noise_root else []
        self.rir_paths = sorted(Path(rir_root).rglob("*.wav")) if rir_root else []
        self.noise_probability = float(noise_probability)
        self.rir_probability = float(rir_probability)
        self.noise_snr_min = float(noise_snr_min)
        self.noise_snr_max = float(noise_snr_max)

    @staticmethod
    def _reverberate(wav: np.ndarray, rir: np.ndarray) -> np.ndarray:
        rir = np.asarray(rir, dtype=np.float32)
        peak = int(np.argmax(np.abs(rir)))
        rir = rir[peak : peak + min(rir.size - peak, SR)]
        rir = rir / max(float(np.sqrt(np.sum(rir * rir))), 1e-6)
        return fftconvolve(wav, rir, mode="full")[: wav.size].astype(np.float32)

    def apply(
        self,
        target: np.ndarray,
        interferer: np.ndarray,
        rng: random.Random,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        target_out = target.copy()
        interferer_out = interferer.copy()
        tags = []
        if self.rir_paths and rng.random() < self.rir_probability:
            rir = load_16k_cached(str(rng.choice(self.rir_paths)))
            target_out = self._reverberate(target_out, rir)
            interferer_out = self._reverberate(interferer_out, rir)
            tags.append("rir")

        mixture = target_out + interferer_out
        if self.noise_paths and rng.random() < self.noise_probability:
            noise = random_crop(
                load_16k_cached(str(rng.choice(self.noise_paths))), mixture.size, rng
            )
            noise = normalize_rms(noise)
            snr_db = rng.uniform(self.noise_snr_min, self.noise_snr_max)
            mixture_rms = float(np.sqrt(np.mean(mixture * mixture) + 1e-12))
            noise_rms = float(np.sqrt(np.mean(noise * noise) + 1e-12))
            noise_scale = mixture_rms / max(noise_rms * 10.0 ** (snr_db / 20.0), 1e-6)
            mixture = mixture + noise * np.float32(noise_scale)
            tags.append(f"noise_{snr_db:.1f}db")
        return target_out, interferer_out, mixture.astype(np.float32), "+".join(tags) or "clean"


def make_case(
    speakers: list[tuple[str, list[Path]]],
    length: int,
    rng: random.Random,
    augmenter: AcousticAugmenter | None = None,
) -> dict[str, np.ndarray | float | str]:
    target_info, interferer_info = rng.sample(speakers, 2)
    target_id, target_files = target_info
    interferer_id, interferer_files = interferer_info
    target_paths = rng.sample(target_files, 2)
    interferer_paths = rng.sample(interferer_files, 3)

    target_enroll = normalize_rms(random_crop(load_16k_cached(str(target_paths[0])), length, rng))
    target_query = normalize_rms(random_crop(load_16k_cached(str(target_paths[1])), length, rng))
    enroll_interferer = normalize_rms(
        random_crop(load_16k_cached(str(interferer_paths[0])), length, rng)
    )
    negative_interferer = normalize_rms(
        random_crop(load_16k_cached(str(interferer_paths[1])), length, rng)
    )
    query_interferer = normalize_rms(
        random_crop(load_16k_cached(str(interferer_paths[2])), length, rng)
    )

    mode_roll = rng.random()
    enroll_tir_db = rng.uniform(-8.0, 8.0)
    if mode_roll < 0.55:
        mode = "hybrid_partial"
        partial = np.zeros(length, dtype=np.float32)
        partial_len = rng.randint(max(1, length // 4), max(2, 3 * length // 4))
        start = rng.randint(0, length - partial_len)
        partial[start : start + partial_len] = enroll_interferer[start : start + partial_len]
        positive = scale_at_ratio(target_enroll, enroll_tir_db) + partial
        negative = negative_interferer
    elif mode_roll < 0.85:
        mode = "full_overlap"
        positive = scale_at_ratio(target_enroll, enroll_tir_db) + enroll_interferer
        negative = negative_interferer
    else:
        mode = "clean_pseudo"
        noise = np.asarray(
            [rng.gauss(0.0, 0.0015) for _ in range(length)], dtype=np.float32
        )
        positive = target_enroll + noise
        negative = noise

    # More mass is assigned to adverse TIRs because these dominate the observed
    # competition errors, while easier cases prevent over-specialization.
    query_tir_db = rng.uniform(-12.0, 6.0)
    target_scaled = scale_at_ratio(target_query, query_tir_db)
    if augmenter is None:
        mixture = target_scaled + query_interferer
        acoustic_condition = "clean"
    else:
        target_scaled, query_interferer, mixture, acoustic_condition = augmenter.apply(
            target_scaled, query_interferer, rng
        )
    mixture, target_scaled, positive, negative = peak_limit(
        mixture, target_scaled, positive, negative
    )
    return {
        "mixture": mixture,
        "target": target_scaled,
        "positive": positive,
        "negative": negative,
        "target_id": target_id,
        "interferer_id": interferer_id,
        "query_tir_db": query_tir_db,
        "mode": mode,
        "acoustic_condition": acoustic_condition,
    }


def batch_to_device(case: dict, device: torch.device) -> tuple[torch.Tensor, ...]:
    values = []
    for key in ("mixture", "target", "positive", "negative"):
        tensor = torch.from_numpy(case[key]).to(device=device, dtype=torch.float32)
        values.append(tensor[None, None, :])
    return tuple(values)


def si_snr_db(estimate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    estimate = estimate.reshape(estimate.shape[0], -1)
    reference = reference.reshape(reference.shape[0], -1)
    n = min(estimate.shape[-1], reference.shape[-1])
    estimate = estimate[:, :n] - estimate[:, :n].mean(dim=-1, keepdim=True)
    reference = reference[:, :n] - reference[:, :n].mean(dim=-1, keepdim=True)
    projection = (
        torch.sum(estimate * reference, dim=-1, keepdim=True)
        * reference
        / (torch.sum(reference * reference, dim=-1, keepdim=True) + 1e-8)
    )
    noise = estimate - projection
    return 10.0 * torch.log10(
        (torch.sum(projection * projection, dim=-1) + 1e-8)
        / (torch.sum(noise * noise, dim=-1) + 1e-8)
    )


@torch.no_grad()
def validate(model, cases: list[dict], device: torch.device) -> dict[str, float]:
    model.eval()
    input_scores = []
    output_scores = []
    for case in cases:
        mixture, target, positive, negative = batch_to_device(case, device)
        cond_emb, _, _ = model.encoder_pos_neg(positive, negative, recons=False)
        estimate = model(mixture, cond_emb)
        input_scores.append(float(si_snr_db(mixture, target).item()))
        output_scores.append(float(si_snr_db(estimate, target).item()))
    improvements = np.asarray(output_scores) - np.asarray(input_scores)
    return {
        "input_si_snr_db": float(np.mean(input_scores)),
        "output_si_snr_db": float(np.mean(output_scores)),
        "si_snri_db": float(np.mean(improvements)),
        "median_si_snri_db": float(np.median(improvements)),
        "improved_ratio": float(np.mean(improvements > 0.0)),
    }


def save_checkpoint(path: Path, model, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "adaptation": metadata}, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="CN-Celeb2_flac/data")
    parser.add_argument("--base-checkpoint", default=None)
    parser.add_argument("--output", default="pretrained/tse_posneg_cnceleb_stage2.pt")
    parser.add_argument("--history", default="output/tse_cnceleb_training_history.json")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--duration-sec", type=float, default=2.0)
    parser.add_argument("--train-speakers", type=int, default=800)
    parser.add_argument("--val-speakers", type=int, default=100)
    parser.add_argument("--val-cases", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--validate-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--noise-root", default="")
    parser.add_argument("--rir-root", default="")
    parser.add_argument("--noise-probability", type=float, default=0.0)
    parser.add_argument("--rir-probability", type=float, default=0.0)
    parser.add_argument("--noise-snr-min", type=float, default=-5.0)
    parser.add_argument("--noise-snr-max", type=float, default=15.0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    all_speakers = eligible_speakers(Path(args.data_root))
    required = args.train_speakers + args.val_speakers
    if len(all_speakers) < required:
        raise RuntimeError(f"Need {required} eligible speakers, found {len(all_speakers)}")
    train_speakers = all_speakers[: args.train_speakers]
    val_speakers = all_speakers[args.train_speakers : required]
    length = int(round(args.duration_sec * SR))

    augmenter = AcousticAugmenter(
        noise_root=args.noise_root,
        rir_root=args.rir_root,
        noise_probability=args.noise_probability,
        rir_probability=args.rir_probability,
        noise_snr_min=args.noise_snr_min,
        noise_snr_max=args.noise_snr_max,
    )
    if not augmenter.noise_paths and not augmenter.rir_paths:
        augmenter = None

    train_rng = random.Random(args.seed)
    val_rng = random.Random(args.seed + 1)
    val_cases = [
        make_case(val_speakers, length, val_rng, augmenter=augmenter)
        for _ in range(args.val_cases)
    ]

    wrapper_kwargs = {"device": args.device}
    if args.base_checkpoint:
        wrapper_kwargs["checkpoint"] = args.base_checkpoint
    wrapper = PositiveNegativeTSE(**wrapper_kwargs)
    model = wrapper.model
    device = wrapper.device

    for param in model.parameters():
        param.requires_grad_(False)
    main_params = list(model.main_params())
    for param in main_params:
        param.requires_grad_(True)
    optimizer = torch.optim.AdamW(main_params, lr=args.learning_rate, weight_decay=1e-5)
    optimizer.zero_grad(set_to_none=True)

    history = []
    baseline = validate(model, val_cases, device)
    baseline_record = {"step": 0, "train_loss": None, **baseline}
    history.append(baseline_record)
    print(json.dumps(baseline_record, ensure_ascii=False))
    best_si_snri = baseline["si_snri_db"]
    best_step = 0
    started = time.perf_counter()
    running_loss = 0.0

    for step in range(1, args.steps + 1):
        model.train()
        model.siamese.eval()
        model.encoder_head.eval()
        case = make_case(train_speakers, length, train_rng, augmenter=augmenter)
        mixture, target, positive, negative = batch_to_device(case, device)
        with torch.no_grad():
            cond_emb, _, _ = model.encoder_pos_neg(positive, negative, recons=False)
        estimate = model(mixture, cond_emb)
        loss = -si_snr_db(estimate, target).mean() / args.grad_accum
        loss.backward()
        running_loss += float(loss.item()) * args.grad_accum

        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(main_params, max_norm=5.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        should_validate = step % args.validate_every == 0 or step == args.steps
        if should_validate:
            metrics = validate(model, val_cases, device)
            window = args.validate_every if step > args.validate_every else step
            record = {
                "step": step,
                "train_loss": running_loss / max(1, window),
                "elapsed_sec": time.perf_counter() - started,
                **metrics,
            }
            history.append(record)
            running_loss = 0.0
            print(json.dumps(record, ensure_ascii=False))
            if metrics["si_snri_db"] > best_si_snri:
                best_si_snri = metrics["si_snri_db"]
                best_step = step
                save_checkpoint(
                    Path(args.output),
                    model,
                    {"best_step": best_step, "best_validation": metrics, "args": vars(args)},
                )

        if args.save_every > 0 and step % args.save_every == 0:
            periodic = Path(args.output).with_name(f"{Path(args.output).stem}_step{step}.pt")
            save_checkpoint(periodic, model, {"step": step, "args": vars(args)})

    summary = {
        "base_checkpoint": str(wrapper.checkpoint),
        "output_checkpoint": str(Path(args.output).resolve()),
        "dataset": str(Path(args.data_root).resolve()),
        "datasetA_used_for_training": False,
        "train_speaker_range": [train_speakers[0][0], train_speakers[-1][0]],
        "validation_speaker_range": [val_speakers[0][0], val_speakers[-1][0]],
        "train_speakers": len(train_speakers),
        "validation_speakers": len(val_speakers),
        "trainable_parameters": int(sum(p.numel() for p in main_params)),
        "augmentation": {
            "noise_files": len(augmenter.noise_paths) if augmenter else 0,
            "rir_files": len(augmenter.rir_paths) if augmenter else 0,
            "noise_probability": args.noise_probability,
            "rir_probability": args.rir_probability,
            "noise_snr_db": [args.noise_snr_min, args.noise_snr_max],
        },
        "best_step": best_step,
        "best_validation_si_snri_db": best_si_snri,
        "history": history,
    }
    history_path = Path(args.history)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
