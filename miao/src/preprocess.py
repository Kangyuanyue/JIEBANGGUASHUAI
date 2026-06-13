"""Audio preprocessing helpers.

This file provides minimal, safe defaults. Real submission should replace or
extend VAD/SNR functions with Silero VAD, FunASR VAD, or the selected model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

try:
    import soundfile as sf
except Exception:  # pragma: no cover
    sf = None


@dataclass
class AudioQuality:
    """Lightweight quality summary used by the decision layer."""

    no_speech: bool = False
    speech_ratio: float = 1.0
    snr_db: float = 0.0
    duration_sec: float = 0.0
    effective_speech_sec: float = 0.0
    score: float = 1.0


def load_audio(path: str | Path, target_sample_rate: int = 16000, mono: bool = True) -> Tuple[np.ndarray, int]:
    """Load a wav file.

    Resampling is intentionally not implemented to avoid hidden dependencies.
    If your data is not 16 kHz, add librosa/torchaudio based resampling here.
    """
    if sf is None:
        raise RuntimeError("soundfile is required. Install it with `pip install soundfile`.")

    audio, sr = sf.read(str(path), always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim == 2 and mono:
        audio = audio.mean(axis=1)

    if sr != target_sample_rate:
        raise ValueError(
            f"Expected {target_sample_rate} Hz audio, got {sr} Hz for {path}. "
            "Please add resampling in preprocess.load_audio."
        )

    return audio, sr


def peak_normalize(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    """Peak normalize audio to a target absolute amplitude."""
    if audio.size == 0:
        return audio
    max_abs = float(np.max(np.abs(audio)))
    if max_abs < 1e-8:
        return audio
    return (audio / max_abs * peak).astype(np.float32)


def simple_energy_vad(audio: np.ndarray, sr: int, frame_ms: int = 30, threshold_ratio: float = 0.10) -> Tuple[np.ndarray, AudioQuality]:
    """Very simple energy VAD used only as a placeholder.

    Real system should replace this with Silero/FunASR VAD. This function trims
    leading/trailing low-energy frames and produces a crude quality summary.
    """
    if audio.size == 0:
        return audio, AudioQuality(no_speech=True, duration_sec=0.0, effective_speech_sec=0.0, score=0.0)

    frame_len = max(1, int(sr * frame_ms / 1000))
    n_frames = int(np.ceil(len(audio) / frame_len))
    energies = []
    for i in range(n_frames):
        frame = audio[i * frame_len : (i + 1) * frame_len]
        if frame.size == 0:
            continue
        energies.append(float(np.mean(frame ** 2)))

    if not energies or max(energies) < 1e-10:
        return audio, AudioQuality(no_speech=True, duration_sec=len(audio) / sr, effective_speech_sec=0.0, score=0.0)

    energies_arr = np.asarray(energies, dtype=np.float32)
    threshold = max(float(np.max(energies_arr)) * threshold_ratio, 1e-8)
    speech_mask = energies_arr > threshold

    if not bool(np.any(speech_mask)):
        return audio, AudioQuality(no_speech=True, duration_sec=len(audio) / sr, effective_speech_sec=0.0, score=0.0)

    idx = np.where(speech_mask)[0]
    start = int(idx[0] * frame_len)
    end = int(min(len(audio), (idx[-1] + 1) * frame_len))
    trimmed = audio[start:end]

    speech_ratio = float(np.mean(speech_mask))
    duration_sec = len(audio) / sr
    effective_speech_sec = len(trimmed) / sr
    snr_db = estimate_snr_db(audio, speech_mask, frame_len)
    quality_score = min(1.0, max(0.0, effective_speech_sec / 1.0)) * min(1.0, max(0.0, (snr_db + 10.0) / 20.0))

    return trimmed.astype(np.float32), AudioQuality(
        no_speech=False,
        speech_ratio=speech_ratio,
        snr_db=snr_db,
        duration_sec=duration_sec,
        effective_speech_sec=effective_speech_sec,
        score=quality_score,
    )


def estimate_snr_db(audio: np.ndarray, speech_mask: np.ndarray, frame_len: int) -> float:
    """Crude frame-energy SNR estimate."""
    energies = []
    for i in range(len(speech_mask)):
        frame = audio[i * frame_len : (i + 1) * frame_len]
        if frame.size:
            energies.append(float(np.mean(frame ** 2) + 1e-12))
    energies_arr = np.asarray(energies, dtype=np.float32)
    speech_energy = float(np.median(energies_arr[speech_mask])) if np.any(speech_mask) else 1e-12
    noise_energy = float(np.median(energies_arr[~speech_mask])) if np.any(~speech_mask) else float(np.min(energies_arr))
    return float(10.0 * np.log10(max(speech_energy, 1e-12) / max(noise_energy, 1e-12)))


def preprocess_audio(audio: np.ndarray, sr: int, normalize_peak: float = 0.95) -> Tuple[np.ndarray, AudioQuality]:
    """Apply peak normalization and placeholder VAD."""
    audio = peak_normalize(audio, peak=normalize_peak)
    return simple_energy_vad(audio, sr)
