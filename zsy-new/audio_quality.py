"""Lightweight audio quality and VAD utilities for competition inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from audio_utils import peak_normalize, resample_mono


@dataclass
class AudioQuality:
    no_speech: bool = False
    speech_ratio: float = 1.0
    snr_db: float = 0.0
    duration_sec: float = 0.0
    effective_speech_sec: float = 0.0
    score: float = 1.0


def _frame_energies(waveform: np.ndarray, frame_len: int) -> np.ndarray:
    if waveform.size == 0:
        return np.asarray([], dtype=np.float32)
    n_frames = int(np.ceil(waveform.size / frame_len))
    vals = []
    for i in range(n_frames):
        frame = waveform[i * frame_len : (i + 1) * frame_len]
        if frame.size:
            vals.append(float(np.mean(frame * frame) + 1e-12))
    return np.asarray(vals, dtype=np.float32)


def estimate_snr_db(energies: np.ndarray, speech_mask: np.ndarray) -> float:
    if energies.size == 0:
        return 0.0
    if np.any(speech_mask):
        speech_energy = float(np.median(energies[speech_mask]))
    else:
        speech_energy = float(np.max(energies))
    if np.any(~speech_mask):
        noise_energy = float(np.median(energies[~speech_mask]))
    else:
        noise_energy = float(np.min(energies))
    return float(10.0 * np.log10(max(speech_energy, 1e-12) / max(noise_energy, 1e-12)))


def trim_by_energy_vad(
    waveform: np.ndarray,
    sr: int,
    frame_ms: int = 30,
    threshold_ratio: float = 0.10,
) -> tuple[np.ndarray, AudioQuality]:
    """Trim leading/trailing low-energy regions and return crude quality stats."""
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    duration_sec = waveform.size / float(sr)
    if waveform.size == 0:
        return waveform, AudioQuality(no_speech=True, duration_sec=0.0, score=0.0)

    frame_len = max(1, int(sr * frame_ms / 1000.0))
    energies = _frame_energies(waveform, frame_len)
    if energies.size == 0 or float(np.max(energies)) < 1e-10:
        return waveform, AudioQuality(no_speech=True, duration_sec=duration_sec, score=0.0)

    threshold = max(float(np.max(energies)) * threshold_ratio, 1e-8)
    speech_mask = energies > threshold
    if not bool(np.any(speech_mask)):
        return waveform, AudioQuality(no_speech=True, duration_sec=duration_sec, score=0.0)

    idx = np.where(speech_mask)[0]
    start = int(idx[0] * frame_len)
    end = int(min(waveform.size, (idx[-1] + 1) * frame_len))
    trimmed = waveform[start:end].astype(np.float32)

    speech_ratio = float(np.mean(speech_mask))
    effective_speech_sec = trimmed.size / float(sr)
    snr_db = estimate_snr_db(energies, speech_mask)
    duration_score = min(1.0, max(0.0, effective_speech_sec / 1.0))
    snr_score = min(1.0, max(0.0, (snr_db + 10.0) / 20.0))
    score = duration_score * snr_score
    return trimmed, AudioQuality(
        no_speech=False,
        speech_ratio=speech_ratio,
        snr_db=snr_db,
        duration_sec=duration_sec,
        effective_speech_sec=effective_speech_sec,
        score=score,
    )


def preprocess_waveform(
    waveform: np.ndarray,
    sr: int,
    target_sr: int = 16000,
    enable_vad: bool = True,
    vad_threshold_ratio: float = 0.10,
) -> tuple[np.ndarray, int, AudioQuality]:
    """Resample, peak-normalize, and optionally trim silence/noise edges."""
    wav = resample_mono(waveform, sr, target_sr)
    wav = peak_normalize(wav)
    if not enable_vad:
        quality = AudioQuality(
            no_speech=wav.size == 0,
            duration_sec=wav.size / float(target_sr),
            effective_speech_sec=wav.size / float(target_sr),
            score=0.0 if wav.size == 0 else 1.0,
        )
        return wav, target_sr, quality
    trimmed, quality = trim_by_energy_vad(wav, target_sr, threshold_ratio=vad_threshold_ratio)
    return trimmed, target_sr, quality
