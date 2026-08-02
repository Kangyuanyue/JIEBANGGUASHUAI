#!/usr/bin/env python3
"""Audio loading, active-region cropping, level control, and deterministic mixing."""

from __future__ import annotations

import math
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np


EPSILON = 1e-12


def load_audio(path: Path, sample_rate: int, ffmpeg: str = "ffmpeg") -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    if shutil.which(ffmpeg) is None:
        raise RuntimeError(f"ffmpeg executable not found: {ffmpeg}")
    command = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "pipe:1",
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    audio = np.frombuffer(result.stdout, dtype="<f4").astype(np.float32, copy=True)
    if audio.size == 0:
        raise ValueError(f"Decoded audio is empty: {path}")
    if not np.all(np.isfinite(audio)):
        raise ValueError(f"Decoded audio contains non-finite values: {path}")
    return audio


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm = np.round(clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm.tobytes())


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)) + EPSILON))


def db_ratio(reference: np.ndarray, other: np.ndarray) -> float | None:
    reference_rms = rms(reference)
    other_rms = rms(other)
    if reference_rms <= 1e-8 or other_rms <= 1e-8:
        return None
    return 20.0 * math.log10(reference_rms / other_rms)


def scale_to_rms_dbfs(audio: np.ndarray, target_dbfs: float) -> np.ndarray:
    current = rms(audio)
    if current <= 1e-8:
        raise ValueError("Cannot level-normalize silent audio")
    target = 10.0 ** (target_dbfs / 20.0)
    return (audio * (target / current)).astype(np.float32)


def scale_for_ratio(reference: np.ndarray, other: np.ndarray, ratio_db: float) -> np.ndarray:
    reference_rms = rms(reference)
    other_rms = rms(other)
    if reference_rms <= 1e-8 or other_rms <= 1e-8:
        raise ValueError("Cannot set a level ratio for silent audio")
    desired_other_rms = reference_rms / (10.0 ** (ratio_db / 20.0))
    return (other * (desired_other_rms / other_rms)).astype(np.float32)


def active_crop(
    audio: np.ndarray,
    sample_rate: int,
    frame_ms: int = 20,
    threshold_below_peak_db: float = 35.0,
    padding_ms: int = 100,
) -> np.ndarray:
    """Crop leading/trailing silence using a simple frame-energy detector."""
    frame = max(1, int(sample_rate * frame_ms / 1000))
    frame_count = int(math.ceil(audio.size / frame))
    padded = np.pad(audio, (0, frame_count * frame - audio.size))
    frame_rms = np.sqrt(np.mean(padded.reshape(frame_count, frame) ** 2, axis=1) + EPSILON)
    peak = float(np.max(frame_rms))
    if peak <= 1e-7:
        raise ValueError("Audio is effectively silent")
    threshold = max(peak * (10.0 ** (-threshold_below_peak_db / 20.0)), 1e-5)
    active = np.flatnonzero(frame_rms >= threshold)
    if active.size == 0:
        raise ValueError("No active audio region detected")
    pad = int(sample_rate * padding_ms / 1000)
    start = max(0, int(active[0]) * frame - pad)
    end = min(audio.size, (int(active[-1]) + 1) * frame + pad)
    return audio[start:end].astype(np.float32, copy=True)


def match_noise(noise: np.ndarray, length: int, start_fraction: float) -> np.ndarray:
    if noise.size == 0:
        raise ValueError("Noise audio is empty")
    if noise.size < length:
        repeats = int(math.ceil(length / noise.size))
        noise = np.tile(noise, repeats)
    maximum_start = max(0, noise.size - length)
    start = int(round(maximum_start * min(max(start_fraction, 0.0), 1.0)))
    return noise[start : start + length].astype(np.float32, copy=True)


def aligned_track(audio: np.ndarray, offset: int, total_length: int) -> np.ndarray:
    output = np.zeros(total_length, dtype=np.float32)
    if offset >= total_length:
        return output
    copy_length = min(audio.size, total_length - offset)
    if copy_length > 0:
        output[offset : offset + copy_length] = audio[:copy_length]
    return output


def overlap_offset(
    primary_length: int,
    secondary_length: int,
    overlap_ratio: float,
    primary_offset: int,
    sample_rate: int,
    max_output_seconds: float,
    trailing_padding_seconds: float,
    alignment_mode: str = "overlap_ratio",
) -> int:
    if alignment_mode == "simultaneous":
        return primary_offset
    if alignment_mode != "overlap_ratio":
        raise ValueError(f"Unsupported alignment mode: {alignment_mode}")
    overlap_samples = int(round(min(primary_length, secondary_length) * overlap_ratio))
    secondary_offset = primary_offset + primary_length - overlap_samples
    maximum_end = int(round((max_output_seconds - trailing_padding_seconds) * sample_rate))
    overflow = secondary_offset + secondary_length - maximum_end
    if overflow > 0:
        secondary_offset = max(0, secondary_offset - overflow)
    return secondary_offset


def measured_overlap_ratio(first: np.ndarray, second: np.ndarray) -> float | None:
    first_active = np.abs(first) > 1e-7
    second_active = np.abs(second) > 1e-7
    first_indices = np.flatnonzero(first_active)
    second_indices = np.flatnonzero(second_active)
    if first_indices.size == 0 or second_indices.size == 0:
        return None
    first_start, first_end = int(first_indices[0]), int(first_indices[-1]) + 1
    second_start, second_end = int(second_indices[0]), int(second_indices[-1]) + 1
    denominator = min(first_end - first_start, second_end - second_start)
    overlap = max(0, min(first_end, second_end) - max(first_start, second_start))
    return float(overlap / denominator)


def apply_peak_ceiling(tracks: list[np.ndarray], ceiling_dbfs: float) -> tuple[list[np.ndarray], float]:
    if not tracks:
        raise ValueError("No tracks supplied")
    mixture = np.sum(np.stack(tracks), axis=0, dtype=np.float64).astype(np.float32)
    peak = float(np.max(np.abs(mixture))) if mixture.size else 0.0
    ceiling = 10.0 ** (ceiling_dbfs / 20.0)
    scale = ceiling / peak if peak > ceiling and peak > 0 else 1.0
    return [(track * scale).astype(np.float32) for track in tracks], float(scale)
