"""
Audio decoding and preprocessing for the competition pipeline.

Self-contained copy of the essential logic from api_server.py — no FastAPI dependency.
"""

from __future__ import annotations

import io
import os
import subprocess
import tempfile
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio


SUPPORTED_EXTS = {"wav", "mp3", "flac", "m4a", "ogg", "opus", "aac", "webm"}


def guess_format(filename: Optional[str], content_type: Optional[str] = None) -> Optional[str]:
    if filename and "." in filename:
        ext = filename.rsplit(".", 1)[-1].lower().strip()
        if ext in SUPPORTED_EXTS:
            return ext
    if content_type:
        ct = content_type.lower().strip()
        mapping = {
            "audio/mpeg": "mp3",
            "audio/mp3": "mp3",
            "audio/wav": "wav",
            "audio/x-wav": "wav",
            "audio/flac": "flac",
            "audio/mp4": "m4a",
            "audio/x-m4a": "m4a",
            "audio/m4a": "m4a",
            "audio/ogg": "ogg",
            "audio/opus": "ogg",
            "audio/webm": "webm",
            "video/webm": "webm",
        }
        return mapping.get(ct)
    return None


def load_audio_bytes(
    file_bytes: bytes,
    filename: Optional[str] = None,
    content_type: Optional[str] = None,
) -> Tuple[np.ndarray, int]:
    """Decode arbitrary audio bytes → mono float32 waveform + sample rate."""
    fmt = guess_format(filename, content_type)

    try:
        data, sr = sf.read(io.BytesIO(file_bytes), dtype="float32", always_2d=False)
        if isinstance(data, np.ndarray) and data.ndim == 2:
            if data.shape[0] < data.shape[1]:
                data = data.T
            data = np.mean(data, axis=1)
        return data.astype(np.float32), int(sr)
    except Exception:
        pass

    tmp_path = None
    librosa_error = ""
    try:
        import librosa

        suffix = f".{fmt}" if fmt else ".audio"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(file_bytes)
            tmp_path = f.name
        y, sr = librosa.load(tmp_path, sr=None, mono=True)
        return y.astype(np.float32), int(sr)
    except Exception as e:
        librosa_error = repr(e)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    try:
        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        in_suffix = f".{fmt}" if fmt else ".audio"
        in_path = out_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=in_suffix) as f:
                f.write(file_bytes)
                in_path = f.name
            out_fd, out_path = tempfile.mkstemp(suffix=".wav")
            os.close(out_fd)
            cmd = [ffmpeg_exe, "-y", "-i", in_path, "-ac", "1", "-ar", "16000", "-f", "wav", out_path]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            data, sr = sf.read(out_path, dtype="float32", always_2d=False)
            if isinstance(data, np.ndarray) and data.ndim == 2:
                data = np.mean(data, axis=1)
            return data.astype(np.float32), int(sr)
        finally:
            for p in (in_path, out_path):
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
    except Exception as e:
        raise RuntimeError(
            f"Failed to decode audio ({filename or 'bytes'}). "
            f"librosa: {librosa_error}; ffmpeg: {repr(e)}"
        ) from e

    raise RuntimeError(f"Failed to decode audio: {librosa_error}")


def load_audio_file(path: str) -> Tuple[np.ndarray, int]:
    with open(path, "rb") as f:
        return load_audio_bytes(f.read(), filename=os.path.basename(path))


def resample_mono(waveform_np: np.ndarray, sr: int, target_sr: int = 16000) -> np.ndarray:
    if waveform_np.ndim != 1:
        waveform_np = np.asarray(waveform_np).reshape(-1).astype(np.float32)
    if sr == target_sr:
        return waveform_np.astype(np.float32)
    wav = torch.from_numpy(waveform_np.astype(np.float32))
    out = torchaudio.functional.resample(wav.unsqueeze(0), sr, target_sr).squeeze(0)
    return out.numpy().astype(np.float32)


def build_segments(
    waveform: np.ndarray,
    sr: int,
    segment_sec: float = 3.0,
    num_segments: int = 5,
) -> list[np.ndarray]:
    """
    Split waveform into `num_segments` evenly-spaced fixed-length chunks.
    Short audio is repeat-padded (same strategy as VoiceDetective eval).
    """
    target_len = int(sr * segment_sec)
    if waveform.size <= 0:
        raise ValueError("Empty waveform.")

    if waveform.size <= target_len:
        repeat = int(np.ceil(target_len / waveform.size))
        padded = np.tile(waveform, repeat)[:target_len]
        return [padded.copy() for _ in range(max(1, num_segments))]

    max_start = waveform.size - target_len
    if num_segments <= 1:
        starts = [max_start // 2]
    else:
        starts = np.linspace(0, max_start, num=num_segments).astype(int).tolist()

    return [waveform[s : s + target_len].copy() for s in starts]


def peak_normalize(waveform: np.ndarray) -> np.ndarray:
    """Match speaker_model.py preprocessing."""
    w = waveform.astype(np.float32).copy()
    w = w - np.mean(w)
    peak = max(float(np.max(np.abs(w))), 1e-6)
    return (w / peak).astype(np.float32)
