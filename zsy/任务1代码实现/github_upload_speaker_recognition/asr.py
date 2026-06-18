"""
ASR backends for Chinese smart-home command recognition.

Primary: FunASR Paraformer (pip install funasr).
Fallback: mock backend for pipeline testing without GPU / model download.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from audio_utils import resample_mono
from config import AsrConfig


_PUNCT_RE = re.compile(
    r"[\s，。！？、；：""''（）【】《》…—·,.!?;:\"'()\[\]<>]+"
)


def normalize_asr_text(text: str, strip_punctuation: bool = True) -> str:
    t = (text or "").strip()
    if strip_punctuation:
        t = _PUNCT_RE.sub("", t)
    return t.strip()


class AsrBackend(ABC):
    @abstractmethod
    def transcribe(self, waveform: np.ndarray, sr: int) -> str:
        ...


class MockAsrBackend(AsrBackend):
    """Returns empty string — use only to test gate + submission format."""

    def transcribe(self, waveform: np.ndarray, sr: int) -> str:
        return ""


class FunAsrBackend(AsrBackend):
    """FunASR Paraformer wrapper."""

    def __init__(self, config: AsrConfig):
        self.config = config
        self._model = None
        self._device = self._resolve_device(config.device)

    @staticmethod
    def _resolve_device(device: str) -> str:
        import torch

        d = (device or "auto").lower()
        if d == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return d

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from funasr import AutoModel
        except ImportError as e:
            raise ImportError(
                "FunASR is not installed. Run: pip install -r requirements.txt\n"
                "Or set asr.backend=mock in config for pipeline testing."
            ) from e

        model_id = self.config.model_name or "paraformer-zh"
        kwargs = {"device": self._device}
        if self.config.model_dir:
            kwargs["model"] = self.config.model_dir
        else:
            # Common FunASR hub names
            hub_map = {
                "paraformer-zh": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                "paraformer-zh-streaming": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
                "sensevoice-small": "iic/SenseVoiceSmall",
            }
            kwargs["model"] = hub_map.get(model_id, model_id)

        self._model = AutoModel(**kwargs)

    def transcribe(self, waveform: np.ndarray, sr: int) -> str:
        self._ensure_model()
        wav = resample_mono(waveform, sr, 16000)

        # FunASR accepts file path or raw input depending on version; use temp wav for compatibility.
        import tempfile
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            sf.write(tmp, wav, 16000)
            result = self._model.generate(input=tmp)
        finally:
            import os

            if os.path.exists(tmp):
                os.remove(tmp)

        text = _extract_funasr_text(result)
        return normalize_asr_text(text, strip_punctuation=self.config.strip_punctuation)


def _extract_funasr_text(result) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("text", "preds", "prediction", "result"):
            if key in result:
                return _extract_funasr_text(result[key])
    if isinstance(result, (list, tuple)):
        if not result:
            return ""
        first = result[0]
        if isinstance(first, dict) and "text" in first:
            return str(first["text"])
        return _extract_funasr_text(first)
    return str(result)


def create_asr_backend(config: AsrConfig) -> AsrBackend:
    backend = (config.backend or "funasr").lower()
    if backend == "mock":
        return MockAsrBackend()
    if backend == "funasr":
        return FunAsrBackend(config)
    raise ValueError(f"Unknown ASR backend: {backend}")
