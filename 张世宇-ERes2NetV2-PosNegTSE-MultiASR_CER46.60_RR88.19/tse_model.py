"""Target-speaker extraction with noisy positive/negative enrollments.

The implementation wraps the official NeurIPS 2025 improved monaural model
without modifying the upstream source tree. Imports are lazy so the regular
speaker-verification pipeline still works when ESPnet is not installed.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio


_TARGET_SR = 16000
_ROOT = Path(__file__).resolve().parent
_DEFAULT_SOURCE_DIR = (
    _ROOT
    / "external_TSE_PosNeg"
    / "TSE-through-Positive-Negative-Enroll-main"
)
_DEFAULT_CHECKPOINT = _DEFAULT_SOURCE_DIR / "improved-monaural.pt"


@dataclass
class EnrollmentPair:
    positive: np.ndarray
    negative: np.ndarray
    sample_rate: int = _TARGET_SR
    negative_source: str = "low_energy_wake_frames"


@dataclass
class TSEOutput:
    waveform: np.ndarray
    sample_rate: int
    elapsed_sec: float
    input_rms: float
    output_rms: float


def _mono_16k(waveform: np.ndarray, sr: int) -> np.ndarray:
    wav = np.asarray(waveform, dtype=np.float32)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=-1)
    wav = wav.reshape(-1)
    if sr != _TARGET_SR:
        tensor = torch.from_numpy(wav).unsqueeze(0)
        wav = (
            torchaudio.functional.resample(tensor, int(sr), _TARGET_SR)
            .squeeze(0)
            .numpy()
        )
    if wav.size:
        wav = wav - float(np.mean(wav))
    return wav.astype(np.float32)


def _repeat_to_length(waveform: np.ndarray, target_len: int) -> np.ndarray:
    wav = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if wav.size == 0:
        return np.zeros(target_len, dtype=np.float32)
    if wav.size >= target_len:
        return wav[:target_len].copy()
    repeats = int(np.ceil(target_len / wav.size))
    return np.tile(wav, repeats)[:target_len].astype(np.float32)


def build_pseudo_enrollments(
    wake_waveform: np.ndarray,
    wake_sr: int,
    min_enrollment_sec: float = 1.5,
    frame_ms: int = 30,
    low_energy_quantile: float = 0.30,
) -> EnrollmentPair:
    """Build a conservative first-pass enrollment pair from a short wake clip.

    The complete wake clip is the noisy positive enrollment. Low-energy frames
    form a pseudo-negative enrollment. This is a fallback for competition data
    that has no target-silent annotation; a KWS-aligned enrollment builder can
    replace it later without changing the TSE model interface.
    """
    wake = _mono_16k(wake_waveform, wake_sr)
    min_len = max(1, int(round(min_enrollment_sec * _TARGET_SR)))
    positive = _repeat_to_length(wake, max(min_len, wake.size))

    frame_len = max(1, int(round(frame_ms * _TARGET_SR / 1000.0)))
    n_frames = wake.size // frame_len
    if n_frames:
        framed = wake[: n_frames * frame_len].reshape(n_frames, frame_len)
        energy = np.mean(framed * framed, axis=1)
        threshold = float(np.quantile(energy, low_energy_quantile))
        selected = framed[energy <= threshold].reshape(-1)
    else:
        selected = wake.copy()

    if selected.size == 0 or float(np.std(selected)) < 1e-7:
        # The official model normalizes enrollment variance, so all-zero input
        # is invalid. Use deterministic very-low-level noise as a safe fallback.
        rng = np.random.default_rng(0)
        scale = max(1e-5, float(np.std(wake)) * 0.01)
        selected = rng.normal(0.0, scale, size=min_len).astype(np.float32)
        negative_source = "deterministic_low_level_noise"
    else:
        negative_source = "low_energy_wake_frames"

    negative = _repeat_to_length(selected, max(min_len, positive.size))
    return EnrollmentPair(
        positive=positive,
        negative=negative,
        sample_rate=_TARGET_SR,
        negative_source=negative_source,
    )


class PositiveNegativeTSE:
    """Lazy wrapper for the official improved monaural Pos/Neg TSE model."""

    def __init__(
        self,
        checkpoint: str | Path = _DEFAULT_CHECKPOINT,
        source_dir: str | Path = _DEFAULT_SOURCE_DIR,
        device: str = "auto",
    ) -> None:
        self.checkpoint = Path(checkpoint).resolve()
        self.source_dir = Path(source_dir).resolve()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self._model: Optional[torch.nn.Module] = None

    def _build_model(self) -> torch.nn.Module:
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"TSE checkpoint not found: {self.checkpoint}")
        if not self.source_dir.is_dir():
            raise FileNotFoundError(f"TSE source directory not found: {self.source_dir}")
        source_text = str(self.source_dir)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)

        from model.tfgridnet_encoder import TFGridNet_encoder
        from improved_model.GridnetAttnHead import GridNetBlock_attnhead
        from improved_model.USEF_TFGridnet import Tar_Model

        encoder = TFGridNet_encoder(
            num_ch=2,
            n_fft=128,
            stride=64,
            num_blocks=1,
            binaural=False,
        )
        encoder_head = GridNetBlock_attnhead(
            layer_num=2,
            pooling_size=1,
            stride=1,
            return_clean_dvec=False,
            out_dim=0,
            refine_layer_num=2,
            fusion_shortcut=[0, 1],
            cut_pos=True,
        )
        model = Tar_Model(
            n_freqs=65,
            hidden_channels=64,
            n_head=4,
            emb_dim=64,
            emb_ks=1,
            emb_hs=1,
            num_layers=3,
            encoder=encoder,
            encoder_head=encoder_head,
            train_encoder=False,
            train_encoder_head=False,
        )
        payload = torch.load(self.checkpoint, map_location="cpu")
        model.load_state_dict(payload["state_dict"], strict=True)
        model.to(self.device)
        model.eval()
        return model

    @property
    def model(self) -> torch.nn.Module:
        if self._model is None:
            self._model = self._build_model()
        return self._model

    def extract(
        self,
        mixture: np.ndarray,
        mixture_sr: int,
        positive_enrollment: np.ndarray,
        positive_sr: int,
        negative_enrollment: np.ndarray,
        negative_sr: int,
    ) -> TSEOutput:
        mix = _mono_16k(mixture, mixture_sr)
        pos = _mono_16k(positive_enrollment, positive_sr)
        neg = _mono_16k(negative_enrollment, negative_sr)
        if mix.size == 0 or pos.size == 0 or neg.size == 0:
            raise ValueError("Mixture and positive/negative enrollments must be non-empty.")

        mix_t = torch.from_numpy(mix).to(self.device)[None, None, :]
        pos_t = torch.from_numpy(pos).to(self.device)[None, None, :]
        neg_t = torch.from_numpy(neg).to(self.device)[None, None, :]

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            cond_emb, _, _ = self.model.encoder_pos_neg(pos_t, neg_t, recons=False)
            output = self.model(mix_t, cond_emb)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - t0

        out = output.squeeze().detach().float().cpu().numpy().reshape(-1)
        if out.size > mix.size:
            out = out[: mix.size]
        elif out.size < mix.size:
            out = np.pad(out, (0, mix.size - out.size))
        return TSEOutput(
            waveform=out.astype(np.float32),
            sample_rate=_TARGET_SR,
            elapsed_sec=float(elapsed),
            input_rms=float(np.sqrt(np.mean(mix * mix) + 1e-12)),
            output_rms=float(np.sqrt(np.mean(out * out) + 1e-12)),
        )


_DEFAULT_MODEL: Optional[PositiveNegativeTSE] = None


def get_default_tse_model() -> PositiveNegativeTSE:
    global _DEFAULT_MODEL
    if _DEFAULT_MODEL is None:
        _DEFAULT_MODEL = PositiveNegativeTSE()
    return _DEFAULT_MODEL
