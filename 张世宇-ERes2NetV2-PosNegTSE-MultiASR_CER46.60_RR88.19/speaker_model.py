"""
Speaker embedding extractor using ECAPA-TDNN (SpeechBrain).

This module is independent from the deepfake detection model in model.py.
It lazily loads the pretrained ECAPA-TDNN checkpoint on first use and
provides a single utility `extract_embedding` that converts a numpy waveform
into a fixed-length, L2-normalized float32 vector suitable for cosine
similarity search.
"""

from __future__ import annotations

import os
import pathlib
import sys
import threading
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
import torchaudio


_TARGET_SR = 16000
_MIN_DURATION_SEC = 0.5


def _patch_torch_amp_compat() -> None:
    """SpeechBrain 1.1 expects torch.amp.custom_fwd/custom_bwd on newer torch."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "custom_fwd") and hasattr(torch.amp, "custom_bwd"):
        return
    if not hasattr(torch, "amp"):
        return

    def _identity_decorator(fn=None, **_kwargs):
        if fn is None:
            def wrap(inner):
                return inner

            return wrap
        return fn

    if not hasattr(torch.amp, "custom_fwd"):
        torch.amp.custom_fwd = _identity_decorator  # type: ignore[attr-defined]
    if not hasattr(torch.amp, "custom_bwd"):
        torch.amp.custom_bwd = _identity_decorator  # type: ignore[attr-defined]


class SpeakerEmbeddingBackend(ABC):
    name: str

    @abstractmethod
    def encode(self, waveform_np: np.ndarray, sr: int) -> np.ndarray:
        ...


class SpeakerEncoder(SpeakerEmbeddingBackend):
    """Thin wrapper around SpeechBrain's ECAPA-TDNN encoder."""

    name = "ecapa"

    def __init__(self, savedir: Optional[str] = None, device: Optional[torch.device] = None):
        _patch_torch_amp_compat()
        # Import here so the rest of the app can start even if speechbrain
        # isn't installed yet (import error will surface only when a speaker
        # endpoint is actually hit).
        from speechbrain.inference.speaker import EncoderClassifier  # type: ignore

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        _default_ecapa = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ecapa")
        savedir = savedir or os.getenv("ECAPA_SAVEDIR", _default_ecapa)
        source = os.getenv("ECAPA_SOURCE", "speechbrain/spkrec-ecapa-voxceleb")

        os.makedirs(savedir, exist_ok=True)
        self._device = device

        # On Windows the default symlink fetch strategy needs admin/dev-mode
        # privileges (WinError 1314). Fall back to plain file copies so the
        # encoder works for unprivileged users out of the box.
        from_hparams_kwargs = {
            "source": source,
            "savedir": savedir,
            "run_opts": {"device": str(device)},
        }
        try:
            from speechbrain.utils.fetching import LocalStrategy  # type: ignore
            from_hparams_kwargs["local_strategy"] = LocalStrategy.COPY
        except Exception:
            from_hparams_kwargs["local_strategy"] = "copy"

        self._model = EncoderClassifier.from_hparams(**from_hparams_kwargs)
        self._model.eval()

        # Probe embedding dim with a tiny dummy input (1 second of silence).
        with torch.no_grad():
            probe = torch.zeros(1, _TARGET_SR, device=device)
            emb = self._model.encode_batch(probe)
        self.embedding_dim = int(emb.shape[-1])

    @property
    def device(self) -> torch.device:
        return self._device

    def encode(self, waveform_np: np.ndarray, sr: int) -> np.ndarray:
        """
        Convert a mono waveform into an L2-normalized float32 embedding.

        Args:
            waveform_np: 1-D numpy array, mono.
            sr: sample rate of the input waveform.

        Returns:
            numpy array of shape (embedding_dim,), dtype float32, L2-normalized.
        """
        if waveform_np is None or waveform_np.size == 0:
            raise ValueError("Empty waveform provided to SpeakerEncoder.encode.")

        if waveform_np.ndim != 1:
            waveform_np = np.asarray(waveform_np).reshape(-1)

        waveform = torch.from_numpy(waveform_np.astype(np.float32))

        # Resample to 16k (required by ECAPA-TDNN VoxCeleb model).
        if sr != _TARGET_SR:
            waveform = torchaudio.functional.resample(waveform.unsqueeze(0), sr, _TARGET_SR).squeeze(0)

        duration = waveform.shape[0] / float(_TARGET_SR)
        if duration < _MIN_DURATION_SEC:
            raise ValueError(
                f"Audio too short for speaker enrollment/identification: "
                f"{duration:.2f}s < {_MIN_DURATION_SEC}s."
            )

        # Remove DC + peak-normalize for microphone robustness.
        waveform = waveform - torch.mean(waveform)
        peak = torch.max(torch.abs(waveform)).clamp_min(1e-6)
        waveform = waveform / peak

        wav = waveform.unsqueeze(0).to(self._device)  # (1, T)
        with torch.no_grad():
            emb = self._model.encode_batch(wav).squeeze().detach().cpu().numpy()

        emb = emb.astype(np.float32).reshape(-1)
        norm = float(np.linalg.norm(emb))
        if norm < 1e-9:
            raise RuntimeError("Zero-norm embedding produced; audio may be silent.")
        emb = emb / norm
        return emb


class WavLMSpeakerEncoder(SpeakerEmbeddingBackend):
    """WavLM speaker-verification embedding backend.

    This optional backend follows the public Hugging Face speaker verification
    checkpoints. It is loaded only when configured, so ECAPA remains the default
    no-extra-dependency path.
    """

    name = "wavlm"

    def __init__(self, model_name: str = "microsoft/wavlm-base-plus-sv", device: Optional[torch.device] = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector  # type: ignore
        except ImportError as e:
            raise ImportError(
                "WavLM backend requires transformers. Install it or remove 'wavlm' "
                "from gate.embedding_backends."
            ) from e

        self._device = device
        self._feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self._model = WavLMForXVector.from_pretrained(model_name).to(device)
        self._model.eval()

    def encode(self, waveform_np: np.ndarray, sr: int) -> np.ndarray:
        if waveform_np is None or waveform_np.size == 0:
            raise ValueError("Empty waveform provided to WavLMSpeakerEncoder.encode.")
        waveform_np = np.asarray(waveform_np, dtype=np.float32).reshape(-1)
        if sr != _TARGET_SR:
            wav = torch.from_numpy(waveform_np)
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, _TARGET_SR).squeeze(0)
            waveform_np = wav.numpy().astype(np.float32)

        duration = waveform_np.shape[0] / float(_TARGET_SR)
        if duration < _MIN_DURATION_SEC:
            raise ValueError(
                f"Audio too short for speaker enrollment/identification: "
                f"{duration:.2f}s < {_MIN_DURATION_SEC}s."
            )

        waveform_np = waveform_np - float(np.mean(waveform_np))
        peak = max(float(np.max(np.abs(waveform_np))), 1e-6)
        waveform_np = (waveform_np / peak).astype(np.float32)
        inputs = self._feature_extractor(
            waveform_np,
            sampling_rate=_TARGET_SR,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model(**inputs)
            emb = out.embeddings.squeeze().detach().cpu().numpy()

        emb = emb.astype(np.float32).reshape(-1)
        norm = float(np.linalg.norm(emb))
        if norm < 1e-9:
            raise RuntimeError("Zero-norm WavLM embedding produced; audio may be silent.")
        return emb / norm


_THREED_SPEAKER_SPECS = {
    "iic/speech_campplus_sv_zh-cn_16k-common": {
        "revision": "v1.0.0",
        "model": {
            "obj": "speakerlab.models.campplus.DTDNN.CAMPPlus",
            "args": {"feat_dim": 80, "embedding_size": 192},
        },
        "model_pt": "campplus_cn_common.bin",
    },
    "iic/speech_eres2netv2_sv_zh-cn_16k-common": {
        "revision": "v1.0.1",
        "model": {
            "obj": "speakerlab.models.eres2net.ERes2NetV2.ERes2NetV2",
            "args": {
                "feat_dim": 80,
                "embedding_size": 192,
                "baseWidth": 26,
                "scale": 2,
                "expansion": 2,
            },
        },
        "model_pt": "pretrained_eres2netv2.ckpt",
    },
}


class ThreeDSpeakerEncoder(SpeakerEmbeddingBackend):
    """3D-Speaker ModelScope backend for CAM++ / ERes2NetV2 embeddings."""

    def __init__(
        self,
        model_id: str,
        name: str,
        local_model_dir: str = "pretrained",
        device: Optional[torch.device] = None,
    ):
        self.name = name
        self.model_id = model_id
        if model_id not in _THREED_SPEAKER_SPECS:
            raise ValueError(f"Unsupported 3D-Speaker model id: {model_id}")

        repo_dir = pathlib.Path(__file__).resolve().parent / "external_3D-Speaker"
        if not repo_dir.is_dir():
            raise RuntimeError(
                f"3D-Speaker repository not found: {repo_dir}. "
                "Clone https://github.com/modelscope/3D-Speaker.git first."
            )
        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))

        from speakerlab.process.processor import FBank  # type: ignore
        from speakerlab.utils.builder import dynamic_import  # type: ignore

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

        conf = _THREED_SPEAKER_SPECS[model_id]
        local_root = pathlib.Path(local_model_dir)
        if not local_root.is_absolute():
            local_root = pathlib.Path(__file__).resolve().parent / local_root
        save_dir = local_root / model_id.split("/")[-1]
        save_dir.mkdir(exist_ok=True, parents=True)

        model_pt = save_dir / conf["model_pt"]
        if not model_pt.is_file():
            matches = list(save_dir.rglob(conf["model_pt"]))
            if matches:
                model_pt = matches[0]

        if not model_pt.is_file():
            try:
                from modelscope.hub.snapshot_download import snapshot_download  # type: ignore
            except ImportError as e:
                raise ImportError(
                    f"3D-Speaker weight not found locally: {model_pt}. "
                    "Either clone the ModelScope model into pretrained/ or install modelscope."
                ) from e

            cache_dir = pathlib.Path(snapshot_download(model_id, revision=conf["revision"]))
            model_pt = cache_dir / conf["model_pt"]
            matches = [] if model_pt.is_file() else list(cache_dir.rglob(conf["model_pt"]))
            if not matches:
                if not model_pt.is_file():
                    raise FileNotFoundError(f"Model weight not found in snapshot: {conf['model_pt']}")
            else:
                model_pt = matches[0]

        state = torch.load(model_pt, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        model_spec = conf["model"]
        model = dynamic_import(model_spec["obj"])(**model_spec["args"])
        model.load_state_dict(state)
        model.to(self._device)
        model.eval()
        self._model = model
        self._feature_extractor = FBank(80, sample_rate=16000, mean_nor=True)

    def encode(self, waveform_np: np.ndarray, sr: int) -> np.ndarray:
        if waveform_np is None or waveform_np.size == 0:
            raise ValueError("Empty waveform provided to ThreeDSpeakerEncoder.encode.")
        waveform_np = np.asarray(waveform_np, dtype=np.float32).reshape(-1)
        wav = torch.from_numpy(waveform_np)
        if sr != _TARGET_SR:
            wav = torchaudio.functional.resample(wav.unsqueeze(0), sr, _TARGET_SR).squeeze(0)
        duration = wav.shape[0] / float(_TARGET_SR)
        if duration < _MIN_DURATION_SEC:
            raise ValueError(
                f"Audio too short for speaker enrollment/identification: "
                f"{duration:.2f}s < {_MIN_DURATION_SEC}s."
            )
        wav = wav - torch.mean(wav)
        wav = wav / torch.max(torch.abs(wav)).clamp_min(1e-6)
        feat = self._feature_extractor(wav.unsqueeze(0)).unsqueeze(0).to(self._device)
        with torch.no_grad():
            emb = self._model(feat).detach().squeeze(0).cpu().numpy()
        emb = emb.astype(np.float32).reshape(-1)
        norm = float(np.linalg.norm(emb))
        if norm < 1e-9:
            raise RuntimeError("Zero-norm 3D-Speaker embedding produced; audio may be silent.")
        return emb / norm


_encoder_lock = threading.Lock()
_encoder_instance: Optional[SpeakerEncoder] = None
_backend_lock = threading.Lock()
_backend_instances: dict[str, SpeakerEmbeddingBackend] = {}


def get_speaker_encoder() -> SpeakerEncoder:
    """Lazy, thread-safe singleton accessor for the ECAPA encoder."""
    global _encoder_instance
    if _encoder_instance is None:
        with _encoder_lock:
            if _encoder_instance is None:
                _encoder_instance = SpeakerEncoder()
    return _encoder_instance


def get_speaker_backend(name: str = "ecapa", **kwargs) -> SpeakerEmbeddingBackend:
    key = (name or "ecapa").strip().lower()
    if key in ("speechbrain", "speechbrain_ecapa"):
        key = "ecapa"
    if key == "wavlm":
        cache_key = f"wavlm:{kwargs.get('model_name', 'microsoft/wavlm-base-plus-sv')}"
    elif key in ("campplus", "eres2netv2"):
        cache_key = f"{key}:{kwargs.get('model_id', key)}"
    else:
        cache_key = key
    if cache_key not in _backend_instances:
        with _backend_lock:
            if cache_key not in _backend_instances:
                if key == "ecapa":
                    _backend_instances[cache_key] = get_speaker_encoder()
                elif key == "wavlm":
                    _backend_instances[cache_key] = WavLMSpeakerEncoder(
                        model_name=kwargs.get("model_name", "microsoft/wavlm-base-plus-sv")
                    )
                elif key == "campplus":
                    _backend_instances[cache_key] = ThreeDSpeakerEncoder(
                        model_id=kwargs.get("model_id", "iic/speech_campplus_sv_zh-cn_16k-common"),
                        name="campplus",
                        local_model_dir=kwargs.get("local_model_dir", "pretrained"),
                    )
                elif key == "eres2netv2":
                    _backend_instances[cache_key] = ThreeDSpeakerEncoder(
                        model_id=kwargs.get("model_id", "iic/speech_eres2netv2_sv_zh-cn_16k-common"),
                        name="eres2netv2",
                        local_model_dir=kwargs.get("local_model_dir", "pretrained"),
                    )
                else:
                    raise ValueError(f"Unknown speaker embedding backend: {name}")
    return _backend_instances[cache_key]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for pre-normalized or raw embeddings."""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
