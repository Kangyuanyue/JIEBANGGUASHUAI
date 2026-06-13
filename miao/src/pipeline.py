"""Inference pipeline skeleton.

This file is intentionally model-agnostic. Replace the TODO sections with the
chosen VAD, speaker verification, pVAD, TSE and ASR implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import yaml

from .decision import Evidence, final_decision, should_reject_early, should_use_tse
from .preprocess import AudioQuality, load_audio, preprocess_audio
from .text_norm import normalize_text


@dataclass
class Prediction:
    content: str
    evidence: Evidence
    accepted: bool
    decision_score: float
    decision_reason: str
    tse_used: bool = False


class CompetitionPipeline:
    """Target-speaker conditioned ASR pipeline.

    Parameters
    ----------
    config:
        Parsed YAML configuration.
    thresholds:
        Parsed YAML threshold dictionary.
    mock:
        If True, do not load real models. This is useful for testing I/O and
        JSON format before model integration.
    """

    def __init__(self, config: Dict[str, Any], thresholds: Dict[str, Any], mock: bool = False):
        self.config = config
        self.thresholds = thresholds
        self.mock = mock
        self.sample_rate = int(config.get("preprocess", {}).get("target_sample_rate", 16000))
        self.normalize_peak = float(config.get("preprocess", {}).get("normalize_peak", 0.95))
        self._load_models()

    @classmethod
    def from_yaml(cls, config_path: str | Path, thresholds_path: str | Path, mock: bool = False) -> "CompetitionPipeline":
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        with open(thresholds_path, "r", encoding="utf-8") as f:
            thresholds = yaml.safe_load(f) or {}
        return cls(config=config, thresholds=thresholds, mock=mock)

    def _load_models(self) -> None:
        """Load real models here.

        TODO for final implementation:
        - speaker encoder: CAM++ / ERes2Net / ECAPA
        - pVAD: target-speaker conditioned frame classifier
        - TSE: VoiceFilter / SpeakerBeam style extractor
        - ASR: Paraformer-zh / SenseVoice / Zipformer
        """
        if self.mock:
            self.speaker_model = None
            self.pvad_model = None
            self.tse_model = None
            self.asr_model = None
            return

        # Keep explicit errors until models are connected. This prevents silent
        # fake scores in real submissions.
        raise NotImplementedError(
            "Real model loading is not implemented. Run with --mock for I/O tests, "
            "or implement _load_models and model inference methods."
        )

    def infer_sample(self, sample: Dict[str, Any]) -> Prediction:
        wake_audio_path = sample.get("wake_audio") or sample.get("唤醒音频") or sample.get("唤醒音频名字")
        query_audio_path = sample.get("query_audio") or sample.get("识别音频") or sample.get("识别音频名字")

        if self.mock:
            # Mock mode: conservative all-reject baseline. This validates JSON
            # structure and duration measurement but is not a competitive model.
            evidence = Evidence(no_speech=False, speaker_similarity=0.0, target_frame_ratio=0.0)
            return Prediction(content="", evidence=evidence, accepted=False, decision_score=0.0, decision_reason="mock_all_reject")

        if not wake_audio_path or not query_audio_path:
            raise ValueError(f"Missing wake/query audio path in sample: {sample}")

        wake_audio, sr = load_audio(wake_audio_path, target_sample_rate=self.sample_rate)
        query_audio, _ = load_audio(query_audio_path, target_sample_rate=self.sample_rate)

        wake_proc, wake_quality = preprocess_audio(wake_audio, sr, normalize_peak=self.normalize_peak)
        query_proc, query_quality = preprocess_audio(query_audio, sr, normalize_peak=self.normalize_peak)

        if query_quality.no_speech:
            evidence = Evidence(no_speech=True, query_snr_db=query_quality.snr_db, wake_quality_score=wake_quality.score)
            return Prediction(content="", evidence=evidence, accepted=False, decision_score=0.0, decision_reason="no_speech")

        target_embedding = self.extract_wake_embedding(wake_proc, sr, wake_quality)
        speaker_similarity = self.compute_speaker_similarity(target_embedding, query_proc, sr)
        pvad_features = self.run_pvad(query_proc, sr, target_embedding)

        evidence = Evidence(
            speaker_similarity=speaker_similarity,
            target_frame_ratio=pvad_features.get("target_frame_ratio", 0.0),
            non_target_frame_ratio=pvad_features.get("non_target_frame_ratio", 0.0),
            overlap_probability=pvad_features.get("overlap_probability", 0.0),
            no_speech=query_quality.no_speech,
            query_snr_db=query_quality.snr_db,
            wake_quality_score=wake_quality.score,
            enrollment_bad_quality=1.0 - wake_quality.score,
            query_noise_penalty=max(0.0, min(1.0, (-query_quality.snr_db) / 10.0)),
        )

        if should_reject_early(evidence, self.thresholds):
            return Prediction(content="", evidence=evidence, accepted=False, decision_score=0.0, decision_reason="early_reject")

        tse_used = should_use_tse(evidence, self.thresholds)
        asr_audio = self.run_target_speaker_extraction(query_proc, sr, target_embedding) if tse_used else query_proc
        raw_text, asr_confidence = self.run_asr(asr_audio, sr)
        fixed_text, command_score = self.postprocess_command(raw_text)

        evidence.asr_confidence = asr_confidence
        evidence.command_prior_score = command_score

        accepted, score, reason = final_decision(evidence, self.thresholds)
        return Prediction(
            content=fixed_text if accepted else "",
            evidence=evidence,
            accepted=accepted,
            decision_score=score,
            decision_reason=reason,
            tse_used=tse_used,
        )

    # ------------------------------------------------------------------
    # Model-specific methods. Implement these for final submission.
    # ------------------------------------------------------------------

    def extract_wake_embedding(self, audio: np.ndarray, sr: int, quality: AudioQuality) -> np.ndarray:
        raise NotImplementedError

    def compute_speaker_similarity(self, target_embedding: np.ndarray, query_audio: np.ndarray, sr: int) -> float:
        raise NotImplementedError

    def run_pvad(self, query_audio: np.ndarray, sr: int, target_embedding: np.ndarray) -> Dict[str, float]:
        raise NotImplementedError

    def run_target_speaker_extraction(self, query_audio: np.ndarray, sr: int, target_embedding: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def run_asr(self, audio: np.ndarray, sr: int) -> Tuple[str, float]:
        raise NotImplementedError

    def postprocess_command(self, raw_text: str) -> Tuple[str, float]:
        text = normalize_text(raw_text, normalize_digits=False)
        command_score = 1.0 if text else 0.0
        return text, command_score
