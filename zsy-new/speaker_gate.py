"""
Target-speaker verification using wake audio as enrollment.

Wraps the existing ECAPA-TDNN encoder from speaker_model.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from audio_quality import AudioQuality
from speaker_model import SpeakerEmbeddingBackend, cosine_similarity, get_speaker_backend

from audio_utils import build_segments, load_audio_file, resample_mono
from config import GateConfig


@dataclass
class GateResult:
    accepted: bool
    similarity: float
    segment_similarities: List[float]
    reason: str
    threshold_used: float = 0.0
    backend_scores: Dict[str, float] | None = None


class SpeakerGate:
    """Enroll target speaker from wake audio; gate command audio."""

    def __init__(self, config: Optional[GateConfig] = None):
        self.config = config or GateConfig()
        self._encoders: Dict[str, SpeakerEmbeddingBackend] = {}
        self._target_emb: Optional[np.ndarray] = None
        self._target_embs: Dict[str, np.ndarray] = {}
        self._enroll_segments: List[np.ndarray] = []

    @property
    def encoder(self):
        return self._get_backend("ecapa")

    def _backend_names(self) -> list[str]:
        names = [str(n).strip().lower() for n in (self.config.embedding_backends or ["ecapa"]) if str(n).strip()]
        return names or ["ecapa"]

    def _get_backend(self, name: str) -> SpeakerEmbeddingBackend:
        key = (name or "ecapa").strip().lower()
        if key not in self._encoders:
            kwargs = {}
            if key == "wavlm":
                kwargs["model_name"] = self.config.wavlm_model_name
            elif key == "campplus":
                kwargs["model_id"] = self.config.campplus_model_id
                kwargs["local_model_dir"] = self.config.modelscope_local_dir
            elif key == "eres2netv2":
                kwargs["model_id"] = self.config.eres2netv2_model_id
                kwargs["local_model_dir"] = self.config.modelscope_local_dir
            self._encoders[key] = get_speaker_backend(key, **kwargs)
        return self._encoders[key]

    def _backend_weight(self, name: str) -> float:
        raw = self.config.backend_weights or {}
        return float(raw.get(name, raw.get(name.lower(), 1.0)))

    def _cascade_enabled(self) -> bool:
        return bool(getattr(self.config, "cascade_enabled", False)) and len(self._backend_names()) > 1

    def _cascade_primary_name(self) -> str:
        names = self._backend_names()
        requested = str(getattr(self.config, "cascade_primary_backend", "") or "").strip().lower()
        if requested and requested in names:
            return requested
        return names[0]

    def _enrollment_backend_names(self) -> list[str]:
        if self._cascade_enabled():
            return [self._cascade_primary_name()]
        return self._backend_names()

    def _aggregate_scores(self, sims: List[float]) -> float:
        if not sims:
            return 0.0
        arr = np.asarray(sims, dtype=np.float64)
        mode = (self.config.aggregate or "topk_mean").strip().lower()
        if mode == "mean":
            return float(np.mean(arr))
        if mode == "median":
            return float(np.median(arr))
        if mode == "max":
            return float(np.max(arr))
        if mode == "topk_mean":
            k = max(1, min(int(self.config.top_k), arr.size))
            top = np.sort(arr)[-k:]
            return float(np.mean(top))
        raise ValueError(f"Unknown gate aggregate mode: {self.config.aggregate}")

    def _compute_target_embedding(self, name: str, segments: List[np.ndarray]) -> np.ndarray:
        backend = self._get_backend(name)
        embs = []
        for seg in segments:
            emb = self._encode_segment_safe(backend, seg)
            if emb is not None:
                embs.append(emb)
        if not embs:
            raise ValueError(f"No valid enrollment segments for {name} speaker embedding.")
        avg = np.mean(np.stack(embs, axis=0), axis=0).astype(np.float32)
        norm = float(np.linalg.norm(avg))
        if norm < 1e-9:
            raise RuntimeError(f"Zero-norm enrollment embedding for {name}.")
        return avg / norm

    def _ensure_target_embedding(self, name: str) -> Optional[np.ndarray]:
        key = (name or "").strip().lower()
        if not key:
            return None
        if key in self._target_embs:
            return self._target_embs[key]
        if not self._enroll_segments:
            return None
        try:
            emb = self._compute_target_embedding(key, self._enroll_segments)
        except (RuntimeError, ValueError):
            return None
        self._target_embs[key] = emb
        if self._target_emb is None:
            self._target_emb = emb
        return emb

    def _score_backend_segments(self, name: str, segments: List[np.ndarray]) -> List[Optional[float]]:
        backend = self._get_backend(name)
        target_emb = self._ensure_target_embedding(name)
        if target_emb is None:
            return []

        scores: List[Optional[float]] = []
        for seg in segments:
            emb = self._encode_segment_safe(backend, seg)
            if emb is None:
                scores.append(None)
            else:
                scores.append(cosine_similarity(target_emb, emb))
        return scores

    def _fuse_backend_scores(
        self,
        backend_names: List[str],
        segments: List[np.ndarray],
        precomputed_scores: Optional[Dict[str, List[Optional[float]]]] = None,
    ) -> tuple[float, List[float], Dict[str, float]]:
        precomputed_scores = precomputed_scores or {}
        backend_scores: Dict[str, float] = {}
        segment_score_sum = np.zeros(len(segments), dtype=np.float64)
        segment_weight_sum = np.zeros(len(segments), dtype=np.float64)

        for name in backend_names:
            scores = precomputed_scores.get(name)
            if scores is None:
                scores = self._score_backend_segments(name, segments)
            valid_scores = [float(score) for score in scores if score is not None]
            if not valid_scores:
                continue

            backend_scores[name] = self._aggregate_scores(valid_scores)
            weight = self._backend_weight(name)
            if weight <= 0:
                continue
            for idx, score in enumerate(scores):
                if score is not None and idx < len(segment_score_sum):
                    segment_score_sum[idx] += weight * float(score)
                    segment_weight_sum[idx] += weight

        weight_sum = 0.0
        weighted_score = 0.0
        for name, score in backend_scores.items():
            weight = self._backend_weight(name)
            if weight <= 0:
                continue
            weighted_score += weight * score
            weight_sum += weight

        agg_sim = float(weighted_score / weight_sum) if weight_sum > 0 else 0.0
        segment_sims = [
            float(score_sum / weight_sum)
            for score_sum, weight_sum in zip(segment_score_sum, segment_weight_sum)
            if weight_sum > 0
        ]
        return agg_sim, segment_sims, backend_scores

    def _reason_for_score(self, agg_sim: float, accepted: bool, threshold: float) -> str:
        if agg_sim < threshold and agg_sim < getattr(self.config, "speaker_reject_low", threshold):
            return "clear_reject"
        if agg_sim >= getattr(self.config, "speaker_accept_high", threshold):
            return "clear_accept"
        if accepted:
            return "accepted_uncertain_band"
        return "uncertain_below_threshold"

    def enroll_from_waveform(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        waveform = resample_mono(waveform, sr, 16000)
        segments = build_segments(
            waveform,
            16000,
            segment_sec=self.config.segment_sec,
            num_segments=self.config.num_segments,
        )
        self._target_embs = {}
        self._enroll_segments = segments
        for name in self._enrollment_backend_names():
            self._target_embs[name] = self._compute_target_embedding(name, segments)
        self._target_emb = self._target_embs.get("ecapa")
        if self._target_emb is None:
            self._target_emb = next(iter(self._target_embs.values()))
        return self._target_emb

    def enroll_from_file(self, wake_path: str) -> np.ndarray:
        wav, sr = load_audio_file(wake_path)
        return self.enroll_from_waveform(wav, sr)

    def _encode_segment_safe(self, backend: SpeakerEmbeddingBackend, segment: np.ndarray) -> Optional[np.ndarray]:
        duration = segment.shape[0] / 16000.0
        if duration < self.config.min_duration_sec:
            return None
        try:
            return backend.encode(segment, 16000)
        except ValueError:
            return None

    def _threshold_for_quality(
        self,
        wake_quality: Optional[AudioQuality] = None,
        query_quality: Optional[AudioQuality] = None,
    ) -> float:
        threshold = float(self.config.threshold)
        if not self.config.dynamic_threshold:
            return threshold
        if wake_quality is not None and wake_quality.score < self.config.min_quality_for_base_threshold:
            threshold += float(self.config.low_quality_threshold_boost)
        if query_quality is not None and query_quality.snr_db < self.config.min_snr_for_base_threshold_db:
            threshold += float(self.config.low_snr_threshold_boost)
        return threshold

    def score_waveform(
        self,
        waveform: np.ndarray,
        sr: int,
        wake_quality: Optional[AudioQuality] = None,
        query_quality: Optional[AudioQuality] = None,
    ) -> GateResult:
        if not self._target_embs:
            raise RuntimeError("Call enroll_from_file/waveform before scoring.")

        waveform = resample_mono(waveform, sr, 16000)
        segments = build_segments(
            waveform,
            16000,
            segment_sec=self.config.segment_sec,
            num_segments=self.config.num_segments,
        )
        threshold = self._threshold_for_quality(wake_quality, query_quality)

        if self._cascade_enabled():
            primary_name = self._cascade_primary_name()
            primary_segment_scores = self._score_backend_segments(primary_name, segments)
            primary_scores = [float(score) for score in primary_segment_scores if score is not None]
            if primary_scores:
                primary_score = self._aggregate_scores(primary_scores)
                primary_backend_scores = {primary_name: primary_score}
                low = float(getattr(self.config, "cascade_low", self.config.threshold))
                high = float(getattr(self.config, "cascade_high", self.config.threshold))
                if primary_score < low:
                    return GateResult(
                        accepted=False,
                        similarity=primary_score,
                        segment_similarities=primary_scores,
                        reason="cascade_primary_reject",
                        threshold_used=threshold,
                        backend_scores=primary_backend_scores,
                    )
                if primary_score >= high:
                    return GateResult(
                        accepted=True,
                        similarity=primary_score,
                        segment_similarities=primary_scores,
                        reason="cascade_primary_accept",
                        threshold_used=threshold,
                        backend_scores=primary_backend_scores,
                    )

                agg_sim, segment_sims, backend_scores = self._fuse_backend_scores(
                    self._backend_names(),
                    segments,
                    precomputed_scores={primary_name: primary_segment_scores},
                )
                accepted = agg_sim >= threshold
                reason = "cascade_" + self._reason_for_score(agg_sim, accepted, threshold)
                return GateResult(
                    accepted=accepted,
                    similarity=agg_sim,
                    segment_similarities=segment_sims,
                    reason=reason,
                    threshold_used=threshold,
                    backend_scores=backend_scores,
                )

        agg_sim, segment_sims, backend_scores = self._fuse_backend_scores(self._backend_names(), segments)

        if not backend_scores:
            return GateResult(
                accepted=False,
                similarity=0.0,
                segment_similarities=[],
                reason="no_valid_segments",
                threshold_used=threshold,
                backend_scores={},
            )

        accepted = agg_sim >= threshold
        reason = self._reason_for_score(agg_sim, accepted, threshold)

        return GateResult(
            accepted=accepted,
            similarity=agg_sim,
            segment_similarities=segment_sims,
            reason=reason,
            threshold_used=threshold,
            backend_scores=backend_scores,
        )

    def should_accept(
        self,
        waveform: np.ndarray,
        sr: int,
        wake_quality: Optional[AudioQuality] = None,
        query_quality: Optional[AudioQuality] = None,
    ) -> GateResult:
        return self.score_waveform(waveform, sr, wake_quality=wake_quality, query_quality=query_quality)

    def reset(self) -> None:
        self._target_emb = None
        self._target_embs = {}
        self._enroll_segments = []
