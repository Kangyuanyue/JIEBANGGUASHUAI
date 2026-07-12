#!/usr/bin/env python3
"""Cached speaker-trial evaluation for external calibration data.

This script mirrors the speaker-gate scoring path used by the competition
pipeline, but caches per-audio speaker embeddings so large trial lists do not
re-encode the same wake/query audio repeatedly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from audio_quality import AudioQuality, preprocess_waveform  # noqa: E402
from audio_utils import build_segments, load_audio_file  # noqa: E402
from config import GateConfig, PipelineConfig, apply_env_overrides, load_config  # noqa: E402
from speaker_eval import SpeakerTrial, compute_metrics, load_trials  # noqa: E402
from speaker_model import SpeakerEmbeddingBackend, cosine_similarity, get_speaker_backend  # noqa: E402


@dataclass
class CachedAudioState:
    quality: AudioQuality | None = None
    waveform: np.ndarray | None = None
    sr: int = 0
    backend_segments: dict[str, list[np.ndarray]] | None = None
    error: str = ""


class SpeakerEmbeddingCache:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self._audio: dict[str, CachedAudioState] = {}
        self._backends: dict[str, SpeakerEmbeddingBackend] = {}
        self.n_audio_loads = 0
        self.n_backend_encodes = 0

    def backend_names(self) -> list[str]:
        names = [str(n).strip().lower() for n in self.cfg.gate.embedding_backends if str(n).strip()]
        return names or ["ecapa"]

    def backend_weight(self, name: str) -> float:
        raw = self.cfg.gate.backend_weights or {}
        return float(raw.get(name, raw.get(name.lower(), 1.0)))

    def get_backend(self, name: str) -> SpeakerEmbeddingBackend:
        key = (name or "ecapa").strip().lower()
        if key not in self._backends:
            kwargs: dict[str, Any] = {}
            if key == "wavlm":
                kwargs["model_name"] = self.cfg.gate.wavlm_model_name
            elif key == "campplus":
                kwargs["model_id"] = self.cfg.gate.campplus_model_id
                kwargs["local_model_dir"] = self.cfg.gate.modelscope_local_dir
            elif key == "eres2netv2":
                kwargs["model_id"] = self.cfg.gate.eres2netv2_model_id
                kwargs["local_model_dir"] = self.cfg.gate.modelscope_local_dir
            self._backends[key] = get_speaker_backend(key, **kwargs)
        return self._backends[key]

    def _state(self, path: str) -> CachedAudioState:
        key = str(Path(path).resolve())
        if key not in self._audio:
            self._audio[key] = CachedAudioState(backend_segments={})
        return self._audio[key]

    def _load_preprocessed(self, path: str) -> tuple[np.ndarray, int, AudioQuality]:
        wav, sr = load_audio_file(path)
        self.n_audio_loads += 1
        return preprocess_waveform(
            wav,
            sr,
            target_sr=self.cfg.target_sr,
            enable_vad=self.cfg.preprocess.enable_vad,
            vad_threshold_ratio=self.cfg.preprocess.vad_threshold_ratio,
        )

    def _prepare_audio(self, path: str) -> CachedAudioState:
        state = self._state(path)
        if state.quality is None:
            try:
                wav, sr, quality = self._load_preprocessed(path)
                state.waveform = wav
                state.sr = sr
                state.quality = quality
            except Exception as e:
                state.error = repr(e)
                state.quality = AudioQuality(no_speech=True, score=0.0)
                state.waveform = np.asarray([], dtype=np.float32)
                state.sr = self.cfg.target_sr
        return state

    def quality(self, path: str) -> AudioQuality:
        state = self._prepare_audio(path)
        assert state.quality is not None
        return state.quality

    def segment_embeddings(self, path: str, backend_name: str) -> list[np.ndarray]:
        backend_name = backend_name.strip().lower()
        state = self._prepare_audio(path)
        assert state.backend_segments is not None
        if backend_name in state.backend_segments:
            return state.backend_segments[backend_name]

        embeddings: list[np.ndarray] = []
        try:
            if state.waveform is None:
                wav, sr, quality = self._load_preprocessed(path)
                state.waveform = wav
                state.sr = sr
                state.quality = quality
            assert state.waveform is not None
            wav = state.waveform
            sr = state.sr or self.cfg.target_sr
            segments = build_segments(
                wav,
                sr,
                segment_sec=self.cfg.gate.segment_sec,
                num_segments=self.cfg.gate.num_segments,
            )
            backend = self.get_backend(backend_name)
            for seg in segments:
                if seg.shape[0] / float(sr) < self.cfg.gate.min_duration_sec:
                    continue
                try:
                    embeddings.append(backend.encode(seg, sr))
                except ValueError:
                    continue
            self.n_backend_encodes += 1
        except Exception as e:
            state.error = repr(e)

        state.backend_segments[backend_name] = embeddings
        state.waveform = None
        return embeddings

    def error(self, path: str) -> str:
        return self._state(path).error

    @property
    def n_cached_audio(self) -> int:
        return len(self._audio)


def _aggregate_scores(sims: list[float], gate: GateConfig) -> float:
    if not sims:
        return 0.0
    arr = np.asarray(sims, dtype=np.float64)
    mode = (gate.aggregate or "topk_mean").strip().lower()
    if mode == "mean":
        return float(np.mean(arr))
    if mode == "median":
        return float(np.median(arr))
    if mode == "max":
        return float(np.max(arr))
    if mode == "topk_mean":
        k = max(1, min(int(gate.top_k), arr.size))
        return float(np.mean(np.sort(arr)[-k:]))
    raise ValueError(f"Unknown gate aggregate mode: {gate.aggregate}")


def _mean_l2_normalized(embeddings: list[np.ndarray]) -> np.ndarray | None:
    if not embeddings:
        return None
    avg = np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32)
    norm = float(np.linalg.norm(avg))
    if norm < 1e-9:
        return None
    return avg / norm


def _threshold_for_quality(cfg: GateConfig, wake_quality: AudioQuality, query_quality: AudioQuality) -> float:
    threshold = float(cfg.threshold)
    if not cfg.dynamic_threshold:
        return threshold
    if wake_quality.score < cfg.min_quality_for_base_threshold:
        threshold += float(cfg.low_quality_threshold_boost)
    if query_quality.snr_db < cfg.min_snr_for_base_threshold_db:
        threshold += float(cfg.low_snr_threshold_boost)
    return threshold


def _score_trial(
    trial: SpeakerTrial,
    cache: SpeakerEmbeddingCache,
) -> tuple[float, dict[str, Any]]:
    cfg = cache.cfg
    wake_quality = cache.quality(trial.enroll_audio)
    query_quality = cache.quality(trial.test_audio)

    threshold_used = _threshold_for_quality(cfg.gate, wake_quality, query_quality)
    detail: dict[str, Any] = {
        "id": trial.trial_id,
        "enroll_audio": trial.enroll_audio,
        "test_audio": trial.test_audio,
        "label": int(trial.label),
        "wake_quality": asdict(wake_quality),
        "query_quality": asdict(query_quality),
        "threshold_used": threshold_used,
    }

    if query_quality.no_speech:
        detail.update(
            {
                "score": 0.0,
                "accepted": False,
                "reason": "no_speech",
                "backend_scores": {},
                "segment_similarities": [],
            }
        )
        return 0.0, detail

    backend_scores: dict[str, float] = {}
    segment_similarities: list[float] = []
    weighted_score = 0.0
    weight_sum = 0.0
    errors: dict[str, str] = {}

    for name in cache.backend_names():
        enroll_embs = cache.segment_embeddings(trial.enroll_audio, name)
        test_embs = cache.segment_embeddings(trial.test_audio, name)
        target_emb = _mean_l2_normalized(enroll_embs)
        if target_emb is None or not test_embs:
            errors[name] = cache.error(trial.enroll_audio) or cache.error(trial.test_audio) or "no_valid_segments"
            continue

        sims = [cosine_similarity(target_emb, emb) for emb in test_embs]
        backend_score = _aggregate_scores(sims, cfg.gate)
        backend_scores[name] = backend_score
        segment_similarities.extend(float(v) for v in sims)
        weight = cache.backend_weight(name)
        if weight > 0:
            weighted_score += weight * backend_score
            weight_sum += weight

    score = float(weighted_score / weight_sum) if weight_sum > 0 else 0.0
    accepted = score >= threshold_used
    if not backend_scores:
        reason = "no_valid_segments"
    elif score < cfg.gate.threshold and score < getattr(cfg.gate, "speaker_reject_low", cfg.gate.threshold):
        reason = "clear_reject"
    elif score >= getattr(cfg.gate, "speaker_accept_high", cfg.gate.threshold):
        reason = "clear_accept"
    elif accepted:
        reason = "accepted_uncertain_band"
    else:
        reason = "uncertain_below_threshold"

    detail.update(
        {
            "score": score,
            "accepted": accepted,
            "reason": reason,
            "backend_scores": backend_scores,
            "segment_similarities": segment_similarities,
            "errors": errors,
        }
    )
    return score, detail


def collect_scores_cached(
    trials: list[SpeakerTrial],
    config_path: str = "",
    progress_every: int = 100,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], SpeakerEmbeddingCache, float]:
    cfg = apply_env_overrides(load_config(config_path or None))
    cache = SpeakerEmbeddingCache(cfg)
    scores: list[float] = []
    labels: list[int] = []
    details: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    for i, trial in enumerate(trials, start=1):
        if not Path(trial.enroll_audio).is_file() or not Path(trial.test_audio).is_file():
            details.append(
                {
                    "id": trial.trial_id,
                    "enroll_audio": trial.enroll_audio,
                    "test_audio": trial.test_audio,
                    "label": int(trial.label),
                    "error": "missing_audio",
                }
            )
            continue
        score, detail = _score_trial(trial, cache)
        scores.append(score)
        labels.append(int(trial.label))
        details.append(detail)

        if progress_every > 0 and (i == 1 or i % progress_every == 0 or i == len(trials)):
            elapsed = time.perf_counter() - t0
            print(
                f"[cached-speaker-eval] {i}/{len(trials)} trials, "
                f"{cache.n_cached_audio} cached audio, {elapsed:.1f}s",
                flush=True,
            )

    return (
        np.asarray(scores, dtype=np.float64),
        np.asarray(labels, dtype=np.int32),
        details,
        cache,
        time.perf_counter() - t0,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Cached speaker verification evaluation/calibration")
    parser.add_argument("--trials", required=True, help="Speaker verification trials csv/jsonl")
    parser.add_argument("--audio-root", type=str, default="")
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--output", type=str, default="output/speaker_eval_cached.json")
    parser.add_argument("--score-dump", type=str, default="")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate at most N trials")
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    trials = load_trials(args.trials, args.audio_root)
    if args.limit > 0:
        trials = trials[: args.limit]

    scores, labels, details, cache, elapsed_sec = collect_scores_cached(
        trials,
        config_path=args.config,
        progress_every=args.progress_every,
    )
    metrics = compute_metrics(scores, labels)
    cfg = cache.cfg
    out = {
        "metrics": asdict(metrics),
        "recommended_gate_threshold": metrics.best_competition_threshold,
        "config": {
            "path": args.config,
            "backends": cache.backend_names(),
            "aggregate": cfg.gate.aggregate,
            "top_k": cfg.gate.top_k,
            "base_threshold": cfg.gate.threshold,
        },
        "runtime": {
            "elapsed_sec": elapsed_sec,
            "avg_sec_per_trial": elapsed_sec / max(1, int(scores.size)),
            "n_cached_audio": cache.n_cached_audio,
            "n_audio_loads": cache.n_audio_loads,
            "n_backend_encodes": cache.n_backend_encodes,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))

    if args.score_dump:
        dump_path = Path(args.score_dump)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
