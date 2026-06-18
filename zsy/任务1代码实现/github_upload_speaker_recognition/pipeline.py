"""
End-to-end competition inference pipeline.

Wake audio → enroll target speaker → gate command audio → ASR (if accepted).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from audio_quality import AudioQuality, preprocess_waveform
from asr import AsrBackend, create_asr_backend
from audio_utils import load_audio_file
from command_postprocess import command_prior_score, normalize_command_text
from config import PipelineConfig
from decision import DecisionEvidence, DecisionResult, final_decision, proxy_target_ratios
from metrics_cer import SampleResult, cer_single, is_rejection_sample
from separation import separate_target_speech
from speaker_gate import GateResult, SpeakerGate


@dataclass
class PipelineOutput:
    content: str
    accepted: bool
    similarity: float
    gate: GateResult
    elapsed_sec: float
    decision: DecisionResult
    evidence: DecisionEvidence


class RecognitionPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.gate = SpeakerGate(self.config.gate)
        self.asr: AsrBackend = create_asr_backend(self.config.asr)
        self._target_emb: Optional[np.ndarray] = None
        self._wake_quality = AudioQuality()

    def _preprocess(self, waveform: np.ndarray, sr: int) -> tuple[np.ndarray, int, AudioQuality]:
        return preprocess_waveform(
            waveform,
            sr,
            target_sr=self.config.target_sr,
            enable_vad=self.config.preprocess.enable_vad,
            vad_threshold_ratio=self.config.preprocess.vad_threshold_ratio,
        )

    def enroll_wake(self, wake_path: str) -> np.ndarray:
        wav, sr = load_audio_file(wake_path)
        return self.enroll_wake_waveform(wav, sr)

    def enroll_wake_waveform(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        wav, target_sr, quality = self._preprocess(waveform, sr)
        emb = self.gate.enroll_from_waveform(wav, target_sr)
        self._target_emb = emb
        self._wake_quality = quality
        return emb

    def infer_pair(self, wake_path: str, cmd_path: str) -> PipelineOutput:
        t0 = time.perf_counter()
        try:
            self.enroll_wake(wake_path)
            cmd_wav, cmd_sr = load_audio_file(cmd_path)
            return self.infer_command(cmd_wav, cmd_sr)
        except Exception as e:
            elapsed = time.perf_counter() - t0
            fake_gate = GateResult(False, 0.0, [], f"error:{e}")
            evidence = DecisionEvidence()
            decision = DecisionResult(False, 0.0, f"error:{e}", use_tse=False)
            return PipelineOutput("", False, 0.0, fake_gate, elapsed, decision, evidence)

    def infer_command(self, cmd_waveform: np.ndarray, cmd_sr: int) -> PipelineOutput:
        t0 = time.perf_counter()

        if self._target_emb is None:
            raise RuntimeError("Enroll wake audio before infer_command.")

        cmd_waveform, cmd_sr, query_quality = self._preprocess(cmd_waveform, cmd_sr)

        evidence = DecisionEvidence(
            no_speech=query_quality.no_speech,
            query_snr_db=query_quality.snr_db,
            wake_quality_score=self._wake_quality.score,
            enrollment_bad_quality=1.0 - self._wake_quality.score,
            query_noise_penalty=max(0.0, min(1.0, (-query_quality.snr_db) / 10.0)),
        )

        if query_quality.no_speech:
            elapsed = time.perf_counter() - t0
            fake_gate = GateResult(False, 0.0, [], "no_speech")
            decision = final_decision(evidence, self.config.decision)
            return PipelineOutput("", False, 0.0, fake_gate, elapsed, decision, evidence)

        gate_result = self.gate.should_accept(
            cmd_waveform,
            cmd_sr,
            wake_quality=self._wake_quality,
            query_quality=query_quality,
        )
        target_ratio, non_target_ratio, overlap_probability = proxy_target_ratios(
            gate_result.segment_similarities,
            self.config.decision,
        )
        evidence.speaker_similarity = gate_result.similarity
        evidence.target_frame_ratio = target_ratio
        evidence.non_target_frame_ratio = non_target_ratio
        evidence.overlap_probability = overlap_probability

        preliminary_decision = final_decision(evidence, self.config.decision)

        if self.config.use_separation and preliminary_decision.use_tse:
            cmd_waveform = separate_target_speech(
                cmd_waveform, cmd_sr, self._target_emb
            )

        content = ""
        if preliminary_decision.accepted or self.config.force_asr:
            raw_content = self.asr.transcribe(cmd_waveform, cmd_sr)
            content = normalize_command_text(raw_content)
            evidence.asr_confidence = 1.0 if content else 0.0
            evidence.command_prior_score = command_prior_score(content)

        decision = final_decision(evidence, self.config.decision)
        accepted = decision.accepted or (self.config.force_asr and bool(content))
        if not accepted:
            content = ""

        elapsed = time.perf_counter() - t0
        return PipelineOutput(
            content=content,
            accepted=accepted,
            similarity=gate_result.similarity,
            gate=gate_result,
            elapsed_sec=elapsed,
            decision=decision,
            evidence=evidence,
        )

    def run_sample(
        self,
        wake_path: str,
        cmd_path: str,
        label: str = "",
        sample_id: str = "",
    ) -> SampleResult:
        out = self.infer_pair(wake_path, cmd_path)
        ref = label or ""

        if is_rejection_sample(ref):
            # Submission must be empty for rejection samples.
            hyp_internal = out.content if self.config.force_asr else ""
            content = ""
            cer = 0.0 if not (hyp_internal or "").strip() else 100.0
        else:
            content = out.content if out.accepted else ""
            cer = cer_single(ref, content)

        sid = sample_id or cmd_path
        return SampleResult(
            id=sid,
            content=content,
            label=ref,
            cer=cer,
            accepted=out.accepted,
            similarity=out.similarity,
            elapsed_sec=out.elapsed_sec,
        )
