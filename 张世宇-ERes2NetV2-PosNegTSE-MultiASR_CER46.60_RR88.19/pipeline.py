"""
End-to-end competition inference pipeline.

Wake audio → enroll target speaker → gate command audio → ASR (if accepted).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from audio_quality import AudioQuality, preprocess_waveform
from asr import AsrBackend, create_asr_backend
from asr_consensus import select_consensus_text, should_select_tse_text
from audio_utils import load_audio_file
from command_postprocess import command_prior_score, normalize_command_text
from config import AsrConfig, PipelineConfig
from decision import DecisionEvidence, DecisionResult, final_decision, proxy_target_ratios
from metrics_cer import SampleResult, cer_single, is_rejection_sample
from tse_model import EnrollmentPair, PositiveNegativeTSE, build_pseudo_enrollments
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
    raw_content: str = ""
    tse_content: str = ""
    tse_similarity: float = 0.0
    selected_audio_path: str = "raw"
    asr_candidates: dict[str, str] = field(default_factory=dict)


class RecognitionPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.gate = SpeakerGate(self.config.gate)
        self.asr: AsrBackend = create_asr_backend(self.config.asr)
        self.secondary_asr: Optional[AsrBackend] = None
        if self.config.asr_ensemble.enabled:
            secondary_config = AsrConfig(
                backend="funasr",
                model_name=self.config.asr_ensemble.secondary_model_name,
                model_dir=self.config.asr_ensemble.secondary_model_dir,
                device=self.config.asr_ensemble.secondary_device,
                strip_punctuation=self.config.asr.strip_punctuation,
            )
            self.secondary_asr = create_asr_backend(secondary_config)
        self._target_emb: Optional[np.ndarray] = None
        self._wake_quality = AudioQuality()
        self._tse_enrollment: Optional[EnrollmentPair] = None
        self._tse_model: Optional[PositiveNegativeTSE] = None

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
        if self.config.use_separation:
            self._tse_enrollment = build_pseudo_enrollments(waveform, sr)
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

        original_cmd = np.asarray(cmd_waveform, dtype=np.float32)
        original_cmd_sr = int(cmd_sr)
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

        content = ""
        raw_content = ""
        tse_content = ""
        tse_similarity = 0.0
        selected_audio_path = "raw"
        asr_candidates: dict[str, str] = {}
        can_run_asr = gate_result.accepted or self.config.force_asr
        if can_run_asr:
            if self.config.asr_ensemble.enabled and self.secondary_asr is not None:
                vad_waveform, vad_sr, _ = preprocess_waveform(
                    original_cmd,
                    original_cmd_sr,
                    target_sr=self.config.target_sr,
                    enable_vad=True,
                    vad_threshold_ratio=self.config.asr_ensemble.vad_threshold_ratio,
                )
                para_vad = normalize_command_text(self.asr.transcribe(vad_waveform, vad_sr))
                para_full = normalize_command_text(self.asr.transcribe(cmd_waveform, cmd_sr))
                sense_vad = normalize_command_text(
                    self.secondary_asr.transcribe(vad_waveform, vad_sr)
                )
                asr_candidates.update(
                    {"para_vad": para_vad, "para_full": para_full, "sense_vad": sense_vad}
                )
                content, _, _ = select_consensus_text(
                    list(asr_candidates.values()),
                    weird_weight=self.config.asr_ensemble.weird_weight,
                    command_weight=self.config.asr_ensemble.command_weight,
                )
                raw_content = content
            else:
                raw_content = normalize_command_text(self.asr.transcribe(cmd_waveform, cmd_sr))
                asr_candidates["primary"] = raw_content
                content = raw_content

        if (
            can_run_asr
            and self.config.use_separation
            and preliminary_decision.use_tse
            and self._tse_enrollment is not None
        ):
            if self._tse_model is None:
                self._tse_model = PositiveNegativeTSE(
                    checkpoint=self.config.separation.checkpoint,
                    device=self.config.separation.device,
                )
            tse_result = self._tse_model.extract(
                mixture=cmd_waveform,
                mixture_sr=cmd_sr,
                positive_enrollment=self._tse_enrollment.positive,
                positive_sr=self._tse_enrollment.sample_rate,
                negative_enrollment=self._tse_enrollment.negative,
                negative_sr=self._tse_enrollment.sample_rate,
            )
            tse_gate = self.gate.score_waveform(
                tse_result.waveform,
                tse_result.sample_rate,
                wake_quality=self._wake_quality,
                query_quality=query_quality,
            )
            tse_similarity = tse_gate.similarity
            tse_content = normalize_command_text(
                self.asr.transcribe(tse_result.waveform, tse_result.sample_rate)
            )
            asr_candidates["tse"] = tse_content
            similarity_gain = tse_similarity - gate_result.similarity
            route_ok = should_select_tse_text(
                baseline_text=content,
                tse_text=tse_content,
                raw_candidates=[text for name, text in asr_candidates.items() if name != "tse"],
                similarity_gain=similarity_gain,
                min_similarity_gain=self.config.separation.min_similarity_gain,
                max_text_distance_ratio=self.config.separation.max_text_distance_ratio,
                require_command_prior_not_worse=(
                    self.config.separation.require_command_prior_not_worse
                ),
            )
            if tse_similarity >= self.config.separation.min_output_similarity and route_ok:
                content = tse_content
                selected_audio_path = "tse"

        if can_run_asr:
            evidence.asr_confidence = 1.0 if content else 0.0
            evidence.command_prior_score = command_prior_score(content)

        decision = final_decision(evidence, self.config.decision)
        accepted = (gate_result.accepted and decision.accepted) or (self.config.force_asr and bool(content))
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
            raw_content=raw_content,
            tse_content=tse_content,
            tse_similarity=tse_similarity,
            selected_audio_path=selected_audio_path,
            asr_candidates=asr_candidates,
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
            content = out.content if out.accepted else ""
            cer = 0.0 if not content.strip() else 100.0
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
