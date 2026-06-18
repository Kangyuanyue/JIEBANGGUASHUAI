"""Competition pipeline configuration."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class GateConfig:
    """Target-speaker verification (wake-audio enrollment)."""

    threshold: float = 0.65
    speaker_reject_low: float = 0.23909343270879338
    speaker_accept_high: float = 0.37844391932749266
    margin: float = 0.05
    segment_sec: float = 3.0
    num_segments: int = 5
    aggregate: str = "topk_mean"  # max | mean | median | topk_mean
    top_k: int = 2
    min_duration_sec: float = 0.5
    embedding_backends: list[str] = field(default_factory=lambda: ["ecapa"])
    backend_weights: Dict[str, float] = field(default_factory=dict)
    wavlm_model_name: str = "microsoft/wavlm-base-plus-sv"
    campplus_model_id: str = "iic/speech_campplus_sv_zh-cn_16k-common"
    eres2netv2_model_id: str = "iic/speech_eres2netv2_sv_zh-cn_16k-common"
    modelscope_local_dir: str = "pretrained"
    dynamic_threshold: bool = True
    low_quality_threshold_boost: float = 0.05
    low_snr_threshold_boost: float = -0.035
    min_quality_for_base_threshold: float = 0.75
    min_snr_for_base_threshold_db: float = 5.0


@dataclass
class PreprocessConfig:
    """Lightweight audio preprocessing before speaker gate / ASR."""

    enable_vad: bool = True
    vad_threshold_ratio: float = 0.10


@dataclass
class AsrConfig:
    """Automatic speech recognition."""

    backend: str = "funasr"  # funasr | mock
    model_name: str = "paraformer-zh"
    model_dir: str = ""  # empty → FunASR hub default
    device: str = "auto"  # auto | cpu | cuda
    use_fp16: bool = True
    # Post-processing: strip wake phrase if echoed in command audio
    strip_punctuation: bool = True


@dataclass
class DecisionConfig:
    """Fusion thresholds for accept/reject and hard-case routing."""

    speaker_accept_high: float = 0.72
    speaker_reject_low: float = 0.45
    min_target_ratio: float = 0.18
    max_target_ratio_for_reject: float = 0.08
    overlap_trigger_threshold: float = 0.35
    low_snr_threshold_db: float = -2.0
    target_low_snr_ratio: float = 0.12
    min_asr_confidence: float = 0.35
    fusion_accept_threshold: float = 0.50
    weight_speaker_similarity: float = 1.20
    weight_target_frame_ratio: float = 0.90
    weight_non_target_frame_ratio: float = -0.70
    weight_asr_confidence: float = 0.40
    weight_command_prior_score: float = 0.25
    weight_enrollment_bad_quality: float = -0.30
    weight_query_noise_penalty: float = -0.20


@dataclass
class PipelineConfig:
    """Top-level settings."""

    target_sr: int = 16000
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)
    # When True, run ASR even if gate fails (debug only)
    force_asr: bool = False
    # Optional separation stage (stub until dataset arrives)
    use_separation: bool = False

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineConfig":
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        preprocess_raw = raw.pop("preprocess", {})
        gate_raw = raw.pop("gate", {})
        asr_raw = raw.pop("asr", {})
        decision_raw = raw.pop("decision", {})
        preprocess = PreprocessConfig(**{k: v for k, v in preprocess_raw.items() if k in PreprocessConfig.__dataclass_fields__})
        gate = GateConfig(**{k: v for k, v in gate_raw.items() if k in GateConfig.__dataclass_fields__})
        asr = AsrConfig(**{k: v for k, v in asr_raw.items() if k in AsrConfig.__dataclass_fields__})
        decision = DecisionConfig(**{k: v for k, v in decision_raw.items() if k in DecisionConfig.__dataclass_fields__})
        top_keys = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        return cls(preprocess=preprocess, gate=gate, asr=asr, decision=decision, **top_keys)

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)


def default_config_path() -> Path:
    return Path(__file__).resolve().parent / "configs" / "default.json"


def load_config(path: Optional[str | Path] = None) -> PipelineConfig:
    cfg_path = Path(path) if path else default_config_path()
    if cfg_path.is_file():
        return PipelineConfig.from_json(cfg_path)
    return PipelineConfig()


def apply_env_overrides(cfg: PipelineConfig) -> PipelineConfig:
    """Allow quick tuning without editing JSON."""
    if v := os.getenv("COMP_GATE_THRESHOLD"):
        cfg.gate.threshold = float(v)
    if v := os.getenv("COMP_GATE_MARGIN"):
        cfg.gate.margin = float(v)
    if v := os.getenv("COMP_GATE_AGGREGATE"):
        cfg.gate.aggregate = v.strip().lower()
    if v := os.getenv("COMP_GATE_TOP_K"):
        cfg.gate.top_k = int(v)
    if v := os.getenv("COMP_GATE_BACKENDS"):
        cfg.gate.embedding_backends = [x.strip().lower() for x in v.split(",") if x.strip()]
    if v := os.getenv("COMP_ASR_BACKEND"):
        cfg.asr.backend = v.strip().lower()
    if v := os.getenv("COMP_ASR_MODEL"):
        cfg.asr.model_name = v.strip()
    if v := os.getenv("ECAPA_SAVEDIR"):
        os.environ.setdefault("ECAPA_SAVEDIR", v)
    return cfg
