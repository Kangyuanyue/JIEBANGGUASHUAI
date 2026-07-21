"""Optional target-speaker extraction for overlapping speech."""

from __future__ import annotations

import numpy as np

from tse_model import PositiveNegativeTSE, get_default_tse_model


def separate_target_speech(
    waveform: np.ndarray,
    sr: int,
    target_embedding: np.ndarray | None = None,
    *,
    positive_enrollment: np.ndarray | None = None,
    negative_enrollment: np.ndarray | None = None,
    enrollment_sr: int = 16000,
    model: PositiveNegativeTSE | None = None,
) -> np.ndarray:
    """Extract target speech when positive/negative enrollments are available.

    Calls made by older code with only a target embedding remain passthrough;
    the Pos/Neg TSE model requires waveform enrollments rather than a generic
    speaker-verification embedding.
    """
    if positive_enrollment is None or negative_enrollment is None:
        return np.asarray(waveform, dtype=np.float32)
    extractor = model or get_default_tse_model()
    result = extractor.extract(
        mixture=waveform,
        mixture_sr=sr,
        positive_enrollment=positive_enrollment,
        positive_sr=enrollment_sr,
        negative_enrollment=negative_enrollment,
        negative_sr=enrollment_sr,
    )
    return result.waveform
