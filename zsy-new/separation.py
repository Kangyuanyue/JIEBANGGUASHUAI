"""
Optional target-speaker extraction for overlapping speech.

Stub implementation: passthrough. Replace with SpEx+/SepFormer when
training data with overlap labels is available.
"""

from __future__ import annotations

import numpy as np


def separate_target_speech(
    waveform: np.ndarray,
    sr: int,
    target_embedding: np.ndarray,
) -> np.ndarray:
    """
    Placeholder — returns input unchanged.

    Future: SpEx+, TD-SpeakerConv-TSE, or SpeechBrain SepFormer conditioned
    on target_embedding.
    """
    return waveform
