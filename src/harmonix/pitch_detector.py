"""Pure pYIN pitch detection for canonical live-audio frames."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any

import librosa
import numpy as np

CANONICAL_SAMPLE_RATE = 44100
FRAME_SIZE = 2048
FMIN_HZ = 65.0
FMAX_HZ = 1000.0
CONFIDENCE_THRESHOLD = 0.5


class TrackingState(str, Enum):
    TRACKING = "tracking"
    TRACKING_LOST = "tracking_lost"


@dataclass(frozen=True)
class PitchDetectionResult:
    frequency_hz: float | None
    confidence: float
    voiced: bool
    tracking_state: TrackingState


class PitchDetector:
    """Detect one canonical audio frame with librosa.pyin."""

    def __init__(
        self,
        *,
        sample_rate: int = CANONICAL_SAMPLE_RATE,
        frame_size: int = FRAME_SIZE,
        fmin_hz: float = FMIN_HZ,
        fmax_hz: float = FMAX_HZ,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.fmin_hz = fmin_hz
        self.fmax_hz = fmax_hz
        self.confidence_threshold = confidence_threshold

    def detect(self, frame: np.ndarray) -> PitchDetectionResult:
        audio = np.asarray(frame, dtype=np.float32)
        if audio.ndim != 1 or audio.size != self.frame_size:
            raise ValueError(f"Expected a mono frame of {self.frame_size} samples.")
        if not np.isfinite(audio).all():
            return self._unvoiced_result()

        f0, voiced_flag, voiced_prob = librosa.pyin(
            audio,
            fmin=self.fmin_hz,
            fmax=self.fmax_hz,
            sr=self.sample_rate,
            frame_length=self.frame_size,
            hop_length=self.frame_size // 4,
            fill_na=np.nan,
        )
        return self._interpret_pyin(f0, voiced_flag, voiced_prob)

    def _interpret_pyin(
        self,
        f0: Any,
        voiced_flag: Any,
        voiced_prob: Any,
    ) -> PitchDetectionResult:
        frequencies = np.asarray(f0, dtype=np.float64).reshape(-1)
        flags = np.asarray(voiced_flag, dtype=bool).reshape(-1)
        probabilities = np.asarray(voiced_prob, dtype=np.float64).reshape(-1)
        count = min(frequencies.size, flags.size, probabilities.size)
        if count == 0:
            return self._unvoiced_result()

        frequencies = frequencies[:count]
        flags = flags[:count]
        probabilities = probabilities[:count]
        valid = (
            flags
            & np.isfinite(frequencies)
            & np.isfinite(probabilities)
            & (probabilities >= self.confidence_threshold)
            & (frequencies >= self.fmin_hz)
            & (frequencies <= self.fmax_hz)
        )
        if not np.any(valid):
            return self._unvoiced_result()

        usable_frequencies = frequencies[valid]
        usable_probabilities = probabilities[valid]
        frequency = float(np.median(usable_frequencies))
        confidence = float(np.median(usable_probabilities))
        if not isfinite(frequency) or not isfinite(confidence):
            return self._unvoiced_result()
        if not self.fmin_hz <= frequency <= self.fmax_hz:
            return self._unvoiced_result()

        return PitchDetectionResult(
            frequency_hz=frequency,
            confidence=confidence,
            voiced=True,
            tracking_state=TrackingState.TRACKING,
        )

    @staticmethod
    def _unvoiced_result() -> PitchDetectionResult:
        return PitchDetectionResult(
            frequency_hz=None,
            confidence=0.0,
            voiced=False,
            tracking_state=TrackingState.TRACKING_LOST,
        )


__all__ = [
    "CANONICAL_SAMPLE_RATE",
    "CONFIDENCE_THRESHOLD",
    "FMAX_HZ",
    "FMIN_HZ",
    "FRAME_SIZE",
    "PitchDetectionResult",
    "PitchDetector",
    "TrackingState",
]
