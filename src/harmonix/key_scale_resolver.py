"""Pure live, upload, and manual key/scale resolution."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from math import isfinite, log2
from typing import Iterable

import numpy as np

from .pitch_detector import PitchDetectionResult, TrackingState

CANONICAL_SAMPLE_RATE = 44100
LIVE_WINDOW_OBSERVATIONS = 16
STABILIZATION_OBSERVATIONS = 4
HYSTERESIS_MARGIN = 0.10
KEY_ACCEPTANCE_CONFIDENCE = 0.60
AMBIGUITY_MARGIN = 0.05

_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float64,
)
_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float64,
)
_ROOTS = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


class AnalysisMode(str, Enum):
    LIVE = "live"
    UPLOAD = "upload"
    MANUAL = "manual"


@dataclass(frozen=True)
class KeyScale:
    root: str
    scale_type: str


@dataclass(frozen=True)
class KeyResolutionResult:
    detected_key: KeyScale | None
    stabilized_key: KeyScale | None
    manual_key: KeyScale | None
    effective_key: KeyScale | None
    confidence: float
    ambiguous: bool
    insufficient_evidence: bool
    analysis_mode: AnalysisMode


@dataclass(frozen=True)
class _ScoredCandidate:
    key: KeyScale
    score: float


class KeyScaleResolver:
    """Resolve musical key context from M4 pitch observations."""

    def __init__(
        self,
        *,
        window_size: int = LIVE_WINDOW_OBSERVATIONS,
        stabilization_observations: int = STABILIZATION_OBSERVATIONS,
        hysteresis_margin: float = HYSTERESIS_MARGIN,
        acceptance_confidence: float = KEY_ACCEPTANCE_CONFIDENCE,
        ambiguity_margin: float = AMBIGUITY_MARGIN,
    ) -> None:
        self.window_size = window_size
        self.stabilization_observations = stabilization_observations
        self.hysteresis_margin = hysteresis_margin
        self.acceptance_confidence = acceptance_confidence
        self.ambiguity_margin = ambiguity_margin
        self._observations: deque[tuple[int, float]] = deque(maxlen=window_size)
        self._detected_key: KeyScale | None = None
        self._detected_confidence = 0.0
        self._stabilized_key: KeyScale | None = None
        self._candidate_streak = 0
        self._manual_key: KeyScale | None = None
        self._pending_manual_key: KeyScale | None = None
        self._mode = AnalysisMode.LIVE

    def observe(self, result: PitchDetectionResult) -> KeyResolutionResult:
        """Consume one M4 result and update live key stabilization."""
        if self._is_usable(result):
            pitch_class = self._frequency_to_pitch_class(result.frequency_hz)
            if pitch_class is not None:
                self._observations.append((pitch_class, float(result.confidence)))
        self._mode = AnalysisMode.MANUAL if self._manual_key is not None or self._pending_manual_key is not None else AnalysisMode.LIVE
        self._refresh_live_result()
        return self.result()

    def observe_pitch(
        self,
        frequency_hz: float | None,
        confidence: float,
        *,
        voiced: bool = True,
        tracking_state: TrackingState = TrackingState.TRACKING,
    ) -> KeyResolutionResult:
        """Convenience boundary for deterministic callers and M4-compatible data."""
        return self.observe(
            PitchDetectionResult(
                frequency_hz=frequency_hz,
                confidence=confidence,
                voiced=voiced,
                tracking_state=tracking_state,
            )
        )

    def analyze_upload(self, observations: Iterable[PitchDetectionResult]) -> KeyResolutionResult:
        """Resolve one key from confidence-weighted evidence across a complete track."""
        evidence: list[tuple[int, float]] = []
        for observation in observations:
            if not self._is_usable(observation):
                continue
            pitch_class = self._frequency_to_pitch_class(observation.frequency_hz)
            if pitch_class is not None:
                evidence.append((pitch_class, float(observation.confidence)))

        result = self._resolve_evidence(evidence, AnalysisMode.UPLOAD)
        if self._manual_key is not None or self._pending_manual_key is not None:
            result = self._with_manual_key(result, AnalysisMode.MANUAL)
        return result

    def set_manual_key(self, root: str, scale_type: str) -> None:
        """Queue a manual key for activation at the next processing boundary."""
        self._pending_manual_key = self._validate_key(root, scale_type)
        self._mode = AnalysisMode.MANUAL

    def apply_processing_boundary(self) -> KeyResolutionResult:
        """Apply a queued manual key at an explicit processing boundary."""
        if self._pending_manual_key is not None:
            self._manual_key = self._pending_manual_key
            self._pending_manual_key = None
        self._mode = AnalysisMode.MANUAL if self._manual_key is not None else AnalysisMode.LIVE
        return self.result()

    def clear_manual_key(self) -> KeyResolutionResult:
        """Queue automatic mode; the change takes effect at the next boundary."""
        self._pending_manual_key = None
        self._manual_key = None
        self._mode = AnalysisMode.LIVE
        return self.result()

    def result(self) -> KeyResolutionResult:
        mode = AnalysisMode.MANUAL if self._manual_key is not None else self._mode
        effective = self._manual_key if self._manual_key is not None else self._stabilized_key
        is_manual = self._manual_key is not None
        ambiguous = (self._detected_key is None and bool(self._observations)) if not is_manual else False
        insufficient = (self._detected_key is None and not ambiguous) if not is_manual else False
        return KeyResolutionResult(
            detected_key=self._detected_key,
            stabilized_key=self._stabilized_key,
            manual_key=self._manual_key,
            effective_key=effective,
            confidence=self._detected_confidence,
            ambiguous=ambiguous,
            insufficient_evidence=insufficient,
            analysis_mode=mode,
        )

    def _refresh_live_result(self) -> None:
        resolved = self._resolve_evidence(list(self._observations), AnalysisMode.LIVE)
        self._detected_key = resolved.detected_key
        self._detected_confidence = resolved.confidence
        if self._detected_key is None:
            self._candidate_streak = 0
            return
        if self._stabilized_key == self._detected_key:
            self._candidate_streak = 0
            return
        if self._stabilized_key is not None:
            stabilized_score = self._score_for_key(list(self._observations), self._stabilized_key)
            if resolved.confidence - stabilized_score < self.hysteresis_margin:
                self._candidate_streak = 0
                return
        self._candidate_streak += 1
        if self._candidate_streak >= self.stabilization_observations:
            self._stabilized_key = self._detected_key
            self._candidate_streak = 0

    def _resolve_evidence(
        self,
        evidence: list[tuple[int, float]],
        mode: AnalysisMode,
    ) -> KeyResolutionResult:
        if not evidence:
            return KeyResolutionResult(
                detected_key=None,
                stabilized_key=None,
                manual_key=None,
                effective_key=None,
                confidence=0.0,
                ambiguous=False,
                insufficient_evidence=True,
                analysis_mode=mode,
            )
        histogram = self._histogram(evidence)
        candidates = self._rank_candidates(histogram)
        best = candidates[0]
        second = candidates[1]
        ambiguous = best.score - second.score < self.ambiguity_margin
        accepted = best.score >= self.acceptance_confidence and not ambiguous
        detected = best.key if accepted else None
        return KeyResolutionResult(
            detected_key=detected,
            stabilized_key=detected if mode == AnalysisMode.UPLOAD else None,
            manual_key=None,
            effective_key=detected,
            confidence=best.score,
            ambiguous=ambiguous,
            insufficient_evidence=False,
            analysis_mode=mode,
        )

    def _rank_candidates(self, histogram: np.ndarray) -> list[_ScoredCandidate]:
        candidates: list[_ScoredCandidate] = []
        for root in range(12):
            for scale_type, profile in (("major", _MAJOR_PROFILE), ("minor", _MINOR_PROFILE)):
                rotated = np.roll(profile, root)
                correlation = float(np.corrcoef(histogram, rotated)[0, 1])
                if not isfinite(correlation):
                    score = 0.0
                else:
                    score = max(0.0, min(1.0, (correlation + 1.0) / 2.0))
                candidates.append(_ScoredCandidate(KeyScale(_ROOTS[root], scale_type), score))
        return sorted(candidates, key=lambda candidate: (-candidate.score, candidate.key.root, candidate.key.scale_type))

    def _score_for_key(self, evidence: list[tuple[int, float]], key: KeyScale) -> float:
        histogram = self._histogram(evidence)
        profile = _MAJOR_PROFILE if key.scale_type == "major" else _MINOR_PROFILE
        correlation = float(np.corrcoef(histogram, np.roll(profile, _ROOTS.index(key.root)))[0, 1])
        if not isfinite(correlation):
            return 0.0
        return max(0.0, min(1.0, (correlation + 1.0) / 2.0))

    @staticmethod
    def _histogram(evidence: list[tuple[int, float]]) -> np.ndarray:
        histogram = np.zeros(12, dtype=np.float64)
        for pitch_class, weight in evidence:
            histogram[pitch_class] += weight
        total = float(histogram.sum())
        if total <= 0.0:
            return histogram
        return histogram / total

    def _with_manual_key(self, result: KeyResolutionResult, mode: AnalysisMode) -> KeyResolutionResult:
        manual = self._manual_key if self._manual_key is not None else self._pending_manual_key
        is_manual = manual is not None
        return KeyResolutionResult(
            detected_key=result.detected_key,
            stabilized_key=result.stabilized_key,
            manual_key=manual,
            effective_key=manual if is_manual else result.effective_key,
            confidence=result.confidence,
            ambiguous=False if is_manual else result.ambiguous,
            insufficient_evidence=False if is_manual else result.insufficient_evidence,
            analysis_mode=mode,
        )

    @staticmethod
    def _is_usable(result: PitchDetectionResult) -> bool:
        return (
            result.voiced
            and result.tracking_state == TrackingState.TRACKING
            and result.frequency_hz is not None
            and isfinite(result.frequency_hz)
            and isfinite(result.confidence)
            and result.confidence > 0.0
        )

    @staticmethod
    def _frequency_to_pitch_class(frequency_hz: float | None) -> int | None:
        if frequency_hz is None or not isfinite(frequency_hz) or frequency_hz <= 0.0:
            return None
        midi = 69.0 + 12.0 * log2(frequency_hz / 440.0)
        if not isfinite(midi):
            return None
        return int(round(midi)) % 12

    @staticmethod
    def _validate_key(root: str, scale_type: str) -> KeyScale:
        normalized_root = root.strip()
        if normalized_root not in _ROOTS:
            raise ValueError(f"Unsupported key root: {root!r}")
        if scale_type not in {"major", "minor"}:
            raise ValueError(f"Unsupported scale type: {scale_type!r}")
        return KeyScale(normalized_root, scale_type)


__all__ = [
    "AMBIGUITY_MARGIN",
    "AnalysisMode",
    "HYSTERESIS_MARGIN",
    "KEY_ACCEPTANCE_CONFIDENCE",
    "KeyResolutionResult",
    "KeyScale",
    "KeyScaleResolver",
    "LIVE_WINDOW_OBSERVATIONS",
    "STABILIZATION_OBSERVATIONS",
]
