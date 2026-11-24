"""Pure music21-backed diatonic harmony target generation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite, log2

from music21 import key as music21_key
from music21 import pitch as music21_pitch

from .domain.models import KeyState
from .key_scale_resolver import KeyScale
from .pitch_detector import FMAX_HZ, FMIN_HZ, TrackingState

TARGET_MIN_HZ = FMIN_HZ
TARGET_MAX_HZ = FMAX_HZ
SUPPORTED_VOICES = ("3rd", "5th", "octave")
_INTERVAL_STEPS = {"3rd": 2, "5th": 4, "octave": 7}


class HarmonyStatus(str, Enum):
    VALID = "valid"
    MISSING_FUNDAMENTAL = "missing_fundamental"
    UNVOICED = "unvoiced"
    TRACKING_LOST = "tracking_lost"
    NONFINITE_FREQUENCY = "nonfinite_frequency"
    MISSING_KEY = "missing_key"
    AMBIGUOUS_KEY = "ambiguous_key"
    INVALID_SCALE = "invalid_scale"
    INVALID_VOICE = "invalid_voice"


@dataclass(frozen=True)
class HarmonyTarget:
    source_frequency_hz: float | None
    voice: str
    source_pitch: str | None
    source_scale_degree: int | None
    target_pitch: str | None
    target_scale_degree: int | None
    target_frequency_hz: float | None
    key: KeyScale | None
    valid: bool
    was_clamped: bool
    status: HarmonyStatus


class HarmonyGenerator:
    """Generate one deterministic diatonic harmony target."""

    def generate(
        self,
        fundamental_frequency_hz: float | None,
        effective_key: KeyScale | KeyState | None,
        voice: str,
        *,
        voiced: bool = True,
        tracking_state: TrackingState = TrackingState.TRACKING,
        key_ambiguous: bool = False,
    ) -> HarmonyTarget:
        if fundamental_frequency_hz is None:
            return self._invalid(voice, HarmonyStatus.MISSING_FUNDAMENTAL, effective_key)
        if not isfinite(fundamental_frequency_hz):
            return self._invalid(voice, HarmonyStatus.NONFINITE_FREQUENCY, effective_key)
        if not voiced:
            return self._invalid(voice, HarmonyStatus.UNVOICED, effective_key)
        if tracking_state == TrackingState.TRACKING_LOST:
            return self._invalid(voice, HarmonyStatus.TRACKING_LOST, effective_key)
        if effective_key is None:
            return self._invalid(voice, HarmonyStatus.MISSING_KEY, None)
        key_value = self._coerce_key(effective_key)
        if key_value is None:
            return self._invalid(voice, HarmonyStatus.INVALID_SCALE, None)
        if key_ambiguous:
            return self._invalid(voice, HarmonyStatus.AMBIGUOUS_KEY, key_value)
        if voice not in _INTERVAL_STEPS:
            return self._invalid(voice, HarmonyStatus.INVALID_VOICE, key_value)
        if fundamental_frequency_hz <= 0.0:
            return self._invalid(voice, HarmonyStatus.NONFINITE_FREQUENCY, key_value)

        try:
            musical_key = music21_key.Key(key_value.root, key_value.scale_type)
            scale = musical_key.getScale()
            nearest_midi = int(round(self._frequency_to_midi(fundamental_frequency_hz)))
            source_pitch = music21_pitch.Pitch(midi=nearest_midi)
            scale_pitches = scale.getPitches(
                music21_pitch.Pitch(midi=0),
                music21_pitch.Pitch(midi=127),
            )
            nearest_scale_index = min(
                range(len(scale_pitches)),
                key=lambda index: (abs(scale_pitches[index].midi - nearest_midi), scale_pitches[index].midi),
            )
            source_scale_pitch = scale_pitches[nearest_scale_index]
            source_degree = int(scale.getScaleDegreeFromPitch(source_scale_pitch))
            target_index = nearest_scale_index + _INTERVAL_STEPS[voice]
            if target_index >= len(scale_pitches):
                raise ValueError("Target harmony pitch exceeds supported MIDI range.")
            target_scale_pitch = scale_pitches[target_index]
            cents = (self._frequency_to_midi(fundamental_frequency_hz) - nearest_midi) * 100.0
            target_midi = float(target_scale_pitch.midi) + cents / 100.0
            target_frequency = self._midi_to_frequency(target_midi)
            if not isfinite(target_frequency):
                return self._invalid(voice, HarmonyStatus.NONFINITE_FREQUENCY, key_value)
            clamped_frequency = min(TARGET_MAX_HZ, max(TARGET_MIN_HZ, target_frequency))
            return HarmonyTarget(
                source_frequency_hz=float(fundamental_frequency_hz),
                voice=voice,
                source_pitch=source_scale_pitch.nameWithOctave,
                source_scale_degree=source_degree,
                target_pitch=target_scale_pitch.nameWithOctave,
                target_scale_degree=((source_degree - 1 + _INTERVAL_STEPS[voice]) % 7) + 1,
                target_frequency_hz=clamped_frequency,
                key=key_value,
                valid=True,
                was_clamped=clamped_frequency != target_frequency,
                status=HarmonyStatus.VALID,
            )
        except (TypeError, ValueError, music21_pitch.PitchException):
            return self._invalid(voice, HarmonyStatus.INVALID_SCALE, key_value)

    @staticmethod
    def _coerce_key(value: KeyScale | KeyState) -> KeyScale | None:
        if isinstance(value, KeyScale):
            if value.scale_type not in {"major", "minor"}:
                return None
            return value
        if isinstance(value, KeyState):
            if value.root is None or value.scale_type not in {"major", "minor"}:
                return None
            return KeyScale(value.root, value.scale_type)
        return None

    @staticmethod
    def _frequency_to_midi(frequency_hz: float) -> float:
        return 69.0 + 12.0 * log2(frequency_hz / 440.0)

    @staticmethod
    def _midi_to_frequency(midi: float) -> float:
        return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)

    @staticmethod
    def _invalid(
        voice: str,
        status: HarmonyStatus,
        key: KeyScale | KeyState | None,
    ) -> HarmonyTarget:
        key_value = HarmonyGenerator._coerce_key(key) if key is not None else None
        return HarmonyTarget(
            source_frequency_hz=None,
            voice=voice,
            source_pitch=None,
            source_scale_degree=None,
            target_pitch=None,
            target_scale_degree=None,
            target_frequency_hz=None,
            key=key_value,
            valid=False,
            was_clamped=False,
            status=status,
        )


__all__ = [
    "HarmonyGenerator",
    "HarmonyStatus",
    "HarmonyTarget",
    "SUPPORTED_VOICES",
    "TARGET_MAX_HZ",
    "TARGET_MIN_HZ",
]
