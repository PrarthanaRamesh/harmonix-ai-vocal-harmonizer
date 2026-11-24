"""pyworld-backed pitch shifting for full buffers and live segments."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any

import numpy as np
import pyworld

from .pitch_detector import CANONICAL_SAMPLE_RATE, TrackingState

SAMPLE_RATE = CANONICAL_SAMPLE_RATE
FRAME_PERIOD_MS = 5.0
F0_FLOOR_HZ = 65.0
F0_CEIL_HZ = 1000.0
LIVE_BLOCK_SIZE = 2048
WORLD_SEGMENT_SIZE = 8192
OVERLAP_SIZE = 4096
HOP_SIZE = 4096
MIN_SHIFT_RATIO = 0.25
MAX_SHIFT_RATIO = 4.0
SILENCE_PEAK = 1e-8

# Number of input blocks per WORLD segment and per hop (derived from constants above).
_BLOCKS_PER_SEGMENT = WORLD_SEGMENT_SIZE // LIVE_BLOCK_SIZE  # 4
_BLOCKS_PER_HOP = HOP_SIZE // LIVE_BLOCK_SIZE                # 2


class PitchShiftStatus(str, Enum):
    VALID = "valid"
    SILENCE = "silence"
    UNVOICED = "unvoiced"
    MISSING_SOURCE_PITCH = "missing_source_pitch"
    NONFINITE_INPUT = "nonfinite_input"
    NONFINITE_PITCH = "nonfinite_pitch"
    TARGET_OUT_OF_RANGE = "target_out_of_range"
    SHIFT_RATIO_UNSUPPORTED = "shift_ratio_unsupported"
    SAMPLE_COUNT_MISMATCH = "sample_count_mismatch"
    WORLD_ANALYSIS_ERROR = "world_analysis_error"
    WORLD_SYNTHESIS_ERROR = "world_synthesis_error"
    ZERO_SHIFT = "zero_shift"


@dataclass(frozen=True)
class PitchShiftResult:
    audio: np.ndarray | None
    sample_rate: int
    input_sample_count: int
    output_sample_count: int
    source_frequency_hz: float | None
    target_frequency_hz: float | None
    shift_ratio: float | None
    status: PitchShiftStatus
    valid: bool
    duration_preserved: bool
    was_clamped: bool


@dataclass(frozen=True)
class _BlockMeta:
    """Pitch and voicing metadata for one LIVE_BLOCK_SIZE-sample input block.

    Stored in ``PitchShifter._meta_queue`` in lock-step with the audio samples
    held in ``PitchShifter._input_buffer``.  The queue is advanced by
    ``_BLOCKS_PER_HOP`` entries each time the audio buffer is slid by
    ``HOP_SIZE`` samples, so the two structures remain exactly synchronised.
    """

    source_frequency_hz: float | None
    target_frequency_hz: float | None
    source_voiced: bool
    tracking_state: TrackingState
    target_was_clamped: bool


class _WorldCore:
    def __init__(self, world_api: Any = pyworld) -> None:
        self.world = world_api

    # ------------------------------------------------------------------
    # Full-buffer path (upload / off-line): single ratio, unchanged.
    # ------------------------------------------------------------------
    def shift(
        self,
        audio: np.ndarray,
        source_frequency_hz: float | None,
        target_frequency_hz: float | None,
        *,
        source_voiced: bool,
        tracking_state: TrackingState,
        target_was_clamped: bool,
    ) -> PitchShiftResult:
        input_count = int(audio.size)
        base = dict(
            sample_rate=SAMPLE_RATE,
            input_sample_count=input_count,
            source_frequency_hz=source_frequency_hz,
            target_frequency_hz=target_frequency_hz,
            shift_ratio=None,
            was_clamped=target_was_clamped,
        )
        if audio.ndim != 1 or not np.isfinite(audio).all():
            return self._result(None, PitchShiftStatus.NONFINITE_INPUT, False, False, **base)
        if audio.size and float(np.max(np.abs(audio))) <= SILENCE_PEAK:
            return self._result(np.zeros_like(audio, dtype=np.float32), PitchShiftStatus.SILENCE, True, True, **base)
        if source_frequency_hz is None:
            return self._result(None, PitchShiftStatus.MISSING_SOURCE_PITCH, False, False, **base)
        if not isfinite(source_frequency_hz) or source_frequency_hz <= 0.0:
            return self._result(None, PitchShiftStatus.NONFINITE_PITCH, False, False, **base)
        if not source_voiced or tracking_state == TrackingState.TRACKING_LOST:
            return self._result(None, PitchShiftStatus.UNVOICED, False, False, **base)
        if target_frequency_hz is None or not isfinite(target_frequency_hz):
            return self._result(None, PitchShiftStatus.NONFINITE_PITCH, False, False, **base)
        if not F0_FLOOR_HZ <= target_frequency_hz <= F0_CEIL_HZ:
            return self._result(None, PitchShiftStatus.TARGET_OUT_OF_RANGE, False, False, **base)

        ratio = float(target_frequency_hz / source_frequency_hz)
        base["shift_ratio"] = ratio
        if not isfinite(ratio):
            return self._result(None, PitchShiftStatus.NONFINITE_PITCH, False, False, **base)
        if not MIN_SHIFT_RATIO <= ratio <= MAX_SHIFT_RATIO:
            return self._result(None, PitchShiftStatus.SHIFT_RATIO_UNSUPPORTED, False, False, **base)
        status = PitchShiftStatus.ZERO_SHIFT if np.isclose(ratio, 1.0, rtol=0.0, atol=1e-12) else PitchShiftStatus.VALID

        working = np.asarray(audio, dtype=np.float64)
        try:
            f0, temporal_positions = self.world.dio(
                working,
                SAMPLE_RATE,
                f0_floor=F0_FLOOR_HZ,
                f0_ceil=F0_CEIL_HZ,
                frame_period=FRAME_PERIOD_MS,
            )
            refined_f0 = self.world.stonemask(working, f0, temporal_positions, SAMPLE_RATE)
            spectrogram = self.world.cheaptrick(
                working,
                refined_f0,
                temporal_positions,
                SAMPLE_RATE,
                f0_floor=F0_FLOOR_HZ,
            )
            aperiodicity = self.world.d4c(working, refined_f0, temporal_positions, SAMPLE_RATE)
        except Exception:
            return self._result(None, PitchShiftStatus.WORLD_ANALYSIS_ERROR, False, False, **base)

        source_f0 = np.asarray(refined_f0, dtype=np.float64).reshape(-1)
        finite_voiced = np.isfinite(source_f0) & (source_f0 > 0.0)
        target_f0 = np.zeros_like(source_f0)
        target_f0[finite_voiced] = source_f0[finite_voiced] * ratio
        target_f0[~np.isfinite(target_f0)] = 0.0
        if not np.any(target_f0 > 0.0):
            return self._result(None, PitchShiftStatus.UNVOICED, False, False, **base)

        try:
            synthesized = self.world.synthesize(
                target_f0,
                spectrogram,
                aperiodicity,
                SAMPLE_RATE,
                frame_period=FRAME_PERIOD_MS,
            )
        except Exception:
            return self._result(None, PitchShiftStatus.WORLD_SYNTHESIS_ERROR, False, False, **base)
        normalized = np.asarray(synthesized, dtype=np.float64)
        if normalized.ndim != 1:
            return self._result(None, PitchShiftStatus.WORLD_SYNTHESIS_ERROR, False, False, **base)
        if not np.isfinite(normalized).all():
            return self._result(None, PitchShiftStatus.WORLD_SYNTHESIS_ERROR, False, False, **base)
        if normalized.size < input_count:
            return self._result(None, PitchShiftStatus.SAMPLE_COUNT_MISMATCH, False, False, **base)
        # WORLD synthesizes on its frame grid; retain the input temporal interval.
        normalized = normalized[:input_count]
        output = normalized.astype(np.float32)
        output.setflags(write=False)
        return self._result(output, status, True, output.size == input_count, **base)

    # ------------------------------------------------------------------
    # Live streaming path: per-frame ratio mapping from block metadata.
    # ------------------------------------------------------------------
    def shift_live(
        self,
        audio: np.ndarray,
        block_metas: list[_BlockMeta],
    ) -> PitchShiftResult:
        """Pitch-shift a WORLD_SEGMENT_SIZE-sample live segment.

        Each PyWorld analysis frame is mapped to the ``_BlockMeta`` entry
        whose block interval contains the frame centre::

            block_idx = clamp(floor(timeaxis[k] * SAMPLE_RATE / LIVE_BLOCK_SIZE), 0, 3)

        The target f0 for that frame is derived from the matched block's
        ``target_frequency_hz / source_frequency_hz`` ratio, applied to
        PyWorld's own ``refined_f0`` estimate for the frame.  A frame is
        silenced if either the block-level or frame-level voicing guard is
        unmet.

        ``block_metas`` must contain exactly ``_BLOCKS_PER_SEGMENT`` entries
        in temporal order (oldest first), matching the audio already held in
        the caller's ``_input_buffer``.

        The result metadata (``source_frequency_hz``, ``target_frequency_hz``,
        ``shift_ratio``, ``was_clamped``) is taken from the **oldest** two
        entries (the blocks being emitted), not the newest block.
        """
        assert len(block_metas) == _BLOCKS_PER_SEGMENT, (
            f"shift_live expects {_BLOCKS_PER_SEGMENT} block metas, got {len(block_metas)}"
        )

        # Metadata reported in the result: the emitted blocks (oldest 2).
        emitted_meta = block_metas[0]  # representative of emitted portion

        input_count = int(audio.size)
        base = dict(
            sample_rate=SAMPLE_RATE,
            input_sample_count=input_count,
            source_frequency_hz=emitted_meta.source_frequency_hz,
            target_frequency_hz=emitted_meta.target_frequency_hz,
            shift_ratio=None,
            was_clamped=emitted_meta.target_was_clamped,
        )

        if audio.ndim != 1 or not np.isfinite(audio).all():
            return self._result(None, PitchShiftStatus.NONFINITE_INPUT, False, False, **base)

        # Validate that at least one emitted block is usable BEFORE the silence check.
        # This mirrors the original shift() ordering where source_voiced/tracking gates
        # fire before amplitude checks, so unvoiced metadata always yields UNVOICED.
        emitted_usable = any(
            m.source_voiced
            and m.tracking_state != TrackingState.TRACKING_LOST
            and m.source_frequency_hz is not None
            and isfinite(m.source_frequency_hz)
            and m.source_frequency_hz > 0.0
            and m.target_frequency_hz is not None
            and isfinite(m.target_frequency_hz)
            and F0_FLOOR_HZ <= m.target_frequency_hz <= F0_CEIL_HZ
            for m in block_metas[:_BLOCKS_PER_HOP]
        )
        if not emitted_usable:
            return self._result(None, PitchShiftStatus.UNVOICED, False, False, **base)

        if audio.size and float(np.max(np.abs(audio))) <= SILENCE_PEAK:
            return self._result(
                np.zeros_like(audio, dtype=np.float32), PitchShiftStatus.SILENCE, True, True, **base
            )

        working = np.asarray(audio, dtype=np.float64)
        try:
            f0, temporal_positions = self.world.dio(
                working,
                SAMPLE_RATE,
                f0_floor=F0_FLOOR_HZ,
                f0_ceil=F0_CEIL_HZ,
                frame_period=FRAME_PERIOD_MS,
            )
            refined_f0 = self.world.stonemask(working, f0, temporal_positions, SAMPLE_RATE)
            spectrogram = self.world.cheaptrick(
                working,
                refined_f0,
                temporal_positions,
                SAMPLE_RATE,
                f0_floor=F0_FLOOR_HZ,
            )
            aperiodicity = self.world.d4c(working, refined_f0, temporal_positions, SAMPLE_RATE)
        except Exception:
            return self._result(None, PitchShiftStatus.WORLD_ANALYSIS_ERROR, False, False, **base)

        # Build per-frame target_f0 using block-aligned metadata.
        source_f0 = np.asarray(refined_f0, dtype=np.float64).reshape(-1)
        target_f0 = np.zeros_like(source_f0)

        # Pre-compute per-block ratios (None means unvoiced/invalid for that block).
        block_ratios: list[float | None] = []
        for meta in block_metas:
            ratio = _block_ratio(meta)
            block_ratios.append(ratio)

        # Map each frame to its block and apply the corresponding ratio.
        tp = np.asarray(temporal_positions, dtype=np.float64)
        block_indices = np.clip(
            np.floor(tp * SAMPLE_RATE / LIVE_BLOCK_SIZE).astype(int),
            0,
            _BLOCKS_PER_SEGMENT - 1,
        )
        for k in range(len(source_f0)):
            src = source_f0[k]
            if not (isfinite(src) and src > 0.0):
                continue  # PyWorld frame-level unvoiced; leave target_f0[k] = 0
            ratio = block_ratios[block_indices[k]]
            if ratio is None:
                continue  # block-level unvoiced/invalid; leave target_f0[k] = 0
            target_f0[k] = src * ratio

        target_f0[~np.isfinite(target_f0)] = 0.0
        if not np.any(target_f0 > 0.0):
            return self._result(None, PitchShiftStatus.UNVOICED, False, False, **base)

        # Choose representative ratio for result metadata from the emitted blocks.
        representative_ratio: float | None = None
        for m in block_metas[:_BLOCKS_PER_HOP]:
            r = _block_ratio(m)
            if r is not None:
                representative_ratio = r
                break

        base["shift_ratio"] = representative_ratio

        # Determine result status from the emitted-block ratio.
        if representative_ratio is not None and np.isclose(
            representative_ratio, 1.0, rtol=0.0, atol=1e-12
        ):
            status = PitchShiftStatus.ZERO_SHIFT
        else:
            status = PitchShiftStatus.VALID

        try:
            synthesized = self.world.synthesize(
                target_f0,
                spectrogram,
                aperiodicity,
                SAMPLE_RATE,
                frame_period=FRAME_PERIOD_MS,
            )
        except Exception:
            return self._result(None, PitchShiftStatus.WORLD_SYNTHESIS_ERROR, False, False, **base)

        normalized = np.asarray(synthesized, dtype=np.float64)
        if normalized.ndim != 1:
            return self._result(None, PitchShiftStatus.WORLD_SYNTHESIS_ERROR, False, False, **base)
        if not np.isfinite(normalized).all():
            return self._result(None, PitchShiftStatus.WORLD_SYNTHESIS_ERROR, False, False, **base)
        if normalized.size < input_count:
            return self._result(None, PitchShiftStatus.SAMPLE_COUNT_MISMATCH, False, False, **base)
        normalized = normalized[:input_count]
        output = normalized.astype(np.float32)
        output.setflags(write=False)
        return self._result(output, status, True, output.size == input_count, **base)

    @staticmethod
    def _result(
        audio: np.ndarray | None,
        status: PitchShiftStatus,
        valid: bool,
        duration_preserved: bool,
        *,
        sample_rate: int,
        input_sample_count: int,
        source_frequency_hz: float | None,
        target_frequency_hz: float | None,
        shift_ratio: float | None,
        was_clamped: bool,
    ) -> PitchShiftResult:
        return PitchShiftResult(
            audio=PitchShifter._readonly_audio(audio),
            sample_rate=sample_rate,
            input_sample_count=input_sample_count,
            output_sample_count=0 if audio is None else int(audio.size),
            source_frequency_hz=source_frequency_hz,
            target_frequency_hz=target_frequency_hz,
            shift_ratio=shift_ratio,
            status=status,
            valid=valid,
            duration_preserved=duration_preserved,
            was_clamped=was_clamped,
        )


def _block_ratio(meta: _BlockMeta) -> float | None:
    """Return the target/source ratio for a block, or None if unusable."""
    if (
        not meta.source_voiced
        or meta.tracking_state == TrackingState.TRACKING_LOST
        or meta.source_frequency_hz is None
        or not isfinite(meta.source_frequency_hz)
        or meta.source_frequency_hz <= 0.0
        or meta.target_frequency_hz is None
        or not isfinite(meta.target_frequency_hz)
        or not (F0_FLOOR_HZ <= meta.target_frequency_hz <= F0_CEIL_HZ)
    ):
        return None
    ratio = meta.target_frequency_hz / meta.source_frequency_hz
    if not isfinite(ratio) or not (MIN_SHIFT_RATIO <= ratio <= MAX_SHIFT_RATIO):
        return None
    return ratio


class PitchShifter:
    """Shared full-buffer and live-segment pyworld pitch-shifting interface."""

    def __init__(self, world_api: Any = pyworld) -> None:
        self._core = _WorldCore(world_api)
        self._input_buffer = np.empty(0, dtype=np.float32)
        self._meta_queue: deque[_BlockMeta] = deque()
        self._previous_output_overlap: np.ndarray | None = None
        self.output_sample_clock = 0

    def shift_full_buffer(
        self,
        audio: np.ndarray,
        source_frequency_hz: float | None,
        target_frequency_hz: float | None,
        *,
        source_voiced: bool = True,
        tracking_state: TrackingState = TrackingState.TRACKING,
        target_was_clamped: bool = False,
    ) -> PitchShiftResult:
        array = np.asarray(audio)
        if array.ndim != 1:
            return self._invalid_shape(array, source_frequency_hz, target_frequency_hz, target_was_clamped)
        return self._core.shift(
            array.astype(np.float32, copy=False),
            source_frequency_hz,
            target_frequency_hz,
            source_voiced=source_voiced,
            tracking_state=tracking_state,
            target_was_clamped=target_was_clamped,
        )

    def process_block(
        self,
        block: np.ndarray,
        source_frequency_hz: float | None,
        target_frequency_hz: float | None,
        *,
        source_voiced: bool = True,
        tracking_state: TrackingState = TrackingState.TRACKING,
        target_was_clamped: bool = False,
    ) -> PitchShiftResult | None:
        array = np.asarray(block)
        if array.ndim != 1 or array.size != LIVE_BLOCK_SIZE:
            raise ValueError(f"Expected a mono block of {LIVE_BLOCK_SIZE} samples.")

        # Append audio and its metadata in lock-step.
        self._input_buffer = np.concatenate(
            (self._input_buffer, array.astype(np.float32, copy=False))
        )
        self._meta_queue.append(
            _BlockMeta(
                source_frequency_hz=source_frequency_hz,
                target_frequency_hz=target_frequency_hz,
                source_voiced=source_voiced,
                tracking_state=tracking_state,
                target_was_clamped=target_was_clamped,
            )
        )

        if self._input_buffer.size < WORLD_SEGMENT_SIZE:
            return None

        # Extract segment and pass aligned block metadata to shift_live().
        segment = self._input_buffer[:WORLD_SEGMENT_SIZE]
        block_metas = list(self._meta_queue)  # exactly _BLOCKS_PER_SEGMENT entries

        # Slide both audio buffer and metadata queue by one hop.
        self._input_buffer = self._input_buffer[HOP_SIZE:]
        for _ in range(_BLOCKS_PER_HOP):
            self._meta_queue.popleft()

        processed = self._core.shift_live(segment, block_metas)

        if not processed.valid or processed.audio is None:
            self._previous_output_overlap = None
            return processed

        if self._previous_output_overlap is None:
            emitted = processed.audio[:HOP_SIZE].copy()
        else:
            emitted = self._crossfade(self._previous_output_overlap, processed.audio[:HOP_SIZE])
        self._previous_output_overlap = processed.audio[HOP_SIZE:].copy()
        self.output_sample_clock += HOP_SIZE

        return PitchShiftResult(
            audio=self._readonly_audio(emitted.astype(np.float32, copy=False)),
            sample_rate=processed.sample_rate,
            input_sample_count=HOP_SIZE,
            output_sample_count=HOP_SIZE,
            source_frequency_hz=processed.source_frequency_hz,
            target_frequency_hz=processed.target_frequency_hz,
            shift_ratio=processed.shift_ratio,
            status=processed.status,
            valid=True,
            duration_preserved=True,
            was_clamped=processed.was_clamped,
        )

    def reset(self) -> None:
        self._input_buffer = np.empty(0, dtype=np.float32)
        self._meta_queue.clear()
        self._previous_output_overlap = None
        self.output_sample_clock = 0

    @staticmethod
    def _crossfade(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
        phase = np.linspace(0.0, np.pi / 2.0, HOP_SIZE, dtype=np.float32)
        fade_in = np.sin(phase, dtype=np.float32)
        fade_out = np.cos(phase, dtype=np.float32)
        fade_out[-1] = 0.0
        fade_in[0] = 0.0
        return previous * fade_out + current * fade_in

    @staticmethod
    def _readonly_audio(audio: np.ndarray | None) -> np.ndarray | None:
        if audio is not None:
            audio.setflags(write=False)
        return audio

    @staticmethod
    def _invalid_shape(
        audio: np.ndarray,
        source_frequency_hz: float | None,
        target_frequency_hz: float | None,
        was_clamped: bool,
    ) -> PitchShiftResult:
        return PitchShiftResult(
            audio=None,
            sample_rate=SAMPLE_RATE,
            input_sample_count=int(audio.size),
            output_sample_count=0,
            source_frequency_hz=source_frequency_hz,
            target_frequency_hz=target_frequency_hz,
            shift_ratio=None,
            status=PitchShiftStatus.NONFINITE_INPUT,
            valid=False,
            duration_preserved=False,
            was_clamped=was_clamped,
        )


__all__ = [
    "F0_CEIL_HZ",
    "F0_FLOOR_HZ",
    "FRAME_PERIOD_MS",
    "HOP_SIZE",
    "LIVE_BLOCK_SIZE",
    "MAX_SHIFT_RATIO",
    "MIN_SHIFT_RATIO",
    "OVERLAP_SIZE",
    "PitchShiftResult",
    "PitchShiftStatus",
    "PitchShifter",
    "SAMPLE_RATE",
    "WORLD_SEGMENT_SIZE",
]
