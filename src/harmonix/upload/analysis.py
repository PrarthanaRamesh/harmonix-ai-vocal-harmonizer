"""Complete-track upload analysis and per-target M7 pre-render orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event, Thread
from time import monotonic
from typing import Any, Callable, Sequence
from uuid import uuid4

import numpy as np

from ..domain.models import KeyState, RecordedBuffers
from ..domain.enums import StylePreset
from ..harmony_generator import HarmonyGenerator
from ..key_scale_resolver import KeyScale, KeyScaleResolver, KeyResolutionResult
from ..input.adapter import DEFAULT_FRAME_SIZE
from ..pitch_detector import CANONICAL_SAMPLE_RATE, PitchDetectionResult, PitchDetector, TrackingState
from ..pitch_shifter import WORLD_SEGMENT_SIZE, PitchShifter
from ..session_state.app_state import StateStore
from ..session_state.commands import (
    CancelUploadAnalysis,
    CompleteUploadAnalysis,
    FailUploadAnalysis,
    RequestUploadCancellation,
    StartUploadAnalysis,
    UpdateUploadAnalysisProgress,
)
from .errors import (
    AmbiguousUploadKeyError,
    InsufficientUploadKeyEvidenceError,
    SilentOrUnusableAudioError,
    UploadAnalysisError,
    UploadCancelledError,
    UploadError,
    UploadPitchShiftError,
)
from .loader import CanonicalFrame, LoadedAudio, ProgressCallback, UploadLoader, prepare_canonical_frames

SUPPORTED_VOICES = ("3rd", "5th", "octave")
ProgressReporter = Callable[[str, int, int | None], None]
MAX_PROGRESS_DISPATCHES_PER_PHASE = 100
MAX_INTERMEDIATE_PROGRESS_DISPATCHES_PER_PHASE = MAX_PROGRESS_DISPATCHES_PER_PHASE - 1


class _ProgressCoalescer:
    """Publish bounded snapshots while retaining exact latest completed work."""

    def __init__(self, publish: ProgressReporter) -> None:
        self.publish = publish
        self.phase: str | None = None
        self.pending: tuple[str, int, int | None] | None = None
        self.last_dispatched: tuple[str, int, int | None] | None = None
        self.phase_total: int | None = None
        self.intermediate_dispatches = 0
        self.final_flushed = False
        self.next_threshold = 1
        self.stride = 1

    def report(self, phase: str, completed: int, total: int | None) -> None:
        if self.phase != phase:
            self.flush()
            self.phase = phase
            self.pending = None
            self.last_dispatched = None
            self.phase_total = total
            self.intermediate_dispatches = 0
            self.final_flushed = False
            self.stride = self._stride_for(total)
            self.next_threshold = self.stride
        elif total != self.phase_total:
            self.phase_total = total
            self._recalculate_checkpoint(completed, total)
        self.pending = (phase, completed, total)
        if (
            not self.final_flushed
            and self.intermediate_dispatches < MAX_INTERMEDIATE_PROGRESS_DISPATCHES_PER_PHASE
            and not (total is not None and completed == total)
            and self._checkpoint_reached(completed, total)
        ):
            self.publish(*self.pending)
            self.last_dispatched = self.pending
            self.intermediate_dispatches += 1
            self._advance_checkpoint(completed, total)

    def __call__(self, phase: str, completed: int, total: int | None) -> None:
        self.report(phase, completed, total)

    def flush(self) -> None:
        if self.pending is None or self.final_flushed:
            return
        if self.pending != self.last_dispatched:
            self.publish(*self.pending)
            self.last_dispatched = self.pending
        self.final_flushed = True

    def finish_phase(self, phase: str, completed: int, total: int | None) -> None:
        if self.phase != phase:
            self.flush()
            self.phase = phase
            self.pending = None
            self.last_dispatched = None
            self.phase_total = total
            self.intermediate_dispatches = 0
            self.final_flushed = False
            self.stride = self._stride_for(total)
            self.next_threshold = self.stride
        self.pending = (phase, completed, total)
        self.phase_total = total
        self.flush()

    @staticmethod
    def _stride_for(total: int | None) -> int:
        if total is None or total <= 0:
            return 1
        return max(1, (total + MAX_PROGRESS_DISPATCHES_PER_PHASE - 1) // MAX_PROGRESS_DISPATCHES_PER_PHASE)

    def _checkpoint_reached(self, completed: int, total: int | None) -> bool:
        if total is not None and completed == total:
            return False
        return completed >= self.next_threshold

    def _advance_checkpoint(self, completed: int, total: int | None) -> None:
        remaining_dispatches = MAX_INTERMEDIATE_PROGRESS_DISPATCHES_PER_PHASE - self.intermediate_dispatches
        if remaining_dispatches <= 0:
            return
        if total is None:
            self.next_threshold = max(completed * 2, completed + 1)
            return
        remaining_work = max(total - completed, 1)
        step = max(1, (remaining_work + remaining_dispatches) // (remaining_dispatches + 1))
        self.stride = max(self.stride, step)
        self.next_threshold = max(self.next_threshold, completed + self.stride)

    def _recalculate_checkpoint(self, completed: int, total: int | None) -> None:
        remaining_dispatches = MAX_INTERMEDIATE_PROGRESS_DISPATCHES_PER_PHASE - self.intermediate_dispatches
        if remaining_dispatches <= 0:
            return
        if total is None:
            self.next_threshold = max(self.next_threshold, completed + 1)
            return
        remaining_work = max(total - completed, 1)
        step = max(1, (remaining_work + remaining_dispatches) // (remaining_dispatches + 1))
        self.stride = max(self.stride, step)
        self.next_threshold = max(self.next_threshold, completed + self.stride)


@dataclass(frozen=True)
class UploadAnalysisResult:
    recorded_buffers: RecordedBuffers
    key_resolution: KeyResolutionResult
    duration_seconds: float
    frame_count: int


@dataclass(frozen=True)
class _TargetRegion:
    sample_start: int
    sample_end: int
    source_frequency_hz: float
    target_frequency_hz: float


class UploadAnalyzer:
    def __init__(
        self,
        *,
        loader: UploadLoader | None = None,
        pitch_detector: PitchDetector | None = None,
        key_resolver_factory: Callable[[], KeyScaleResolver] | None = None,
        harmony_generator: HarmonyGenerator | None = None,
        pitch_shifters: dict[str, PitchShifter] | None = None,
    ) -> None:
        self.loader = loader or UploadLoader()
        self.pitch_detector = pitch_detector or PitchDetector(frame_size=DEFAULT_FRAME_SIZE)
        if getattr(self.pitch_detector, "frame_size", DEFAULT_FRAME_SIZE) != DEFAULT_FRAME_SIZE:
            raise ValueError(f"Upload analysis requires {DEFAULT_FRAME_SIZE}-sample M4 frames.")
        self._key_resolver_factory = key_resolver_factory or KeyScaleResolver
        self.harmony_generator = harmony_generator or HarmonyGenerator()
        self.pitch_shifters = pitch_shifters or {voice: PitchShifter() for voice in SUPPORTED_VOICES}
        self.frame_size = DEFAULT_FRAME_SIZE

    def analyze(
        self,
        path: str,
        selected_voices: Sequence[str],
        *,
        manual_key: KeyScale | None = None,
        cancel_event: Any | None = None,
        progress: ProgressReporter | None = None,
    ) -> UploadAnalysisResult:
        voices = tuple(dict.fromkeys(selected_voices))
        invalid = set(voices).difference(SUPPORTED_VOICES)
        if invalid:
            raise ValueError(f"Unsupported voices: {sorted(invalid)}")
        try:
            loaded = self.loader.load(
                path,
                cancel_event=cancel_event,
                progress=self._phase_progress(progress, "decode", total=None),
            )
            self._finish_progress(progress, "decode", loaded.source_sample_count, loaded.source_sample_count)
            self._check_cancel(cancel_event)
            frames = prepare_canonical_frames(loaded.audio, self.frame_size)
            if not frames:
                raise SilentOrUnusableAudioError("Upload is shorter than one analyzable canonical frame.")

            pitches: list[PitchDetectionResult] = []
            for index, frame in enumerate(frames, start=1):
                self._check_cancel(cancel_event)
                pitches.append(self.pitch_detector.detect(frame.audio))
                self._report(progress, "pitch_analysis", index, len(frames))
            self._finish_progress(progress, "pitch_analysis", len(frames), len(frames))
            self._check_cancel(cancel_event)

            if not any(self._is_voiced(result) for result in pitches):
                raise SilentOrUnusableAudioError("Upload contains no usable voiced pitch observations.")
            resolver = self._key_resolver_factory()
            if resolver is None:
                raise UploadAnalysisError("M5 resolver factory returned no resolver.")
            if manual_key is not None:
                resolver.set_manual_key(manual_key.root, manual_key.scale_type)
                resolver.apply_processing_boundary()
            key_result = self._analyze_complete_track(resolver, pitches, cancel_event, progress)
            if key_result.ambiguous:
                raise AmbiguousUploadKeyError("M5 marked the complete-track key evidence ambiguous.")
            if key_result.insufficient_evidence or key_result.effective_key is None:
                raise InsufficientUploadKeyEvidenceError("M5 reported insufficient key evidence for the upload.")

            target_maps = self._generate_targets(frames, pitches, key_result, voices, cancel_event, progress)
            voice_audio = {
                voice: np.zeros(loaded.audio.size, dtype=np.float32)
                for voice in SUPPORTED_VOICES
            }
            regions_by_voice = {
                voice: self._target_regions(frames, pitches, target_maps[voice], loaded.audio.size)
                for voice in voices
            }
            prerender_total = sum(len(regions) for regions in regions_by_voice.values())
            self._report(progress, "prerender", 0, prerender_total)
            prerender_completed = 0
            for voice in voices:
                for region in regions_by_voice[voice]:
                    self._check_cancel(cancel_event)
                    self._render_region(loaded.audio, voice_audio[voice], region, voice)
                    prerender_completed += 1
                    self._report(progress, "prerender", prerender_completed, prerender_total)
            self._finish_progress(progress, "prerender", prerender_completed, prerender_total)

            self._check_cancel(cancel_event)
            original = np.asarray(loaded.audio, dtype=np.float32).copy()
            for audio in (original, *voice_audio.values()):
                if audio.ndim != 1 or audio.size != loaded.audio.size or not np.isfinite(audio).all():
                    raise UploadAnalysisError("Upload output violated the exact mono sample-count contract.")
            key = KeyState(
                root=key_result.effective_key.root,
                scale_type=key_result.effective_key.scale_type,
                auto_detect=manual_key is None,
                detection_confidence=key_result.confidence,
            )
            buffers = RecordedBuffers(
                original=original,
                a_per_voice=voice_audio,
                sample_rate=CANONICAL_SAMPLE_RATE,
            )
            return UploadAnalysisResult(
                recorded_buffers=buffers,
                key_resolution=key_result,
                duration_seconds=original.size / CANONICAL_SAMPLE_RATE,
                frame_count=len(frames),
            )
        except UploadError:
            raise
        except Exception as exc:
            raise UploadAnalysisError(f"Unexpected upload analysis failure: {exc}") from exc

    def _analyze_complete_track(
        self,
        resolver: KeyScaleResolver,
        pitches: list[PitchDetectionResult],
        cancel_event: Any | None,
        progress: ProgressReporter | None,
    ) -> KeyResolutionResult:
        def observations():
            for index, pitch in enumerate(pitches, start=1):
                self._check_cancel(cancel_event)
                yield pitch
                self._report(progress, "key_analysis", index, len(pitches))

        self._check_cancel(cancel_event)
        result = resolver.analyze_upload(observations())
        self._check_cancel(cancel_event)
        self._finish_progress(progress, "key_analysis", len(pitches), len(pitches))
        return result

    def _generate_targets(
        self,
        frames: Sequence[CanonicalFrame],
        pitches: Sequence[PitchDetectionResult],
        key_result: KeyResolutionResult,
        voices: tuple[str, ...],
        cancel_event: Any | None,
        progress: ProgressReporter | None,
    ) -> dict[str, list[Any | None]]:
        total = len(frames) * len(voices)
        completed = 0
        output = {voice: [None] * len(frames) for voice in voices}
        for frame_index, pitch in enumerate(pitches):
            for voice in voices:
                self._check_cancel(cancel_event)
                if self._is_voiced(pitch):
                    target = self.harmony_generator.generate(
                        pitch.frequency_hz,
                        key_result.effective_key,
                        voice,
                        voiced=pitch.voiced,
                        tracking_state=pitch.tracking_state,
                        key_ambiguous=key_result.ambiguous,
                    )
                    if target.valid and target.target_frequency_hz is not None:
                        output[voice][frame_index] = target
                completed += 1
                self._report(progress, "harmony_generation", completed, total)
        if not voices:
            self._report(progress, "harmony_generation", 0, 0)
        self._finish_progress(progress, "harmony_generation", completed, total)
        return output

    def _target_regions(
        self,
        frames: Sequence[CanonicalFrame],
        pitches: Sequence[PitchDetectionResult],
        targets: Sequence[Any | None],
        sample_count: int,
    ) -> tuple[_TargetRegion, ...]:
        regions: list[_TargetRegion] = []
        current: _TargetRegion | None = None
        for frame, pitch, target in zip(frames, pitches, targets):
            start = frame.sample_start
            end = min(sample_count, start + self.frame_size)
            valid = self._is_voiced(pitch) and target is not None and end > start
            if not valid:
                if current is not None:
                    regions.append(current)
                    current = None
                continue
            source_hz = float(pitch.frequency_hz)
            target_hz = float(target.target_frequency_hz)
            adjacent = current is not None and current.sample_end == start
            same_target = (
                current is not None
                and current.source_frequency_hz == source_hz
                and current.target_frequency_hz == target_hz
            )
            if adjacent and same_target:
                current = _TargetRegion(current.sample_start, end, source_hz, target_hz)
            else:
                if current is not None:
                    regions.append(current)
                current = _TargetRegion(start, end, source_hz, target_hz)
        if current is not None:
            regions.append(current)
        return tuple(regions)

    def _render_region(
        self,
        source: np.ndarray,
        destination: np.ndarray,
        region: _TargetRegion,
        voice: str,
    ) -> None:
        context_start, context_end = self._context_bounds(source.size, region.sample_start, region.sample_end)
        context = source[context_start:context_end]
        try:
            result = self.pitch_shifters[voice].shift_full_buffer(
                context,
                region.source_frequency_hz,
                region.target_frequency_hz,
                source_voiced=True,
                tracking_state=TrackingState.TRACKING,
            )
        except Exception as exc:
            raise UploadPitchShiftError(f"M7 raised while rendering {voice}: {exc}") from exc
        if (
            not result.valid
            or result.audio is None
            or result.sample_rate != CANONICAL_SAMPLE_RATE
            or result.output_sample_count != context.size
            or result.audio.size != context.size
            or not result.duration_preserved
        ):
            raise UploadPitchShiftError(
                f"M7 failed for {voice} target region with status {result.status.value}."
            )
        crop_start = region.sample_start - context_start
        crop_end = crop_start + region.sample_end - region.sample_start
        rendered_region = result.audio[crop_start:crop_end]
        if rendered_region.size != region.sample_end - region.sample_start:
            raise UploadPitchShiftError(f"M7 returned an incomplete {voice} target region.")
        destination[region.sample_start : region.sample_end] = np.asarray(rendered_region, dtype=np.float32)

    @staticmethod
    def _context_bounds(sample_count: int, start: int, end: int) -> tuple[int, int]:
        region_length = end - start
        desired = max(WORLD_SEGMENT_SIZE, region_length)
        missing = max(0, desired - region_length)
        left = min(start, missing // 2)
        right = min(sample_count - end, missing - left)
        left += min(start - left, missing - left - right)
        return start - left, end + right

    @staticmethod
    def _phase_progress(
        callback: ProgressReporter | None,
        phase: str,
        *,
        total: int | None,
    ) -> ProgressCallback | None:
        if callback is None:
            return None

        def report(_phase: str, completed: int, _total: int) -> None:
            UploadAnalyzer._report(callback, phase, completed, total)

        return report

    @staticmethod
    def _finish_progress(
        callback: ProgressReporter | None,
        phase: str,
        completed: int,
        total: int | None,
    ) -> None:
        if callback is None:
            return
        finish = getattr(callback, "finish_phase", None)
        if callable(finish):
            finish(phase, completed, total)
        else:
            callback(phase, completed, total)

    @staticmethod
    def _report(callback: ProgressReporter | None, phase: str, completed: int, total: int | None) -> None:
        if callback is not None:
            callback(phase, completed, total)

    @staticmethod
    def _check_cancel(cancel_event: Any | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise UploadCancelledError("Upload analysis was cancelled.")

    @staticmethod
    def _is_voiced(result: PitchDetectionResult) -> bool:
        return (
            result.voiced
            and result.tracking_state is TrackingState.TRACKING
            and result.frequency_hz is not None
            and np.isfinite(result.frequency_hz)
            and np.isfinite(result.confidence)
            and result.confidence > 0.0
        )


class _SessionCancellationProbe:
    def __init__(self, store: StateStore, request_id: str) -> None:
        self.store = store
        self.request_id = request_id

    def is_set(self) -> bool:
        session = self.store.snapshot().session
        return (
            session is None
            or session.upload_analysis.request_id != self.request_id
            or session.upload_analysis.cancellation_requested
        )


class UploadAnalysisWorker:
    """Run analysis in one worker; canonical SessionState owns all task status."""

    def __init__(self, store: StateStore, analyzer: UploadAnalyzer | None = None) -> None:
        self.store = store
        self.analyzer = analyzer or UploadAnalyzer()
        self._thread: Thread | None = None
        self._threads: list[Thread] = []
        self._request_id: str | None = None

    def start(
        self,
        path: str,
        selected_voices: Sequence[str],
        *,
        manual_key: KeyScale | None = None,
        style_preset: StylePreset = StylePreset.POP,
        keep_configure_open: bool = False,
        continuation_request_id: str | None = None,
    ) -> str:
        # Starting a newer request changes canonical request ownership. Older
        # workers observe the mismatch through their cancellation probe, and
        # reducer guards reject any late results.
        request_id = uuid4().hex
        voice_tuple = tuple(dict.fromkeys(selected_voices))
        requested_key = (
            KeyState(root=manual_key.root, scale_type=manual_key.scale_type, auto_detect=False)
            if manual_key is not None
            else None
        )
        self.store.dispatch(
            StartUploadAnalysis(
                request_id,
                path,
                voice_tuple,
                style_preset,
                requested_key,
                keep_configure_open,
                continuation_request_id,
            )
        )
        self._request_id = request_id
        self._thread = Thread(
            target=self._run,
            args=(request_id, path, voice_tuple, manual_key),
            name="harmonix-upload-analysis",
            daemon=True,
        )
        self._threads = [thread for thread in self._threads if thread.is_alive()]
        self._threads.append(self._thread)
        try:
            self._thread.start()
        except Exception as exc:
            self.store.dispatch(FailUploadAnalysis(request_id, type(exc).__name__, str(exc)))
            raise
        return request_id

    def request_cancel(self) -> None:
        if self._request_id is not None:
            self.store.dispatch(RequestUploadCancellation(self._request_id))

    def join(self, timeout: float | None = None) -> None:
        deadline = None if timeout is None else monotonic() + timeout
        for thread in tuple(self._threads):
            if thread.ident is None:
                continue
            remaining = None if deadline is None else max(0.0, deadline - monotonic())
            thread.join(remaining)

    @property
    def is_alive(self) -> bool:
        return any(thread.is_alive() for thread in self._threads)

    def _run(
        self,
        request_id: str,
        path: str,
        voices: tuple[str, ...],
        manual_key: KeyScale | None,
    ) -> None:
        cancellation = _SessionCancellationProbe(self.store, request_id)

        def publish(phase: str, completed: int, total: int | None) -> None:
            self.store.dispatch(UpdateUploadAnalysisProgress(request_id, phase, completed, total))

        coalescer = _ProgressCoalescer(publish)
        try:
            result = self.analyzer.analyze(
                path,
                voices,
                manual_key=manual_key,
                cancel_event=cancellation,
                progress=coalescer,
            )
            coalescer.flush()
            if cancellation.is_set():
                self.store.dispatch(CancelUploadAnalysis(request_id))
                return
            self.store.dispatch(
                CompleteUploadAnalysis(
                    request_id,
                    result.recorded_buffers,
                    KeyState(
                        root=result.key_resolution.effective_key.root,
                        scale_type=result.key_resolution.effective_key.scale_type,
                        auto_detect=manual_key is None,
                        detection_confidence=result.key_resolution.confidence,
                    ),
                    result.duration_seconds,
                    KeyState(
                        root=result.key_resolution.detected_key.root if result.key_resolution.detected_key else None,
                        scale_type=result.key_resolution.detected_key.scale_type if result.key_resolution.detected_key else None,
                        auto_detect=True,
                        detection_confidence=result.key_resolution.confidence,
                    ),
                )
            )
        except UploadCancelledError:
            coalescer.flush()
            self.store.dispatch(CancelUploadAnalysis(request_id))
        except Exception as exc:
            coalescer.flush()
            session = self.store.snapshot().session
            if session is not None and session.upload_analysis.request_id == request_id and session.upload_analysis.cancellation_requested:
                self.store.dispatch(CancelUploadAnalysis(request_id))
            else:
                self.store.dispatch(FailUploadAnalysis(request_id, type(exc).__name__, str(exc)))
