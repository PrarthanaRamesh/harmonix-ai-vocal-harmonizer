"""Synchronized, pre-rendered upload playback and sounddevice output control."""

from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, SimpleQueue
from typing import Any, Callable
from uuid import uuid4

import numpy as np
import sounddevice as sd

from .domain.errors import PlaybackOutputError, PlaybackSourceError, StateError
from .domain.models import RecordedBuffers
from .session_state.app_state import StateStore
from .session_state.commands import (
    PlaybackEngineUpdate,
    PlaybackFailed,
    PlaybackPause,
    PlaybackPlay,
    PlaybackRecovered,
    PlaybackSeek,
    PlaybackSeekApplied,
    PlaybackSetVoices,
)

SUPPORTED_VOICES = ("3rd", "5th", "octave")
VOICE_FADE_SECONDS = 0.150


@dataclass(frozen=True)
class _PreparedBlock:
    audio: np.ndarray
    sample_start: int
    sample_end: int
    playing: bool
    ended: bool
    gains_after: dict[str, float]
    fade_remaining_after: dict[str, int]


class PlaybackEngine:
    """Own one authoritative sample offset for all upload playback tracks."""

    def __init__(
        self,
        recorded_buffers: RecordedBuffers,
        rendered_voices: tuple[str, ...] | list[str] | set[str],
        *,
        initially_enabled: tuple[str, ...] | list[str] | set[str] = (),
    ) -> None:
        self.original, self.voice_buffers, self.sample_rate, self.total_samples = self._validate_source(
            recorded_buffers, rendered_voices
        )
        available = set(self.voice_buffers)
        enabled = set(initially_enabled)
        if enabled.difference(available):
            raise PlaybackSourceError("Initial playback voices include a voice that was not rendered.")
        self.available_voices = tuple(voice for voice in SUPPORTED_VOICES if voice in available)
        self.sample_offset = 0
        self.is_playing = False
        self.ended = False
        self.error: PlaybackOutputError | None = None
        self._intent_queue: SimpleQueue[object] = SimpleQueue()
        self._event_queue: SimpleQueue[object] = SimpleQueue()
        self._latest_position_update: PlaybackEngineUpdate | None = None
        self._active_voices = enabled
        self._target_gains = {voice: float(voice in enabled) for voice in self.available_voices}
        self._gains = dict(self._target_gains)
        self._fade_starts = dict(self._gains)
        self._fade_remaining = {voice: 0 for voice in self.available_voices}
        self._prepared: _PreparedBlock | None = None
        self._fade_samples = max(1, round(VOICE_FADE_SECONDS * self.sample_rate))
        self._output_failed = False
        self._resume_after_recovery = False

    @staticmethod
    def _validate_source(
        recorded_buffers: RecordedBuffers,
        rendered_voices: tuple[str, ...] | list[str] | set[str],
    ) -> tuple[np.ndarray, dict[str, np.ndarray], int, int]:
        if not isinstance(recorded_buffers, RecordedBuffers):
            raise PlaybackSourceError("Playback source must be RecordedBuffers.")
        sample_rate = recorded_buffers.sample_rate
        original = recorded_buffers.original
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            raise PlaybackSourceError("Playback source must have a positive integer sample rate.")
        if not isinstance(original, np.ndarray) or original.dtype != np.float32 or original.ndim != 1:
            raise PlaybackSourceError("Original playback audio must be a mono float32 NumPy array.")
        if original.size == 0 or not np.isfinite(original).all():
            raise PlaybackSourceError("Original playback audio must be non-empty and finite.")
        if not isinstance(recorded_buffers.a_per_voice, dict):
            raise PlaybackSourceError("Per-voice playback buffers must be a mapping.")
        unknown = set(recorded_buffers.a_per_voice).difference(SUPPORTED_VOICES)
        if unknown:
            raise PlaybackSourceError(f"Unsupported playback buffers: {sorted(unknown)}")
        rendered = set(rendered_voices)
        if rendered.difference(SUPPORTED_VOICES):
            raise PlaybackSourceError(f"Unsupported rendered voices: {sorted(rendered.difference(SUPPORTED_VOICES))}")
        if rendered.difference(recorded_buffers.a_per_voice):
            raise PlaybackSourceError("A rendered voice is missing its pre-rendered playback buffer.")
        voice_buffers: dict[str, np.ndarray] = {}
        for voice, audio in recorded_buffers.a_per_voice.items():
            if not isinstance(audio, np.ndarray) or audio.dtype != np.float32 or audio.ndim != 1:
                raise PlaybackSourceError(f"Voice {voice} must be a mono float32 NumPy array.")
            if audio.size != original.size:
                raise PlaybackSourceError(f"Voice {voice} does not match the original sample count.")
            if not np.isfinite(audio).all():
                raise PlaybackSourceError(f"Voice {voice} contains non-finite samples.")
            if voice in rendered:
                voice_buffers[voice] = audio
        return original, voice_buffers, sample_rate, int(original.size)

    @property
    def position_seconds(self) -> float:
        return self.sample_offset / self.sample_rate

    def enqueue(self, intent: object) -> None:
        if not isinstance(intent, (PlaybackPlay, PlaybackPause, PlaybackSeek, PlaybackSetVoices)):
            raise TypeError(f"Unsupported playback intent: {type(intent).__name__}")
        self._intent_queue.put(intent)

    def prepare_block(self, frame_count: int) -> np.ndarray:
        """Prepare one output block without advancing the authoritative sample clock."""
        if frame_count < 0:
            raise ValueError("frame_count must not be negative")
        if self._prepared is not None:
            raise RuntimeError("The previous playback block has not been committed or aborted.")
        if not self._output_failed:
            self._apply_pending_intents()
        if self._output_failed or not self.is_playing or frame_count == 0:
            block = np.zeros(frame_count, dtype=np.float32)
            self._prepared = _PreparedBlock(
                block,
                self.sample_offset,
                self.sample_offset,
                self.is_playing,
                self.ended,
                dict(self._gains),
                dict(self._fade_remaining),
            )
            return block

        sample_start = self.sample_offset
        sample_end = min(sample_start + frame_count, self.total_samples)
        sample_count = sample_end - sample_start
        tracks = [self.original[sample_start:sample_end]]
        gains_after = dict(self._gains)
        fade_remaining_after = dict(self._fade_remaining)
        for voice in self.available_voices:
            target = self._target_gains[voice]
            gain = self._gains[voice]
            remaining = self._fade_remaining[voice]
            if remaining <= 0:
                continue
            transition_count = min(sample_count, remaining)
            elapsed = self._fade_samples - remaining
            progress = (elapsed + np.arange(1, transition_count + 1, dtype=np.float32)) / self._fade_samples
            gains = self._fade_starts[voice] + (target - self._fade_starts[voice]) * progress
            voice_gain = np.full(sample_count, target, dtype=np.float32)
            voice_gain[:transition_count] = gains
            tracks.append(self.voice_buffers[voice][sample_start:sample_end] * voice_gain)
            gains_after[voice] = float(gains[-1]) if transition_count else gain
            fade_remaining_after[voice] = remaining - transition_count
            if fade_remaining_after[voice] == 0:
                gains_after[voice] = target
        for voice in self.available_voices:
            if self._target_gains[voice] == 1.0 and self._gains[voice] == 1.0:
                tracks.append(self.voice_buffers[voice][sample_start:sample_end])
            elif self._target_gains[voice] == 0.0 and self._gains[voice] == 0.0:
                continue
        mixed = np.mean(np.stack(tracks, axis=0), axis=0).astype(np.float32, copy=False)
        block = np.zeros(frame_count, dtype=np.float32)
        block[:sample_count] = mixed
        is_ended = sample_end == self.total_samples
        self._prepared = _PreparedBlock(
            block,
            sample_start,
            sample_end,
            not is_ended,
            is_ended,
            gains_after,
            fade_remaining_after,
        )
        return block

    def commit_block(self) -> None:
        prepared = self._prepared
        if prepared is None:
            raise RuntimeError("No playback block is prepared.")
        self._prepared = None
        self._gains = prepared.gains_after
        self._fade_remaining = prepared.fade_remaining_after
        if prepared.sample_end > prepared.sample_start:
            self.sample_offset = prepared.sample_end
            self.is_playing = prepared.playing
            self.ended = prepared.ended
        self._publish_position()

    def abort_block(self) -> None:
        self._prepared = None

    def render_block(self, frame_count: int) -> np.ndarray:
        """Render and commit a successful deterministic output block."""
        block = self.prepare_block(frame_count)
        self.commit_block()
        return block

    def fail_output(self, error: BaseException) -> None:
        if self._output_failed:
            return
        self.abort_block()
        self._resume_after_recovery = self.is_playing
        self._output_failed = True
        self.is_playing = False
        self.error = PlaybackOutputError(str(error))
        self._event_queue.put(
            PlaybackFailed(type(error).__name__, str(error), self.position_seconds)
        )

    def confirm_recovery(self) -> None:
        """Resume output after the controller has confirmed stream startup."""
        self._output_failed = False
        self.error = None
        self.is_playing = self._resume_after_recovery
        self._resume_after_recovery = False

    def reject_recovery(self, error: BaseException) -> None:
        self._output_failed = True
        self.is_playing = False
        self.error = PlaybackOutputError(str(error))
        self._event_queue.put(PlaybackFailed(type(error).__name__, str(error), self.position_seconds))

    def drain_events(self) -> tuple[object, ...]:
        events: list[object] = []
        while True:
            try:
                events.append(self._event_queue.get_nowait())
            except Empty:
                break
        latest = self._latest_position_update
        self._latest_position_update = None
        if latest is not None:
            events.append(latest)
        return tuple(events)

    def _apply_pending_intents(self) -> None:
        intents: list[object] = []
        while True:
            try:
                intents.append(self._intent_queue.get_nowait())
            except Empty:
                break
        latest_seek = next(
            (intent for intent in reversed(intents) if isinstance(intent, PlaybackSeek)),
            None,
        )
        for intent in intents:
            if isinstance(intent, PlaybackSeek) and intent is not latest_seek:
                continue
            if isinstance(intent, PlaybackPlay):
                if self.ended or self.sample_offset >= self.total_samples:
                    self.sample_offset = 0
                    self.ended = False
                self.is_playing = True
            elif isinstance(intent, PlaybackPause):
                self.is_playing = False
            elif isinstance(intent, PlaybackSeek):
                sample = round(intent.seek_seconds * self.sample_rate)
                self.sample_offset = min(max(sample, 0), self.total_samples)
                self.ended = self.sample_offset == self.total_samples
                if self.ended:
                    self.is_playing = False
                self._event_queue.put(
                    PlaybackSeekApplied(
                        intent.request_id,
                        self.position_seconds,
                        self.is_playing,
                        self.ended,
                    )
                )
            elif isinstance(intent, PlaybackSetVoices):
                enabled = set(intent.enabled_voices)
                invalid = enabled.difference(self.available_voices)
                if invalid:
                    self._event_queue.put(
                        PlaybackFailed("PlaybackVoiceError", f"Unrendered voices requested: {sorted(invalid)}", self.position_seconds)
                    )
                    continue
                for voice in self.available_voices:
                    target = float(voice in enabled)
                    if target != self._target_gains[voice]:
                        self._target_gains[voice] = target
                        self._fade_starts[voice] = self._gains[voice]
                        self._fade_remaining[voice] = self._fade_samples
                self._active_voices = enabled

    def _publish_position(self) -> None:
        self._latest_position_update = PlaybackEngineUpdate(
            self.position_seconds,
            self.is_playing,
            self.ended,
        )


class PlaybackController:
    """Route transport intents and stream events outside the audio callback."""

    def __init__(
        self,
        store: StateStore,
        recorded_buffers: RecordedBuffers,
        rendered_voices: tuple[str, ...] | list[str] | set[str],
        *,
        initially_enabled: tuple[str, ...] | list[str] | set[str] = (),
        output_stream_factory: Callable[..., Any] | None = None,
        output_device: int | str | None = None,
    ) -> None:
        self.store = store
        self.engine = PlaybackEngine(recorded_buffers, rendered_voices, initially_enabled=initially_enabled)
        self.output_stream_factory = output_stream_factory or sd.OutputStream
        self.output_device = output_device
        self._stream: Any = None
        self._recovering = False
        self._recovery_attempted = False
        self._terminal_error: PlaybackFailed | None = None

    def play(self) -> None:
        intent = PlaybackPlay()
        self.store.dispatch_playback_intent(intent)
        self.engine.enqueue(intent)
        self._ensure_stream()

    def pause(self) -> None:
        intent = PlaybackPause()
        self.store.dispatch_playback_intent(intent)
        self.engine.enqueue(intent)

    def seek(self, seek_seconds: float, request_id: str | None = None) -> str:
        identifier = request_id or uuid4().hex
        intent = PlaybackSeek(identifier, seek_seconds)
        self.store.dispatch_playback_intent(intent)
        self.engine.enqueue(intent)
        return identifier

    def set_voices(self, enabled_voices: tuple[str, ...] | list[str] | set[str]) -> None:
        intent = PlaybackSetVoices(tuple(enabled_voices))
        self.store.dispatch_playback_intent(intent)
        self.engine.enqueue(intent)

    def process_events(self) -> None:
        """Publish coalesced state outside the callback and attempt one confirmed recovery."""
        events = self.engine.drain_events()
        failure: PlaybackFailed | None = self._terminal_error
        for event in events:
            if isinstance(event, PlaybackFailed):
                failure = event
            elif isinstance(event, (PlaybackEngineUpdate, PlaybackSeekApplied)):
                self.store.publish_playback_event(event)
            elif isinstance(event, PlaybackRecovered):
                self.store.publish_playback_event(event)
        if failure is None:
            return
        self.store.publish_playback_event(failure)
        if self._recovery_attempted:
            self._terminal_error = failure
            return
        self._recovery_attempted = True
        try:
            self._close_stream()
            self._start_stream()
        except Exception as exc:
            rejected = PlaybackFailed(type(exc).__name__, str(exc), self.engine.position_seconds)
            self.engine.reject_recovery(exc)
            self._terminal_error = rejected
            self.store.publish_playback_event(rejected)
            return
        self.engine.confirm_recovery()
        recovered = PlaybackRecovered(self.engine.position_seconds, False)
        self.store.publish_playback_event(recovered)
        self._terminal_error = None

    def close(self) -> None:
        self._close_stream()

    def _audio_callback(self, outdata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        del time_info
        if self._terminal_error is not None or self.engine._output_failed:
            outdata.fill(0.0)
            return
        if status is not None and bool(getattr(status, "output_underflow", False)):
            outdata.fill(0.0)
            self.engine.fail_output(PlaybackOutputError("sounddevice reported output underflow"))
            return
        try:
            block = self.engine.prepare_block(frames)
            outdata.fill(0.0)
            outdata[: min(frames, block.size), 0] = block[:frames]
            self.engine.commit_block()
        except Exception as exc:
            outdata.fill(0.0)
            self.engine.fail_output(exc)

    def _ensure_stream(self) -> None:
        if self._stream is None:
            self._start_stream()

    def _start_stream(self) -> None:
        stream = self.output_stream_factory(
            device=self.output_device,
            samplerate=self.engine.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream = stream
        try:
            stream.start()
        except Exception:
            self._stream = None
            try:
                stream.close()
            except Exception:
                pass
            raise

    def _close_stream(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is None:
            return
        try:
            stream.stop()
        finally:
            stream.close()