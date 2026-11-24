"""M8 live audio coordinator for the tested M3-M7 components."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import monotonic
from typing import Any

import numpy as np

from .harmony_generator import HarmonyGenerator
from .domain.models import RecordedBuffers
from .input.adapter import DEFAULT_FRAME_SIZE, DeviceInfo, InputAdapter
from .key_scale_resolver import KeyScaleResolver
from .latency import LatencyEvent, LatencyMeasurement, LatencyProvenance, LatencyReport, summarize_latency
from .pitch_detector import CANONICAL_SAMPLE_RATE, PitchDetector
from .pitch_shifter import HOP_SIZE, PitchShifter, WORLD_SEGMENT_SIZE
from .visualizer_bridge import LiveTelemetryFrame, VisualizerBridge

VOICE_FADE_SECONDS = 0.150
VOICE_NAMES = ("3rd", "5th", "octave")


class OutputUnderflowError(RuntimeError):
    """Raised when the output callback has no harmonized audio available."""


@dataclass(frozen=True)
class LiveAudioConfig:
    sample_rate: int = CANONICAL_SAMPLE_RATE
    block_size: int = DEFAULT_FRAME_SIZE
    input_queue_size: int = 16
    output_queue_size: int = 16
    voice_fade_seconds: float = VOICE_FADE_SECONDS


@dataclass(frozen=True)
class _OutputChunk:
    audio: np.ndarray
    capture_timestamp: float | None
    pipeline_timestamp: float | None
    sample_start: int


@dataclass(frozen=True)
class _InputChunk:
    audio: np.ndarray
    capture_timestamp: float | None
    pipeline_timestamp: float | None


class LiveAudioCoordinator:
    """Own M8 queues, worker processing, full-duplex stream, and recording."""

    def __init__(
        self,
        input_adapter: InputAdapter,
        *,
        pitch_detector: PitchDetector | None = None,
        key_resolver: KeyScaleResolver | None = None,
        harmony_generator: HarmonyGenerator | None = None,
        pitch_shifters: dict[str, PitchShifter] | None = None,
        config: LiveAudioConfig | None = None,
        visualizer_bridge: VisualizerBridge | None = None,
    ) -> None:
        self.input_adapter = input_adapter
        self.config = config or LiveAudioConfig()
        self.pitch_detector = pitch_detector or PitchDetector()
        self.key_resolver = key_resolver or KeyScaleResolver()
        self.harmony_generator = harmony_generator or HarmonyGenerator()
        self.pitch_shifters = pitch_shifters or {voice: PitchShifter() for voice in VOICE_NAMES}
        self.visualizer_bridge = visualizer_bridge
        self.request_id = ""
        self._input_queue: Queue[_InputChunk] = Queue(maxsize=self.config.input_queue_size)
        self._output_queue: Queue[_OutputChunk] = Queue(maxsize=self.config.output_queue_size)
        self._output_pending: deque[_OutputChunk] = deque()
        self._queue_lock = Lock()
        self._measurements: list[LatencyMeasurement] = []
        self._latency_events: list[LatencyEvent] = []
        self._trial = 0
        self._stream: Any = None
        self._worker: Thread | None = None
        self._stop_event = Event()
        # Starts set so that coordinators used in unit tests (without calling start()) behave
        # normally. Cleared by start() before the stream opens; re-set by the worker thread
        # after its first output chunk is enqueued. Until re-set, _callback writes silence
        # without marking the coordinator as underflowed (startup grace period).
        self._output_ready: Event = Event()
        self._output_ready.set()
        self._active_voices: set[str] = set()
        self._processing_voices: set[str] = set()
        self._pending_voices: set[str] | None = None
        self._voice_targets: set[str] = set()
        self._voice_gains = {voice: 0.0 for voice in VOICE_NAMES}
        self._input_sample_clock = 0
        self._input_history: deque[tuple[float | None, float | None, int]] = deque(maxlen=4)
        self._canonical_pending = np.empty(0, dtype=np.float32)
        self.recorded = RecordedBuffers(
            original=np.empty(0, dtype=np.float32),
            a_per_voice={voice: np.empty(0, dtype=np.float32) for voice in VOICE_NAMES},
            sample_rate=self.config.sample_rate,
        )
        self._original_recording_chunks: list[np.ndarray] = []
        self._voice_recording_chunks: dict[str, list[np.ndarray]] = {voice: [] for voice in VOICE_NAMES}
        self._recording_dirty = False
        self.last_error: Exception | None = None
        self.underflowed = False
        self.input_queue_overflowed = False
        self.underflow_error: OutputUnderflowError | None = None
        self.active_input_device: DeviceInfo | None = None
        # Fix 2: dedicated detector thread state.
        # PitchDetector.detect() (librosa.pyin) takes up to 58 ms — exceeding the 46 ms
        # callback interval. Running it in a separate thread decouples detection latency
        # from the worker's output-enqueue cadence. The worker reads the latest cached
        # result; at most one frame of pitch lag (~46 ms) is introduced.
        self._detect_queue: Queue[np.ndarray] = Queue(maxsize=1)
        self._last_pitch: Any = None
        self._pitch_lock = Lock()
        self._detector_thread: Thread | None = None

    @property
    def active_voices(self) -> frozenset[str]:
        return frozenset(self._active_voices)

    def set_active_voices(self, voices: set[str] | list[str] | tuple[str, ...]) -> None:
        requested = set(voices)
        invalid = requested.difference(VOICE_NAMES)
        if invalid:
            raise ValueError(f"Unsupported voices: {sorted(invalid)}")
        self._pending_voices = requested

    def process_block(
        self,
        block: np.ndarray,
        capture_timestamp: float | None = None,
        pipeline_timestamp: float | None = None,
    ) -> np.ndarray | None:
        """Process one canonical block on the worker side; return one output hop if ready."""
        array = np.asarray(block, dtype=np.float32)
        if array.ndim != 1 or array.size != self.config.block_size:
            raise ValueError(f"Expected a mono block of {self.config.block_size} samples.")
        if capture_timestamp is None and pipeline_timestamp is None:
            pipeline_timestamp = monotonic()
        capture_time = float(capture_timestamp) if capture_timestamp is not None else None
        pipeline_time = float(pipeline_timestamp) if pipeline_timestamp is not None else None
        self._record_input(array)
        if self._pending_voices is not None:
            self._voice_targets = self._pending_voices
            self._active_voices = set(self._voice_targets)
            self._processing_voices.update(self._voice_targets)
            self._pending_voices = None
        self._input_history.append((capture_time, pipeline_time, self._input_sample_clock - array.size))
        self._enqueue_for_detect(array)
        pitch = self._read_latest_pitch(array)
        key_result = self.key_resolver.observe(pitch)
        self._publish_telemetry(array, pitch, key_result)
        if not self._processing_voices:
            output = array.copy()
            voice_outputs: dict[str, np.ndarray] = {}
        else:
            voice_outputs, hop_ready = self._process_harmony(pitch, key_result, array)
            if hop_ready:
                if len(self._original_recording_chunks) >= 4:
                    dry_segment = np.concatenate(self._original_recording_chunks[-4:], dtype=np.float32)
                    dry_hop = dry_segment[: HOP_SIZE]
                else:
                    dry_hop = np.zeros(HOP_SIZE, dtype=np.float32)
                if voice_outputs:
                    harmony_mean = np.mean(np.stack(list(voice_outputs.values())), axis=0).astype(np.float32)
                    output = (0.5 * dry_hop + 0.5 * harmony_mean).astype(np.float32)
                else:
                    output = dry_hop
            else:
                output = None
        if output is not None and self._processing_voices:
            origin = self._input_history[0] if len(self._input_history) == 4 else self._input_history[-1]
            self._record_output(voice_outputs, origin[2])
            self._advance_voice_gains(output.size)
            return output
        if output is not None:
            return output
        return None

    def _publish_telemetry(self, array: np.ndarray, pitch: Any, key_result: Any) -> None:
        if self.visualizer_bridge is None:
            return
        input_level = float(np.sqrt(np.mean(array**2))) if array.size else 0.0
        in_tune = False
        freq = getattr(pitch, "frequency_hz", None)
        voiced = getattr(pitch, "voiced", False)
        confidence = float(getattr(pitch, "confidence", 0.0))
        if voiced and freq is not None and freq > 0.0:
            from math import log2

            midi = 69.0 + 12.0 * log2(freq / 440.0)
            in_tune = abs(midi - round(midi)) <= 0.25 and confidence >= 0.6
        tracking_state = getattr(pitch, "tracking_state", "idle")
        tracking_state_str = getattr(tracking_state, "value", str(tracking_state))

        physical_ms = None
        pipeline_ms = None
        provenance = "unavailable"
        if self._measurements:
            latest_m = self._measurements[-1]
            physical_ms = latest_m.latency_ms
            pipeline_ms = latest_m.pipeline_latency_ms
            provenance = getattr(latest_m.provenance, "value", str(latest_m.provenance))

        frame = LiveTelemetryFrame(
            request_id=self.request_id,
            input_level=input_level,
            current_f0=float(freq) if voiced and freq is not None else None,
            confidence=confidence,
            tracking_state=tracking_state_str,
            in_tune=in_tune,
            detected_key=getattr(key_result, "detected_key", None),
            effective_key=getattr(key_result, "effective_key", None),
            physical_latency_ms=physical_ms,
            pipeline_latency_ms=pipeline_ms,
            latency_provenance=provenance,
            mic_connected=self._stream is not None,
            error_message=str(self.last_error) if self.last_error else None,
        )
        self.visualizer_bridge.publish(frame)

    def start(self, device_index: int | None = None, request_id: str | None = None) -> DeviceInfo | None:
        if request_id is not None:
            self.request_id = request_id
        if self._stream is not None:
            return self.active_input_device
        self._stop_event.clear()
        self._output_ready.clear()  # Enter startup grace period; re-set by worker after first output chunk
        with self._pitch_lock:
            self._last_pitch = None
        device = self.input_adapter.start_capture(device_index)
        stream_device_index = device.index if device is not None else device_index
        self._stream = self.input_adapter.create_duplex_stream(self._callback, stream_device_index)
        try:
            self._stream.start()
            # Fix 3: Pre-fill output queue with silence when the stream runs at a non-canonical
            # rate. At 48 kHz, each input callback yields only 1881 canonical samples, so the
            # accumulator takes more than one callback to produce a 2048-sample output block.
            # Pre-seeding N0 silence chunks ensures the first N0 output callbacks find audio
            # in the queue instead of underflowing. N0 is derived from the rate arithmetic so
            # it scales correctly with any non-canonical native rate.
            import math
            native_rate = self.input_adapter.native_sample_rate or self.config.sample_rate
            if native_rate != self.config.sample_rate:
                canonical_per_callback = self.config.block_size * (self.config.sample_rate / native_rate)
                n0 = math.ceil(self.config.block_size / canonical_per_callback)
                silence = np.zeros(self.config.block_size, dtype=np.float32)
                for _ in range(n0):
                    try:
                        self._output_queue.put_nowait(
                            _OutputChunk(silence.copy(), None, None, 0)
                        )
                    except Full:
                        break  # Queue unexpectedly full; proceed without additional pre-fill
            self._worker = Thread(target=self._worker_loop, name="harmonix-live-worker", daemon=True)
            self._worker.start()
            # Fix 2: start dedicated detector thread after the worker so the output pipeline
            # is already live when the first pitch result is cached.
            self._detector_thread = Thread(
                target=self._detector_loop, name="harmonix-pitch-detector", daemon=True
            )
            self._detector_thread.start()
            self.active_input_device = device
            return device
        except Exception:
            self.active_input_device = None
            stream = self._stream
            self._stream = None
            try:
                stream.stop()
            except Exception as exc:
                self.last_error = exc
            finally:
                try:
                    stream.close()
                except Exception as exc:
                    self.last_error = exc
                self.input_adapter.stop_capture()
            raise

    @property
    def is_alive(self) -> bool:
        return self._stream is not None or (self._worker is not None and self._worker.is_alive())

    def stop(self, timeout: float | None = 1.0) -> bool:
        self._stop_event.set()
        stream = self._stream
        self._stream = None
        try:
            if stream is not None:
                try:
                    stream.stop()
                except Exception as exc:
                    self.last_error = exc
                finally:
                    try:
                        stream.close()
                    except Exception as exc:
                        self.last_error = exc
            if self._detector_thread is not None:
                self._detector_thread.join(timeout=timeout)
                if not self._detector_thread.is_alive():
                    self._detector_thread = None
            if self._worker is not None:
                self._worker.join(timeout=timeout)
                if not self._worker.is_alive():
                    self._worker = None
                    return True
                return False
            return True
        finally:
            self.active_input_device = None
            self.input_adapter.stop_capture()

    def shutdown(self, timeout: float | None = None) -> bool:
        """Shut down the stream and worker, ensuring the worker reference is retained if alive."""
        return self.stop(timeout=timeout)

    def recorded_buffers(self) -> RecordedBuffers:
        self._materialize_recorded_buffers()
        return self.recorded

    def latency_report(self) -> LatencyReport:
        return summarize_latency(self._measurements, self._latency_events)

    def _callback(self, indata: np.ndarray, outdata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        adc_time = self._time_value(time_info, "inputBufferAdcTime")
        capture_time = adc_time if self._usable_input_adc_time(adc_time) else None
        callback_time = self._time_value(time_info, "currentTime")
        output_time = self._time_value(time_info, "outputBufferDacTime")
        input_queue_overflowed = False
        try:
            self._input_queue.put_nowait(
                _InputChunk(np.asarray(indata[:, 0], dtype=np.float32).copy(), capture_time, callback_time)
            )
        except Full:
            self.input_queue_overflowed = True
            input_queue_overflowed = True
        outdata.fill(0.0)
        remaining = frames
        offset = 0
        emitted: list[tuple[_OutputChunk, int, int]] = []
        if not self._output_ready.is_set():
            # Startup grace period: the worker has not yet produced any output. Silence (already
            # written above) is the correct output here, not an underflow. Clear remaining so
            # downstream latency-event checks do not flag this callback as a queue starvation.
            remaining = 0
        else:
            with self._queue_lock:
                while True:
                    try:
                        self._output_pending.append(self._output_queue.get_nowait())
                    except Empty:
                        break
                while remaining and self._output_pending:
                    chunk = self._output_pending[0]
                    take = min(remaining, chunk.audio.size)
                    outdata[offset : offset + take, 0] = chunk.audio[:take]
                    emitted.append((chunk, take, offset))
                    if take == chunk.audio.size:
                        self._output_pending.popleft()
                    else:
                        self._output_pending[0] = _OutputChunk(
                            chunk.audio[take:], chunk.capture_timestamp, chunk.pipeline_timestamp, chunk.sample_start + take
                        )
                    offset += take
                    remaining -= take
                if remaining:
                    self.underflowed = True
                    self.underflow_error = OutputUnderflowError("Harmonized output queue underflowed.")
        callback_input_overflow = self._status_flag(status, "input_overflow")
        callback_output_underflow = self._status_flag(status, "output_underflow")
        if input_queue_overflowed or remaining or callback_input_overflow or callback_output_underflow:
            self._latency_events.append(
                LatencyEvent(
                    callback_timestamp=callback_time,
                    output_timestamp=output_time,
                    input_queue_overflowed=input_queue_overflowed,
                    output_queue_underflowed=remaining > 0,
                    callback_input_overflow=callback_input_overflow,
                    callback_output_underflow=callback_output_underflow,
                )
            )
        for origin, sample_count, sample_offset in emitted:
            if output_time is None:
                continue
            adjusted_output_time = output_time + (sample_offset / self.config.sample_rate)
            provenance = (
                LatencyProvenance.VERIFIED_CAPTURE_TO_DAC
                if origin.capture_timestamp is not None
                else (
                    LatencyProvenance.PIPELINE_CALLBACK_TO_DAC
                    if origin.pipeline_timestamp is not None
                    else LatencyProvenance.PHYSICAL_CAPTURE_UNAVAILABLE
                )
            )
            self._measurements.append(
                LatencyMeasurement(
                    trial=self._trial,
                    capture_timestamp=origin.capture_timestamp,
                    output_timestamp=adjusted_output_time,
                    pipeline_timestamp=origin.pipeline_timestamp,
                    provenance=provenance,
                    sample_start=origin.sample_start,
                    sample_count=sample_count,
                    active_voice_count=len(self._active_voices),
                    mode="live",
                    input_device=str(self.active_input_device.index if self.active_input_device is not None else self.input_adapter.device_index),
                    output_device=None,
                    sample_rate=self.config.sample_rate,
                    block_size=self.config.block_size,
                    processing_segment_size=WORLD_SEGMENT_SIZE,
                    queue_overflowed=False,
                    queue_underflowed=False,
                )
            )
            self._trial += 1

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            if not self.process_pending_once():
                self._stop_event.wait(0.001)

    def _detector_loop(self) -> None:
        """Dedicated thread: dequeues frames, calls PitchDetector.detect(), caches the result.
        Runs until _stop_event is set. Blocks on _detect_queue.get(timeout=...) so it does not
        spin when no frames are available."""
        from .pitch_detector import PitchDetectionResult, TrackingState
        while not self._stop_event.is_set():
            try:
                frame = self._detect_queue.get(timeout=0.05)
            except Exception:
                continue
            try:
                result = self.pitch_detector.detect(frame)
            except Exception as exc:
                self.last_error = exc
                result = PitchDetectionResult(
                    frequency_hz=None,
                    confidence=0.0,
                    voiced=False,
                    tracking_state=TrackingState.TRACKING_LOST,
                )
            with self._pitch_lock:
                self._last_pitch = result

    def _enqueue_for_detect(self, frame: np.ndarray) -> None:
        """Offer the current frame to the detector thread. If the queue is full (detector
        is still busy with the previous frame), the frame is silently dropped — the worker
        will use the most recently cached result instead."""
        try:
            self._detect_queue.put_nowait(frame.copy())
        except Full:
            pass  # Detector is behind; use cached result on the next _read_latest_pitch call

    def _read_latest_pitch(self, frame: np.ndarray) -> Any:
        """Return the most recently completed pitch result, or detect synchronously when no
        cached result exists yet (only on the very first block before the detector thread
        has completed its first call)."""
        with self._pitch_lock:
            cached = self._last_pitch
        if cached is not None:
            return cached
        # Detector thread has not yet produced a result (early startup). Run synchronously
        # exactly once so the rest of the pipeline has a valid pitch object immediately.
        from .pitch_detector import PitchDetectionResult, TrackingState
        try:
            result = self.pitch_detector.detect(frame)
        except Exception as exc:
            self.last_error = exc
            result = PitchDetectionResult(
                frequency_hz=None,
                confidence=0.0,
                voiced=False,
                tracking_state=TrackingState.TRACKING_LOST,
            )
        with self._pitch_lock:
            if self._last_pitch is None:
                self._last_pitch = result
        return result

    def process_pending_once(self) -> bool:
        try:
            input_chunk = self._input_queue.get_nowait()
        except Empty:
            return False
        native_rate = self.input_adapter.native_sample_rate or self.config.sample_rate
        canonical = self.input_adapter.process_native_chunk(input_chunk.audio, native_rate)
        self._canonical_pending = np.concatenate((self._canonical_pending, canonical))
        emitted: list[_OutputChunk] = []
        while self._canonical_pending.size >= self.config.block_size:
            frame = self._canonical_pending[: self.config.block_size]
            self._canonical_pending = self._canonical_pending[self.config.block_size :]
            candidate = self.process_block(frame, input_chunk.capture_timestamp, input_chunk.pipeline_timestamp)
            if candidate is not None:
                origin = self._input_history[0] if len(self._input_history) == 4 else self._input_history[-1]
                if native_rate != self.config.sample_rate and hasattr(self.input_adapter, "process_canonical_output"):
                    output_audio = self.input_adapter.process_canonical_output(candidate, native_rate)
                else:
                    output_audio = candidate
                emitted.append(_OutputChunk(output_audio, origin[0], origin[1], origin[2]))
        for chunk in emitted:
            try:
                self._output_queue.put_nowait(chunk)
                self._output_ready.set()  # Idempotent; signals _callback that output is available
            except Full:
                self._latency_events.append(
                    LatencyEvent(callback_timestamp=None, output_timestamp=None, output_queue_overflowed=True)
                )
        return True

    def _process_harmony(self, pitch: Any, key_result: Any, block: np.ndarray) -> tuple[dict[str, np.ndarray], bool]:
        has_key = key_result.effective_key is not None and not key_result.ambiguous
        outputs: dict[str, np.ndarray] = {}
        hop_ready = False
        for voice in sorted(self._processing_voices):
            target = None
            if has_key:
                target = self.harmony_generator.generate(
                    pitch.frequency_hz,
                    key_result.effective_key,
                    voice,
                    voiced=pitch.voiced,
                    tracking_state=pitch.tracking_state,
                    key_ambiguous=key_result.ambiguous,
                )
            target_freq = target.target_frequency_hz if target is not None and target.valid else None
            is_voiced = bool(pitch.voiced and target_freq is not None)
            shifted = self.pitch_shifters[voice].process_block(
                block,
                pitch.frequency_hz if is_voiced else None,
                target_freq,
                source_voiced=is_voiced,
                tracking_state=pitch.tracking_state,
                target_was_clamped=target.was_clamped if target is not None else False,
            )
            if shifted is not None:
                hop_ready = True
                if shifted.valid and shifted.audio is not None:
                    outputs[voice] = self._apply_voice_fade(voice, shifted.audio)
        return outputs, hop_ready

    def _apply_voice_fade(self, voice: str, audio: np.ndarray) -> np.ndarray:
        target = 1.0 if voice in self._voice_targets else 0.0
        current = self._voice_gains[voice]
        step = audio.size / (self.config.sample_rate * self.config.voice_fade_seconds)
        next_gain = min(1.0, current + step) if target else max(0.0, current - step)
        gain = np.linspace(current, next_gain, audio.size, dtype=np.float32)
        return audio * gain

    def _advance_voice_gains(self, sample_count: int) -> None:
        for voice in tuple(self._processing_voices):
            target = 1.0 if voice in self._voice_targets else 0.0
            step = sample_count / (self.config.sample_rate * self.config.voice_fade_seconds)
            if target:
                self._voice_gains[voice] = min(1.0, self._voice_gains[voice] + step)
            else:
                self._voice_gains[voice] = max(0.0, self._voice_gains[voice] - step)
            if target == 0.0 and self._voice_gains[voice] <= 0.0:
                self._voice_gains[voice] = 0.0
                self._processing_voices.discard(voice)

    def _record_input(self, block: np.ndarray) -> None:
        self._original_recording_chunks.append(block.copy())
        for voice in VOICE_NAMES:
            self._voice_recording_chunks[voice].append(np.zeros(block.size, dtype=np.float32))
        self._recording_dirty = True
        self._input_sample_clock += block.size

    def _record_output(self, outputs: dict[str, np.ndarray], sample_start: int) -> None:
        # M7 output is delayed by its initial fill; recorded arrays remain input-clock aligned.
        for voice, output in outputs.items():
            chunks = self._voice_recording_chunks[voice]
            chunk_index, chunk_offset = divmod(sample_start, self.config.block_size)
            output_offset = 0
            while output_offset < output.size and chunk_index < len(chunks):
                chunk = chunks[chunk_index]
                count = min(output.size - output_offset, chunk.size - chunk_offset)
                if count <= 0:
                    break
                chunk[chunk_offset : chunk_offset + count] = output[output_offset : output_offset + count]
                output_offset += count
                chunk_index += 1
                chunk_offset = 0
            if output_offset:
                self._recording_dirty = True

    def _materialize_recorded_buffers(self) -> None:
        if not self._recording_dirty:
            return
        self.recorded.original = (
            np.concatenate(self._original_recording_chunks).astype(np.float32, copy=False)
            if self._original_recording_chunks
            else np.empty(0, dtype=np.float32)
        )
        for voice in VOICE_NAMES:
            chunks = self._voice_recording_chunks[voice]
            self.recorded.a_per_voice[voice] = (
                np.concatenate(chunks).astype(np.float32, copy=False) if chunks else np.empty(0, dtype=np.float32)
            )
        self._recording_dirty = False

    @staticmethod
    def _time_value(time_info: Any, name: str) -> float | None:
        value = getattr(time_info, name, None)
        if value is None and isinstance(time_info, dict):
            value = time_info.get(name)
        return float(value) if value is not None else None

    @staticmethod
    def _usable_input_adc_time(value: float | None) -> bool:
        """PortAudio's zero ADC time is unusable on the documented Windows device."""
        return value is not None and np.isfinite(value) and value > 0.0

    @staticmethod
    def _status_flag(status: Any, name: str) -> bool:
        if status is None:
            return False
        if isinstance(status, dict):
            return bool(status.get(name, False))
        return bool(getattr(status, name, False))
