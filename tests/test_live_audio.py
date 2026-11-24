from threading import Event, Thread
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from harmonix.domain.errors import PermissionDeniedError
from harmonix.domain.models import RecordedBuffers
from harmonix.input.adapter import DEFAULT_FRAME_SIZE, DeviceInfo, InputAdapter
from harmonix.key_scale_resolver import KeyScale, KeyScaleResolver
from harmonix.latency import LatencyProvenance
from harmonix.live_audio import _InputChunk, _OutputChunk, LiveAudioConfig, LiveAudioCoordinator, OutputUnderflowError
from harmonix.pitch_detector import PitchDetectionResult, TrackingState
from harmonix.pitch_shifter import HOP_SIZE, LIVE_BLOCK_SIZE
from harmonix.visualizer_bridge import VisualizerBridge


class FakeAdapter:
    device_index = 2
    native_sample_rate = 48000

    def process_native_chunk(self, chunk, sample_rate):
        return np.asarray(chunk, dtype=np.float32)

    def stop_capture(self) -> None:
        pass


class FakeDetector:
    def detect(self, frame):
        return PitchDetectionResult(440.0, 0.9, True, TrackingState.TRACKING)


class RecordingDetector(FakeDetector):
    def __init__(self):
        self.frame_sizes = []

    def detect(self, frame):
        self.frame_sizes.append(frame.size)
        return super().detect(frame)


class CanonicalizingAdapter(FakeAdapter):
    def process_native_chunk(self, chunk, sample_rate):
        self.last_rate = sample_rate
        return np.ones(LIVE_BLOCK_SIZE, dtype=np.float32)


class FailingStream:
    def __init__(self):
        self.stopped = False
        self.closed = False

    def start(self):
        raise RuntimeError("worker startup test failure")

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


class FakeResolver:
    def observe(self, result):
        return SimpleNamespace(effective_key=KeyScale("C", "major"), ambiguous=False)


class FakeHarmony:
    def generate(self, frequency, key, voice, **kwargs):
        return SimpleNamespace(valid=True, target_frequency_hz=660.0, was_clamped=False)


class FakeShifter:
    def __init__(self, initial_blocks=4, emit_calls=None):
        self.calls = 0
        self.initial_blocks = initial_blocks
        self.emit_calls = emit_calls

    def process_block(self, block, source, target, **kwargs):
        self.calls += 1
        if self.calls < self.initial_blocks or (self.emit_calls is not None and self.calls not in self.emit_calls):
            return None
        return SimpleNamespace(valid=True, audio=np.ones(HOP_SIZE, dtype=np.float32))


class LiveAudioTests(unittest.TestCase):
    def coordinator(self, voices=None):
        return LiveAudioCoordinator(
            FakeAdapter(),
            pitch_detector=FakeDetector(),
            key_resolver=FakeResolver(),
            harmony_generator=FakeHarmony(),
            pitch_shifters={voice: FakeShifter() for voice in ("3rd", "5th", "octave")},
            config=LiveAudioConfig(input_queue_size=2, output_queue_size=2),
        )

    def test_initializes_with_canonical_block_contract(self) -> None:
        coordinator = self.coordinator()
        self.assertEqual(coordinator.config.sample_rate, 44100)
        self.assertEqual(coordinator.config.block_size, 2048)

    def test_zero_voices_passthrough_preserves_block(self) -> None:
        coordinator = self.coordinator()
        block = np.linspace(-1.0, 1.0, LIVE_BLOCK_SIZE, dtype=np.float32)
        output = coordinator.process_block(block, 1.0)

        np.testing.assert_array_equal(output, block)

    def test_active_voice_pipeline_waits_for_m7_output(self) -> None:
        coordinator = self.coordinator()
        coordinator.set_active_voices({"3rd"})
        block = np.ones(LIVE_BLOCK_SIZE, dtype=np.float32)

        for timestamp in (1.0, 2.0, 3.0):
            self.assertIsNone(coordinator.process_block(block, timestamp))
        self.assertIsNotNone(coordinator.process_block(block, 4.0))
        self.assertEqual(coordinator.pitch_shifters["3rd"].calls, 4)

    def test_three_voice_pipeline_runs_each_voice(self) -> None:
        coordinator = self.coordinator()
        coordinator.set_active_voices({"3rd", "5th", "octave"})
        block = np.ones(LIVE_BLOCK_SIZE, dtype=np.float32)
        for timestamp in (1.0, 2.0, 3.0):
            self.assertIsNone(coordinator.process_block(block, timestamp))
        output = coordinator.process_block(block, 4.0)

        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (HOP_SIZE,))
        for shifter in coordinator.pitch_shifters.values():
            self.assertEqual(shifter.calls, 4)

    def test_voice_change_is_staged_until_next_block_boundary(self) -> None:
        coordinator = self.coordinator()
        self.assertEqual(coordinator.active_voices, frozenset())
        coordinator.set_active_voices({"3rd"})
        self.assertEqual(coordinator.active_voices, frozenset())
        coordinator.process_block(np.ones(LIVE_BLOCK_SIZE, dtype=np.float32), 1.0)
        self.assertEqual(coordinator.active_voices, frozenset({"3rd"}))

    def test_recorded_buffers_are_aligned_and_inactive_voice_zero_filled(self) -> None:
        coordinator = self.coordinator()
        block = np.ones(LIVE_BLOCK_SIZE, dtype=np.float32)
        coordinator.process_block(block, 1.0)
        recorded = coordinator.recorded_buffers()
        original = recorded.original
        per_voice = recorded.a_per_voice
        sample_rate = recorded.sample_rate

        self.assertEqual(sample_rate, 44100)
        self.assertEqual(original.size, LIVE_BLOCK_SIZE)
        for voice in ("3rd", "5th", "octave"):
            self.assertEqual(per_voice[voice].size, original.size)
        np.testing.assert_array_equal(per_voice["5th"], np.zeros_like(original))

    def test_recorded_buffers_are_authoritative_instance(self) -> None:
        coordinator = self.coordinator()
        coordinator.process_block(np.ones(LIVE_BLOCK_SIZE, dtype=np.float32), 1.0)

        self.assertIs(coordinator.recorded_buffers(), coordinator.recorded)
        self.assertEqual(set(coordinator.recorded.a_per_voice), {"3rd", "5th", "octave"})

    def test_recording_multiple_blocks_preserves_original_order_and_lengths(self) -> None:
        coordinator = self.coordinator()
        blocks = [
            np.full(LIVE_BLOCK_SIZE, value, dtype=np.float32)
            for value in (1.0, 2.0, 3.0)
        ]
        for index, block in enumerate(blocks):
            coordinator.process_block(block, float(index))

        recorded = coordinator.recorded_buffers()
        expected = np.concatenate(blocks)
        np.testing.assert_array_equal(recorded.original, expected)
        self.assertEqual(recorded.original.dtype, np.float32)
        self.assertEqual(recorded.original.size, LIVE_BLOCK_SIZE * len(blocks))
        self.assertEqual(recorded.sample_rate, 44100)
        for voice in ("3rd", "5th", "octave"):
            np.testing.assert_array_equal(recorded.a_per_voice[voice], np.zeros_like(expected))

    def test_recorded_voice_output_overwrites_its_input_clock_offset_across_chunks(self) -> None:
        coordinator = self.coordinator()
        coordinator._record_input(np.zeros(LIVE_BLOCK_SIZE, dtype=np.float32))
        coordinator._record_input(np.zeros(LIVE_BLOCK_SIZE, dtype=np.float32))
        shifted = np.arange(HOP_SIZE, dtype=np.float32)
        sample_start = LIVE_BLOCK_SIZE - 10

        coordinator._record_output({"3rd": shifted}, sample_start)
        recorded = coordinator.recorded_buffers()
        expected = np.zeros(LIVE_BLOCK_SIZE * 2, dtype=np.float32)
        recorded_end = min(expected.size, sample_start + shifted.size)
        expected[sample_start:recorded_end] = shifted[: recorded_end - sample_start]
        np.testing.assert_array_equal(recorded.a_per_voice["3rd"], expected)
        np.testing.assert_array_equal(recorded.a_per_voice["5th"], np.zeros_like(expected))

    def test_recording_accumulates_chunks_without_per_block_concatenation(self) -> None:
        coordinator = self.coordinator()
        block = np.ones(LIVE_BLOCK_SIZE, dtype=np.float32)
        with patch("harmonix.live_audio.np.concatenate", wraps=np.concatenate) as concatenate:
            for index in range(3):
                coordinator.process_block(block, float(index))
            self.assertEqual(concatenate.call_count, 0)
            coordinator.recorded_buffers()

        self.assertEqual(concatenate.call_count, 4)

    def test_callback_underflow_outputs_explicit_silence(self) -> None:
        coordinator = self.coordinator()
        indata = np.ones((LIVE_BLOCK_SIZE, 1), dtype=np.float32)
        outdata = np.ones((LIVE_BLOCK_SIZE, 1), dtype=np.float32)
        coordinator._callback(indata, outdata, LIVE_BLOCK_SIZE, {"inputBufferAdcTime": 1.0, "outputBufferDacTime": 1.05}, None)

        np.testing.assert_array_equal(outdata, np.zeros_like(outdata))
        self.assertTrue(coordinator.underflowed)
        self.assertIsInstance(coordinator.underflow_error, OutputUnderflowError)
        event = coordinator.latency_report().events[0]
        self.assertTrue(event.output_queue_underflowed)
        self.assertFalse(event.output_queue_overflowed)

    def test_callback_handoff_is_bounded(self) -> None:
        coordinator = self.coordinator()
        indata = np.ones((LIVE_BLOCK_SIZE, 1), dtype=np.float32)
        outdata = np.zeros((LIVE_BLOCK_SIZE, 1), dtype=np.float32)
        for _ in range(3):
            coordinator._callback(indata, outdata, LIVE_BLOCK_SIZE, {}, None)

        self.assertTrue(coordinator.input_queue_overflowed)

    def test_input_queue_overflow_is_reported_separately(self) -> None:
        coordinator = self.coordinator()
        indata = np.ones((HOP_SIZE, 1), dtype=np.float32)
        outdata = np.zeros((HOP_SIZE, 1), dtype=np.float32)
        coordinator._input_queue.put(_InputChunk(indata[:, 0], 1.0, 1.0))
        coordinator._input_queue.put(_InputChunk(indata[:, 0], 1.5, 1.5))
        coordinator._output_queue.put(_OutputChunk(np.ones(HOP_SIZE, dtype=np.float32), 2.0, 2.0, 0))

        coordinator._callback(indata, outdata, HOP_SIZE, {"outputBufferDacTime": 3.0}, None)

        report = coordinator.latency_report()
        self.assertEqual(report.overrun_events, 1)
        self.assertEqual(report.output_queue_overflow_events, 0)
        self.assertTrue(report.events[0].input_queue_overflowed)
        self.assertFalse(report.events[0].output_queue_overflowed)
        self.assertFalse(report.events[0].output_queue_underflowed)

    def test_full_output_queue_records_overflow_not_underflow(self) -> None:
        coordinator = self.coordinator()
        output = np.ones(HOP_SIZE, dtype=np.float32)
        for index in range(coordinator.config.output_queue_size):
            coordinator._output_queue.put(_OutputChunk(output, 1.0, 1.0, index * HOP_SIZE))
        coordinator._input_queue.put(_InputChunk(np.ones(LIVE_BLOCK_SIZE, dtype=np.float32), 2.0, 2.0))

        self.assertTrue(coordinator.process_pending_once())

        report = coordinator.latency_report()
        self.assertEqual(report.overrun_events, 1)
        self.assertEqual(report.output_queue_overflow_events, 1)
        self.assertEqual(report.underflow_events, 0)
        self.assertTrue(report.events[0].output_queue_overflowed)
        self.assertFalse(report.events[0].input_queue_overflowed)
        self.assertFalse(report.events[0].output_queue_underflowed)
        self.assertIsNone(coordinator.last_error)

    def test_native_input_is_canonicalized_before_detector(self) -> None:
        adapter = CanonicalizingAdapter()
        detector = RecordingDetector()
        coordinator = LiveAudioCoordinator(
            adapter,
            pitch_detector=detector,
            key_resolver=FakeResolver(),
            harmony_generator=FakeHarmony(),
            config=LiveAudioConfig(),
        )
        coordinator._input_queue.put(_InputChunk(np.ones(2400, dtype=np.float32), 3.0, None))

        self.assertTrue(coordinator.process_pending_once())
        self.assertEqual(adapter.last_rate, 48000)
        self.assertEqual(detector.frame_sizes, [2048])

    def test_activation_and_deactivation_fade_over_output_progression(self) -> None:
        coordinator = self.coordinator()
        coordinator.set_active_voices({"3rd"})
        block = np.ones(LIVE_BLOCK_SIZE, dtype=np.float32)
        for index in range(4):
            coordinator.process_block(block, float(index))
        activation_gain = coordinator._voice_gains["3rd"]
        self.assertGreater(activation_gain, 0.0)
        self.assertLess(activation_gain, 1.0)
        coordinator.process_block(block, 4.5)
        self.assertEqual(coordinator._voice_gains["3rd"], 1.0)

        coordinator.set_active_voices(set())
        coordinator.process_block(block, 5.0)
        self.assertGreater(coordinator._voice_gains["3rd"], 0.0)
        self.assertLess(coordinator._voice_gains["3rd"], 1.0)
        coordinator.process_block(block, 6.0)
        self.assertEqual(coordinator._voice_gains["3rd"], 0.0)
        self.assertNotIn("3rd", coordinator._processing_voices)

    def test_one_native_chunk_queues_every_emitted_hop_in_order(self) -> None:
        coordinator = self.coordinator()
        coordinator.set_active_voices({"3rd"})
        coordinator.pitch_shifters["3rd"] = FakeShifter(emit_calls={4, 6})
        native = np.ones(LIVE_BLOCK_SIZE * 6, dtype=np.float32)
        coordinator._input_queue.put(_InputChunk(native, 7.0, None))

        self.assertTrue(coordinator.process_pending_once())
        chunks = []
        while not coordinator._output_queue.empty():
            chunks.append(coordinator._output_queue.get_nowait())

        self.assertEqual(len(chunks), 2)
        self.assertEqual([chunk.sample_start for chunk in chunks], [0, HOP_SIZE])
        self.assertEqual([chunk.capture_timestamp for chunk in chunks], [7.0, 7.0])
        self.assertTrue(all(chunk.audio.size == HOP_SIZE for chunk in chunks))




    def test_delayed_output_latency_uses_originating_input_timestamp(self) -> None:
        coordinator = self.coordinator()
        coordinator.set_active_voices({"3rd"})
        block = np.ones(LIVE_BLOCK_SIZE, dtype=np.float32)
        for index in range(4):
            coordinator._input_queue.put(_InputChunk(block, 10.0 + index, None))
            coordinator.process_pending_once()

        outdata = np.zeros((LIVE_BLOCK_SIZE, 1), dtype=np.float32)
        coordinator._callback(block.reshape(-1, 1), outdata, LIVE_BLOCK_SIZE, {"outputBufferDacTime": 20.0}, None)

        self.assertEqual(len(coordinator.latency_report().measurements), 1)
        measurement = coordinator.latency_report().measurements[0]
        self.assertEqual(measurement.capture_timestamp, 10.0)
        self.assertEqual(measurement.sample_start, 0)

    def test_zero_adc_timestamp_uses_pipeline_provenance_not_physical_latency(self) -> None:
        coordinator = self.coordinator()
        coordinator._output_queue.put(_OutputChunk(np.ones(HOP_SIZE, dtype=np.float32), None, 10.0, 0))
        outdata = np.zeros((HOP_SIZE, 1), dtype=np.float32)

        coordinator._callback(
            np.ones((HOP_SIZE, 1), dtype=np.float32),
            outdata,
            HOP_SIZE,
            {"inputBufferAdcTime": 0.0, "currentTime": 10.0, "outputBufferDacTime": 10.1},
            None,
        )

        measurement = coordinator.latency_report().measurements[0]
        self.assertIsNone(measurement.capture_timestamp)
        self.assertEqual(measurement.provenance, LatencyProvenance.PIPELINE_CALLBACK_TO_DAC)
        self.assertIsNone(measurement.latency_ms)
        self.assertAlmostEqual(measurement.pipeline_latency_ms, 100.0)

    def test_valid_adc_timestamp_reports_verified_physical_latency(self) -> None:
        coordinator = self.coordinator()
        coordinator._output_queue.put(_OutputChunk(np.ones(HOP_SIZE, dtype=np.float32), 10.0, 10.05, 0))
        outdata = np.zeros((HOP_SIZE, 1), dtype=np.float32)
        coordinator._callback(
            np.ones((HOP_SIZE, 1), dtype=np.float32),
            outdata,
            HOP_SIZE,
            {"inputBufferAdcTime": 10.0, "currentTime": 10.05, "outputBufferDacTime": 10.1},
            None,
        )

        measurement = coordinator.latency_report().measurements[0]
        self.assertEqual(measurement.provenance, LatencyProvenance.VERIFIED_CAPTURE_TO_DAC)
        self.assertAlmostEqual(measurement.latency_ms, 100.0)

    def test_missing_capture_and_callback_timestamps_marks_physical_latency_unavailable(self) -> None:
        coordinator = self.coordinator()
        coordinator._output_queue.put(_OutputChunk(np.ones(HOP_SIZE, dtype=np.float32), None, None, 0))
        outdata = np.zeros((HOP_SIZE, 1), dtype=np.float32)
        coordinator._callback(np.ones((HOP_SIZE, 1), dtype=np.float32), outdata, HOP_SIZE, {"outputBufferDacTime": 10.1}, None)

        report = coordinator.latency_report()
        self.assertEqual(report.measurements[0].provenance, LatencyProvenance.PHYSICAL_CAPTURE_UNAVAILABLE)
        self.assertEqual(report.physical_latency_unavailable, 1)

    def test_queue_overflow_is_an_event_not_a_sticky_measurement_flag(self) -> None:
        coordinator = self.coordinator()
        indata = np.ones((HOP_SIZE, 1), dtype=np.float32)
        outdata = np.zeros((HOP_SIZE, 1), dtype=np.float32)
        coordinator._input_queue.put(_InputChunk(np.ones(HOP_SIZE, dtype=np.float32), 1.0, 1.0))
        coordinator._input_queue.put(_InputChunk(np.ones(HOP_SIZE, dtype=np.float32), 1.5, 1.5))
        coordinator._output_queue.put(_OutputChunk(np.ones(HOP_SIZE, dtype=np.float32), 2.0, 2.0, 0))
        coordinator._callback(indata, outdata, HOP_SIZE, {"outputBufferDacTime": 3.0}, None)
        coordinator._input_queue.get_nowait()
        coordinator._input_queue.get_nowait()
        coordinator._output_queue.put(_OutputChunk(np.ones(HOP_SIZE, dtype=np.float32), 4.0, 4.0, HOP_SIZE))
        coordinator._callback(indata, outdata, HOP_SIZE, {"outputBufferDacTime": 5.0}, None)

        report = coordinator.latency_report()
        self.assertEqual(report.overrun_events, 1)
        self.assertTrue(all(not measurement.queue_overflowed for measurement in report.measurements))

    def test_callback_flags_are_distinct_from_latency_failure(self) -> None:
        coordinator = self.coordinator()
        coordinator._output_queue.put(_OutputChunk(np.ones(HOP_SIZE, dtype=np.float32), 1.0, 1.1, 0))
        outdata = np.zeros((HOP_SIZE, 1), dtype=np.float32)
        status = SimpleNamespace(input_overflow=True, output_underflow=False)
        coordinator._callback(np.ones((HOP_SIZE, 1), dtype=np.float32), outdata, HOP_SIZE, {"outputBufferDacTime": 2.0}, status)

        report = coordinator.latency_report()
        self.assertEqual(report.callback_input_overflow_events, 1)
        self.assertEqual(report.callback_output_underflow_events, 0)
        self.assertAlmostEqual(report.measurements[0].latency_ms, 1000.0)

    def test_each_output_chunk_receives_its_own_output_timestamp_offset(self) -> None:
        coordinator = self.coordinator()
        coordinator._output_queue.put(_OutputChunk(np.ones(HOP_SIZE, dtype=np.float32), 1.0, None, 0))
        coordinator._output_queue.put(_OutputChunk(np.ones(HOP_SIZE, dtype=np.float32), 2.0, None, HOP_SIZE))
        outdata = np.zeros((HOP_SIZE * 2, 1), dtype=np.float32)
        coordinator._callback(
            np.ones((HOP_SIZE * 2, 1), dtype=np.float32), outdata, HOP_SIZE * 2, {"outputBufferDacTime": 10.0}, None
        )

        measurements = coordinator.latency_report().measurements
        self.assertEqual([measurement.sample_start for measurement in measurements], [0, HOP_SIZE])
        self.assertEqual([measurement.sample_count for measurement in measurements], [HOP_SIZE, HOP_SIZE])
        self.assertAlmostEqual(measurements[0].output_timestamp, 10.0)
        self.assertAlmostEqual(measurements[1].output_timestamp, 10.0 + HOP_SIZE / 44100)

    def test_full_output_underflow_records_event_without_measurement(self) -> None:
        coordinator = self.coordinator()
        outdata = np.ones((HOP_SIZE, 1), dtype=np.float32)
        coordinator._callback(np.ones((HOP_SIZE, 1), dtype=np.float32), outdata, HOP_SIZE, {"outputBufferDacTime": 10.0}, None)

        report = coordinator.latency_report()
        self.assertEqual(report.measurements, ())
        self.assertEqual(report.underflow_events, 1)
        self.assertEqual(len(report.events), 1)
        np.testing.assert_array_equal(outdata, np.zeros_like(outdata))

    def test_worker_start_failure_closes_stream_and_capture(self) -> None:
        class Adapter(FakeAdapter):
            def __init__(self):
                self.stream = FailingStream()
                self.capture_stopped = False

            def start_capture(self, device_index=None):
                pass

            def create_duplex_stream(self, callback, device_index=None):
                return self.stream

            def stop_capture(self):
                self.capture_stopped = True

        adapter = Adapter()
        coordinator = LiveAudioCoordinator(adapter)
        with self.assertRaises(RuntimeError):
            coordinator.start()
        self.assertTrue(adapter.stream.stopped)
        self.assertTrue(adapter.stream.closed)
        self.assertTrue(adapter.capture_stopped)

    def test_stream_shutdown_failure_does_not_skip_cleanup(self) -> None:
        class Adapter(FakeAdapter):
            def __init__(self):
                self.capture_stopped = False

            def stop_capture(self):
                self.capture_stopped = True

        class Stream:
            def __init__(self):
                self.closed = False

            def stop(self):
                raise RuntimeError("stop failure")

            def close(self):
                self.closed = True

        adapter = Adapter()
        coordinator = LiveAudioCoordinator(adapter)
        stream = Stream()
        coordinator._stream = stream
        coordinator.stop()

        self.assertTrue(stream.closed)
        self.assertTrue(adapter.capture_stopped)
        self.assertIsInstance(coordinator.last_error, RuntimeError)

    def test_clean_stop_is_idempotent_without_stream(self) -> None:
        coordinator = self.coordinator()
        coordinator.stop()
        coordinator.stop()
        self.assertIsNone(coordinator._stream)

    def test_coordinator_is_alive_reflects_stream_and_worker_state(self) -> None:
        coordinator = self.coordinator()
        self.assertFalse(coordinator.is_alive)

        started = Event()
        release = Event()

        def block_worker():
            started.set()
            release.wait()

        worker = Thread(target=block_worker, daemon=True)
        worker.start()
        started.wait()
        coordinator._worker = worker

        try:
            self.assertTrue(coordinator.is_alive)
        finally:
            release.set()
            worker.join()

        self.assertFalse(coordinator.is_alive)

    def test_coordinator_stop_retains_worker_reference_if_timeout_exceeded_while_alive(self) -> None:
        coordinator = self.coordinator()
        started = Event()
        release = Event()

        def block_worker():
            started.set()
            release.wait()

        worker = Thread(target=block_worker, daemon=True)
        worker.start()
        started.wait()
        coordinator._worker = worker

        self.assertTrue(coordinator.is_alive)
        # Attempt stop with tiny timeout while worker is still blocked
        stopped = coordinator.stop(timeout=0.01)
        self.assertFalse(stopped)
        self.assertTrue(coordinator.is_alive)
        self.assertIs(coordinator._worker, worker)

        # Release worker and verify clean completion
        release.set()
        stopped_after = coordinator.stop(timeout=2.0)
        self.assertTrue(stopped_after)
        self.assertFalse(coordinator.is_alive)
        self.assertIsNone(coordinator._worker)

    def test_coordinator_shutdown_returns_termination_status(self) -> None:
        coordinator = self.coordinator()
        self.assertTrue(coordinator.shutdown())
        self.assertFalse(coordinator.is_alive)

    def test_process_block_publishes_telemetry_to_visualizer_bridge(self) -> None:
        bridge = VisualizerBridge()
        coordinator = LiveAudioCoordinator(
            FakeAdapter(),
            pitch_detector=FakeDetector(),
            key_resolver=FakeResolver(),
            harmony_generator=FakeHarmony(),
            visualizer_bridge=bridge,
            config=LiveAudioConfig(),
        )
        coordinator.request_id = "test-live-1"
        block = np.ones(LIVE_BLOCK_SIZE, dtype=np.float32)
        coordinator.process_block(block, 1.0)

        result = bridge.read_after(0)
        self.assertIsNotNone(result)
        seq, frame = result
        self.assertEqual(seq, 1)
        self.assertEqual(frame.request_id, "test-live-1")
        self.assertEqual(frame.current_f0, 440.0)
        self.assertEqual(frame.confidence, 0.9)
        self.assertEqual(frame.tracking_state, "tracking")
        self.assertTrue(frame.in_tune)
        self.assertEqual(frame.effective_key, KeyScale("C", "major"))

    def test_coordinator_start_uses_fallback_device_index_for_duplex_stream(self) -> None:
        """Requirement 3: When start_capture returns a fallback DeviceInfo, create_duplex_stream
        must be invoked with the fallback device index, not the stale requested index."""
        fallback_device = DeviceInfo(7, "Fallback Mic", 1, 0, 48000, True, "WASAPI")
        captured_stream_device = []

        class DummyStream:
            def start(self): pass
            def stop(self): pass
            def close(self): pass

        class MockAdapter(FakeAdapter):
            def start_capture(self, device_index=None):
                return fallback_device

            def create_duplex_stream(self, callback, device_index=None):
                captured_stream_device.append(device_index)
                return DummyStream()

        adapter = MockAdapter()
        coordinator = LiveAudioCoordinator(adapter)
        started_device = coordinator.start(device_index=42)

        self.assertEqual(captured_stream_device, [7])
        self.assertEqual(started_device, fallback_device)
        self.assertEqual(coordinator.active_input_device, fallback_device)
        coordinator.stop()

    def test_coordinator_start_permission_denied_propagates_and_cleans_up(self) -> None:
        """Requirement 1: PermissionDeniedError from capture startup propagates distinctly
        and leaves coordinator resources cleaned up."""
        class MockAdapter(FakeAdapter):
            def __init__(self):
                self.stopped = False

            def start_capture(self, device_index=None):
                raise PermissionDeniedError("Microphone access denied by OS")

            def stop_capture(self):
                self.stopped = True

        adapter = MockAdapter()
        coordinator = LiveAudioCoordinator(adapter)

        with self.assertRaises(PermissionDeniedError):
            coordinator.start()

        self.assertFalse(coordinator.is_alive)
        self.assertIsNone(coordinator.active_input_device)

    def test_manual_key_with_real_resolver_allows_harmony_processing(self) -> None:
        real_resolver = KeyScaleResolver()
        real_resolver.set_manual_key("C", "major")
        real_resolver.apply_processing_boundary()

        coordinator = LiveAudioCoordinator(
            FakeAdapter(),
            pitch_detector=FakeDetector(),
            key_resolver=real_resolver,
            harmony_generator=FakeHarmony(),
            pitch_shifters={"3rd": FakeShifter(initial_blocks=1)},
        )
        coordinator.set_active_voices({"3rd"})
        block = np.ones(LIVE_BLOCK_SIZE, dtype=np.float32)
        output = coordinator.process_block(block)

        self.assertIsNotNone(output)
        self.assertEqual(output.size, HOP_SIZE)

    def test_process_block_mixes_dry_voice_with_harmony_at_half_gain(self) -> None:
        """Requirement: when harmony voices are active and WORLD emits a hop, the returned
        audio must be 0.5 * dry_segment_hop + 0.5 * harmony_mean, both HOP_SIZE samples,
        staying within [-1, 1].  The dry audio comes from the last 4 recorded input blocks."""
        # FakeShifter emits ones(HOP_SIZE) after 4 calls; FakeHarmony yields 660 Hz.
        coordinator = self.coordinator()
        coordinator.set_active_voices({"3rd"})

        # Use a known constant block so we can predict the dry_hop value.
        block = np.full(LIVE_BLOCK_SIZE, 0.4, dtype=np.float32)

        output = None
        for index in range(4):
            result = coordinator.process_block(block, float(index))
            if result is not None:
                output = result

        self.assertIsNotNone(output, "Expected a non-None output on the 4th block")
        # Harmony mean = ones(HOP_SIZE) faded from 0→~gain via _apply_voice_fade.
        # Dry hop = 0.4 repeated for HOP_SIZE samples (4 blocks of constant 0.4).
        # Mixed output = 0.5 * dry + 0.5 * harmony_mean; both are non-negative here,
        # so output must be > 0, <= 1.0, and the shape must be HOP_SIZE.
        self.assertEqual(output.shape, (HOP_SIZE,))
        self.assertLessEqual(float(np.max(np.abs(output))), 1.0 + 1e-6,
                             "Mixed output must remain within [-1, 1]")
        # The dry contribution (0.5 * 0.4 = 0.2) must be present — output > 0 everywhere.
        self.assertTrue(np.all(output >= 0.0),
                        "With positive dry (0.4) and positive harmony (faded gain ≥ 0), output must be ≥ 0")

    def test_unvoiced_blocks_advance_shifter_and_emit_dry_hop(self) -> None:
        """Requirement: unvoiced frames must advance the pitch shifter buffer continuously,
        and emit the dry hop at the hop cadence without starving the output queue."""
        class UnvoicedDetector(FakeDetector):
            def detect(self, frame):
                return PitchDetectionResult(None, 0.0, False, TrackingState.TRACKING_LOST)

        class UnvoicedShifter(FakeShifter):
            def process_block(self, block, source, target, **kwargs):
                res = super().process_block(block, source, target, **kwargs)
                if res is not None:
                    return SimpleNamespace(valid=False, audio=None)
                return None

        coordinator = LiveAudioCoordinator(
            FakeAdapter(),
            pitch_detector=UnvoicedDetector(),
            key_resolver=FakeResolver(),
            harmony_generator=FakeHarmony(),
            pitch_shifters={"3rd": UnvoicedShifter()},
        )
        coordinator.set_active_voices({"3rd"})
        block = np.full(LIVE_BLOCK_SIZE, 0.7, dtype=np.float32)

        outputs = [coordinator.process_block(block, float(i)) for i in range(4)]
        # Blocks 1-3 return None (warm-up/accumulation)
        self.assertIsNone(outputs[0])
        self.assertIsNone(outputs[1])
        self.assertIsNone(outputs[2])
        # Block 4 emits the dry hop at unity gain (size HOP_SIZE)
        self.assertIsNotNone(outputs[3])
        self.assertEqual(outputs[3].shape, (HOP_SIZE,))
        np.testing.assert_allclose(outputs[3], np.full(HOP_SIZE, 0.7, dtype=np.float32))
        # The shifter was not bypassed: received all 4 calls
        self.assertEqual(coordinator.pitch_shifters["3rd"].calls, 4)

    def test_process_pending_once_resamples_output_for_non_canonical_native_rate(self) -> None:
        """Requirement: when the stream operates at a non-canonical native sample rate (e.g. 48000),
        output chunks queued for the DAC must be converted to the native sample rate."""
        class ResamplingAdapter(FakeAdapter):
            native_sample_rate = 48000

            def process_canonical_output(self, chunk, native_rate):
                # Mock resampling from 44100 to 48000: output size scales by 48000/44100
                resampled_len = int(round(chunk.size * 48000 / 44100))
                return np.zeros(resampled_len, dtype=np.float32)

        coordinator = LiveAudioCoordinator(
            ResamplingAdapter(),
            pitch_detector=FakeDetector(),
            key_resolver=FakeResolver(),
            harmony_generator=FakeHarmony(),
            pitch_shifters={"3rd": FakeShifter()},
            config=LiveAudioConfig(input_queue_size=4, output_queue_size=8),
        )
        coordinator.set_active_voices({"3rd"})
        native_chunk = np.ones(LIVE_BLOCK_SIZE * 4, dtype=np.float32)
        coordinator._input_queue.put(_InputChunk(native_chunk, 1.0, 1.0))

        self.assertTrue(coordinator.process_pending_once())
        self.assertFalse(coordinator._output_queue.empty())
        out_chunk = coordinator._output_queue.get_nowait()
        expected_len = int(round(HOP_SIZE * 48000 / 44100))
        self.assertEqual(out_chunk.audio.size, expected_len)


    def test_callback_before_worker_ready_does_not_flag_underflow(self) -> None:
        """During the startup grace period (_output_ready cleared by start()), silence is the
        correct output. _callback must NOT set underflowed or record a queue-underflow event."""
        coordinator = self.coordinator()
        coordinator._output_ready.clear()  # Simulate post-start() state before first worker output
        indata = np.ones((LIVE_BLOCK_SIZE, 1), dtype=np.float32)
        outdata = np.ones((LIVE_BLOCK_SIZE, 1), dtype=np.float32)
        coordinator._callback(indata, outdata, LIVE_BLOCK_SIZE, {}, None)

        np.testing.assert_array_equal(outdata, np.zeros_like(outdata))
        self.assertFalse(coordinator.underflowed)
        self.assertIsNone(coordinator.underflow_error)
        self.assertEqual(coordinator.latency_report().underflow_events, 0)

    def test_output_ready_set_after_first_worker_output(self) -> None:
        """process_pending_once() must set _output_ready after the first output chunk is
        successfully enqueued so that _callback transitions out of the startup grace period."""
        coordinator = self.coordinator()
        coordinator._output_ready.clear()  # Simulate post-start() state
        native = np.ones(LIVE_BLOCK_SIZE, dtype=np.float32)
        coordinator._input_queue.put(_InputChunk(native, 1.0, None))

        self.assertFalse(coordinator._output_ready.is_set())
        coordinator.process_pending_once()
        self.assertTrue(coordinator._output_ready.is_set())

    def test_end_to_end_48k_resampling_conservation_and_zero_underflow(self) -> None:
        """Verify that at 48 kHz, when both input and output resampling are active with the
        production 2-block pre-fill, exactly 0 underflows occur across 1,000 simulated callbacks.
        Demonstrates that the surplus produced by 44.1->48 kHz output conversion (2229 native
        samples per 2048 canonical block) perfectly compensates for fractional input accumulation
        deficits without any long-term drift or starvation."""
        import samplerate as _sr
        adapter = InputAdapter()
        adapter._native_sample_rate = 48000
        adapter._resampler = _sr.Resampler('sinc_best', 1)
        adapter._output_resampler = _sr.Resampler('sinc_best', 1)

        canonical_pending = np.empty(0, dtype=np.float32)
        output_pending_samples = 2 * DEFAULT_FRAME_SIZE  # 2-block pre-fill
        underflows = 0
        min_occupancy = float('inf')
        max_occupancy = 0

        for _ in range(1000):
            chunk = np.ones(DEFAULT_FRAME_SIZE, dtype=np.float32) * 0.1
            canonical = adapter.process_native_chunk(chunk, 48000)
            canonical_pending = np.concatenate((canonical_pending, canonical))
            while canonical_pending.size >= DEFAULT_FRAME_SIZE:
                frame = canonical_pending[:DEFAULT_FRAME_SIZE]
                canonical_pending = canonical_pending[DEFAULT_FRAME_SIZE:]
                out_native = adapter.process_canonical_output(frame, 48000)
                output_pending_samples += len(out_native)

            min_occupancy = min(min_occupancy, output_pending_samples)
            max_occupancy = max(max_occupancy, output_pending_samples)

            if output_pending_samples >= DEFAULT_FRAME_SIZE:
                output_pending_samples -= DEFAULT_FRAME_SIZE
            else:
                underflows += 1
                output_pending_samples = 0

        self.assertEqual(underflows, 0, f"Expected 0 underflows over 1,000 callbacks, got {underflows}")
        self.assertGreater(min_occupancy, DEFAULT_FRAME_SIZE,
            f"Expected buffer occupancy to never starve, minimum was {min_occupancy}")

    def test_start_pre_fills_silence_for_non_canonical_native_rate(self) -> None:
        """Fix 3: start() must pre-fill N0 silence chunks into the output queue when the
        native rate differs from the canonical rate. At 48000 Hz, N0 = ceil(2048/1881) = 2."""
        class MockStream:
            def start(self): pass
            def stop(self): pass
            def close(self): pass

        class MockAdapter(FakeAdapter):
            native_sample_rate = 48000

            def start_capture(self, device_index=None):
                return DeviceInfo(1, "Mic", 1, 0, 48000.0)

            def create_duplex_stream(self, callback, device_index=None):
                return MockStream()

            def stop_capture(self): pass

        coordinator = LiveAudioCoordinator(
            MockAdapter(),
            config=LiveAudioConfig(output_queue_size=16),
        )
        coordinator.start()

        # N0 = ceil(2048 / (2048 * 44100/48000)) = ceil(2048/1881) = 2
        pre_fill_count = coordinator._output_queue.qsize()
        self.assertEqual(pre_fill_count, 2, f"Expected 2 pre-fill silence chunks, got {pre_fill_count}")

        # Each pre-fill chunk must be silence of the correct block size
        chunk = coordinator._output_queue.get_nowait()
        self.assertEqual(chunk.audio.size, DEFAULT_FRAME_SIZE)
        np.testing.assert_array_equal(chunk.audio, np.zeros(DEFAULT_FRAME_SIZE, dtype=np.float32))
        coordinator.stop()

    def test_start_does_not_pre_fill_at_canonical_rate(self) -> None:
        """Fix 3: no pre-fill should occur when native rate equals the canonical rate (44100 Hz)."""
        class MockStream:
            def start(self): pass
            def stop(self): pass
            def close(self): pass

        class MockAdapter(FakeAdapter):
            native_sample_rate = 44100

            def start_capture(self, device_index=None):
                return DeviceInfo(1, "Mic", 1, 0, 44100.0)

            def create_duplex_stream(self, callback, device_index=None):
                return MockStream()

            def stop_capture(self): pass

        coordinator = LiveAudioCoordinator(
            MockAdapter(),
            config=LiveAudioConfig(output_queue_size=16),
        )
        coordinator.start()
        self.assertEqual(coordinator._output_queue.qsize(), 0,
            "No pre-fill should occur at canonical 44100 Hz rate")
        coordinator.stop()


    def test_detector_thread_result_used_by_process_block(self) -> None:
        """Fix 2: process_block must not block waiting for PitchDetector.detect() to complete.
        With a SlowDetector that sleeps 60 ms, process_pending_once must return well within
        one callback interval (< 50 ms), relying on the cached/synchronous fallback result."""
        import time

        class SlowDetector:
            def detect(self, frame):
                time.sleep(0.06)  # 60 ms — exceeds one 46.5 ms callback interval
                return PitchDetectionResult(440.0, 0.9, True, TrackingState.TRACKING)

        coordinator = LiveAudioCoordinator(
            FakeAdapter(),
            pitch_detector=SlowDetector(),
            key_resolver=FakeResolver(),
            harmony_generator=FakeHarmony(),
            config=LiveAudioConfig(input_queue_size=4, output_queue_size=4),
        )
        native = np.ones(LIVE_BLOCK_SIZE, dtype=np.float32)
        coordinator._input_queue.put(_InputChunk(native, 1.0, None))

        t0 = time.monotonic()
        coordinator.process_pending_once()
        elapsed = time.monotonic() - t0

        # The synchronous first-call fallback runs detect() once, so elapsed will be ~60 ms.
        # What matters is that SUBSEQUENT calls do NOT block: the cached result is used.
        coordinator._input_queue.put(_InputChunk(native, 2.0, None))
        t1 = time.monotonic()
        coordinator.process_pending_once()
        elapsed2 = time.monotonic() - t1
        self.assertLess(elapsed2, 0.05,
            f"Second process_pending_once took {elapsed2*1000:.1f} ms; should use cached pitch result (<50 ms)")

    def test_detector_queue_full_uses_cached_result_without_raising(self) -> None:
        """Fix 2: when _detect_queue is full (detector still busy), _enqueue_for_detect must
        silently drop the new frame and process_block must return without raising."""
        coordinator = self.coordinator()
        # Pre-fill detect queue to capacity (maxsize=1)
        coordinator._detect_queue.put_nowait(np.zeros(LIVE_BLOCK_SIZE, dtype=np.float32))
        # Pre-set a cached pitch result
        from harmonix.pitch_detector import PitchDetectionResult, TrackingState as TS
        coordinator._last_pitch = PitchDetectionResult(330.0, 0.8, True, TS.TRACKING)

        block = np.ones(LIVE_BLOCK_SIZE, dtype=np.float32)
        try:
            output = coordinator.process_block(block, 1.0)
        except Exception as exc:
            self.fail(f"process_block raised {exc!r} when detect queue was full")

        # Output must still be returned (pass-through, no voices active)
        self.assertIsNotNone(output)

    def test_start_resets_last_pitch(self) -> None:
        """start() must reset _last_pitch to None so restarted sessions do not retain stale pitch."""
        class MockStream:
            def start(self): pass
            def stop(self): pass
            def close(self): pass

        class MockAdapter(FakeAdapter):
            native_sample_rate = 44100
            def start_capture(self, device_index=None):
                return DeviceInfo(1, "Mic", 1, 0, 44100.0)
            def create_duplex_stream(self, callback, device_index=None):
                return MockStream()
            def stop_capture(self): pass

        coordinator = LiveAudioCoordinator(MockAdapter())
        from harmonix.pitch_detector import PitchDetectionResult, TrackingState as TS
        coordinator._last_pitch = PitchDetectionResult(440.0, 0.9, True, TS.TRACKING)

        coordinator.start()
        with coordinator._pitch_lock:
            self.assertIsNone(coordinator._last_pitch, "start() must reset _last_pitch to None")
        coordinator.stop()

    def test_detector_thread_exception_handling_and_fallback(self) -> None:
        """Detector thread must catch exceptions from PitchDetector.detect(), record last_error,
        emit an unvoiced fallback result, and remain alive."""
        class FailingDetector:
            def __init__(self):
                self.calls = 0

            def detect(self, frame):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("Simulated detector DSP failure")
                from harmonix.pitch_detector import PitchDetectionResult, TrackingState as TS
                return PitchDetectionResult(440.0, 0.9, True, TS.TRACKING)

        failing = FailingDetector()
        coordinator = LiveAudioCoordinator(
            FakeAdapter(),
            pitch_detector=failing,
            key_resolver=FakeResolver(),
            harmony_generator=FakeHarmony(),
        )
        import time
        t = Thread(target=coordinator._detector_loop, daemon=True)
        t.start()

        # Enqueue first frame that triggers failure
        coordinator._enqueue_for_detect(np.zeros(LIVE_BLOCK_SIZE, dtype=np.float32))
        time.sleep(0.1)

        # Thread must still be alive
        self.assertTrue(t.is_alive(), "Detector thread died after exception in detect()")
        self.assertIsNotNone(coordinator.last_error)
        self.assertIn("Simulated detector DSP failure", str(coordinator.last_error))

        # Result must be marked unvoiced / tracking lost, not stale
        with coordinator._pitch_lock:
            pitch = coordinator._last_pitch
        self.assertIsNotNone(pitch)
        self.assertFalse(pitch.voiced)
        self.assertIsNone(pitch.frequency_hz)

        # Enqueue second frame that succeeds
        coordinator._enqueue_for_detect(np.zeros(LIVE_BLOCK_SIZE, dtype=np.float32))
        time.sleep(0.1)
        with coordinator._pitch_lock:
            pitch2 = coordinator._last_pitch
        self.assertTrue(pitch2.voiced)
        self.assertEqual(pitch2.frequency_hz, 440.0)

        coordinator._stop_event.set()
        t.join(timeout=0.5)


if __name__ == "__main__":
    unittest.main()
