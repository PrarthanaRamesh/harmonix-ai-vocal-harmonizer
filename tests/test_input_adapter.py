import unittest
from unittest.mock import patch

import numpy as np

from harmonix.config.validation import validate_frame_size
from harmonix.config.validation import format_audio_device_selector
from harmonix.domain.errors import (
    CaptureStartError,
    DeviceUnavailableError,
    NoInputDeviceError,
    PermissionDeniedError,
    QueueOverflowError,
    StaleSelectedDeviceError,
)
from harmonix.input.adapter import DeviceInfo, InputAdapter


class InputAdapterTests(unittest.TestCase):
    def test_validate_frame_size_accepts_canonical_and_bounds(self) -> None:
        validate_frame_size(256)
        validate_frame_size(2048)
        validate_frame_size(4096)

        with self.assertRaises(ValueError):
            validate_frame_size(255)

        with self.assertRaises(ValueError):
            validate_frame_size(4097)

    def test_list_devices_returns_structured_device_entries(self) -> None:
        devices = InputAdapter.list_devices()
        self.assertIsInstance(devices, list)
        if devices:
            device = devices[0]
            self.assertIsInstance(device, DeviceInfo)
            self.assertGreaterEqual(device.index, 0)
            self.assertTrue(device.name)
            self.assertGreaterEqual(device.max_input_channels, 1)
            self.assertGreater(device.default_samplerate, 0)

    def test_input_listing_preserves_filter_and_output_listing_adds_output_devices(self) -> None:
        devices = [
            {"name": "USB Mic", "hostapi": 0, "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 48000},
            {"name": "Speakers", "hostapi": 1, "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 44100},
            {"name": "Duplex", "hostapi": 0, "max_input_channels": 1, "max_output_channels": 2, "default_samplerate": 44100},
        ]
        with patch("harmonix.input.adapter.sd.query_devices", return_value=devices), patch(
            "harmonix.input.adapter.sd.query_hostapis", return_value=[{"name": "WASAPI"}, {"name": "MME"}]
        ), patch.object(InputAdapter, "_default_index", side_effect=lambda direction: 0 if direction == "input" else 1):
            inputs = InputAdapter.list_devices()
            outputs = InputAdapter.list_output_devices()

        self.assertEqual([item.name for item in inputs], ["USB Mic", "Duplex"])
        self.assertEqual([item.name for item in outputs], ["Speakers", "Duplex"])
        self.assertEqual(inputs[0].host_api_name, "WASAPI")
        self.assertEqual(outputs[0].host_api_name, "MME")
        self.assertTrue(inputs[0].is_default)
        self.assertTrue(outputs[0].is_default)

    def test_stream_uses_unique_selectors_capability_checks_and_requested_blocksize(self) -> None:
        input_device = DeviceInfo(2, "USB Mic", 1, 0, 48000, False, "WASAPI")
        output_device = DeviceInfo(4, "Speakers", 0, 2, 48000, False, "MME")
        adapter = InputAdapter(
            input_selector=format_audio_device_selector("USB Mic", "WASAPI"),
            output_selector=format_audio_device_selector("Speakers", "MME"),
            frame_size=1024,
        )
        with patch.object(InputAdapter, "list_devices", return_value=[input_device]), patch.object(
            InputAdapter, "list_output_devices", return_value=[output_device]
        ), patch("harmonix.input.adapter.sd.check_input_settings") as check_input, patch(
            "harmonix.input.adapter.sd.check_output_settings"
        ) as check_output, patch("harmonix.input.adapter.sd.Stream", return_value="stream") as stream:
            result = adapter.create_duplex_stream(lambda *args: None)

        self.assertEqual(result, "stream")
        check_input.assert_called_once_with(device=2, channels=1, dtype="float32", samplerate=48000)
        check_output.assert_called_once_with(device=4, channels=1, dtype="float32", samplerate=48000)
        self.assertEqual(stream.call_args.kwargs["device"], (2, 4))
        self.assertEqual(stream.call_args.kwargs["blocksize"], 1024)

    def test_none_selectors_resolve_current_system_defaults_explicitly(self) -> None:
        input_device = DeviceInfo(2, "Default Mic", 1, 0, 44100, True, "WASAPI")
        output_device = DeviceInfo(4, "Default Speakers", 0, 2, 44100, True, "WASAPI")
        adapter = InputAdapter(input_selector=None, output_selector=None, frame_size=256)
        with patch.object(InputAdapter, "_default_index", side_effect=lambda direction: 2 if direction == "input" else 4), patch.object(
            InputAdapter, "list_devices", return_value=[input_device]
        ), patch.object(InputAdapter, "list_output_devices", return_value=[output_device]), patch(
            "harmonix.input.adapter.sd.check_input_settings"
        ), patch("harmonix.input.adapter.sd.check_output_settings"), patch(
            "harmonix.input.adapter.sd.Stream", return_value="stream"
        ) as stream:
            self.assertEqual(adapter.create_duplex_stream(lambda *args: None), "stream")

        self.assertEqual(stream.call_args.kwargs["device"], (2, 4))
        self.assertEqual(stream.call_args.kwargs["blocksize"], 256)

    def test_missing_or_ambiguous_saved_device_fails_before_stream_creation(self) -> None:
        selector = format_audio_device_selector("USB Mic", "WASAPI")
        adapter = InputAdapter(input_selector=selector)
        with patch.object(InputAdapter, "list_devices", return_value=[]), patch(
            "harmonix.input.adapter.sd.Stream"
        ) as stream:
            with self.assertRaises(DeviceUnavailableError):
                adapter.create_duplex_stream(lambda *args: None)
            stream.assert_not_called()

        matching = DeviceInfo(2, "USB Mic", 1, 0, 48000, False, "WASAPI")
        duplicate = DeviceInfo(5, "USB Mic", 1, 0, 48000, False, "WASAPI")
        with patch.object(InputAdapter, "list_devices", return_value=[matching, duplicate]), patch(
            "harmonix.input.adapter.sd.Stream"
        ) as stream:
            with self.assertRaises(ValueError):
                adapter.create_duplex_stream(lambda *args: None)
            stream.assert_not_called()

    def test_stream_creation_failure_clears_capture_attempt_state(self) -> None:
        device = DeviceInfo(2, "USB Mic", 1, 0, 48000, False, "WASAPI")
        adapter = InputAdapter(device_index=2)
        with patch.object(InputAdapter, "list_devices", return_value=[device]), patch.object(
            InputAdapter, "list_output_devices", return_value=[DeviceInfo(4, "Speakers", 0, 2, 48000, False, "MME")]
        ), patch.object(InputAdapter, "_default_index", side_effect=lambda direction: 2 if direction == "input" else 4), patch("harmonix.input.adapter.samplerate.Resampler", return_value=object()), patch(
            "harmonix.input.adapter.sd.check_input_settings"
        ), patch("harmonix.input.adapter.sd.check_output_settings"), patch(
            "harmonix.input.adapter.sd.Stream", side_effect=RuntimeError("device open failed")
        ):
            adapter.start_capture()
            self.assertTrue(adapter.is_capturing())
            with self.assertRaises(CaptureStartError):
                adapter.create_duplex_stream(lambda *args: None)

        self.assertFalse(adapter.is_capturing())
        self.assertIsNone(adapter.native_sample_rate)

    def test_capability_check_failure_clears_capture_attempt_without_constructing_stream(self) -> None:
        input_device = DeviceInfo(2, "USB Mic", 1, 0, 48000, False, "WASAPI")
        output_device = DeviceInfo(4, "Speakers", 0, 2, 48000, False, "MME")
        adapter = InputAdapter(device_index=2)
        with patch.object(InputAdapter, "list_devices", return_value=[input_device]), patch.object(
            InputAdapter, "list_output_devices", return_value=[output_device]
        ), patch.object(InputAdapter, "_default_index", side_effect=lambda direction: 2 if direction == "input" else 4), patch(
            "harmonix.input.adapter.samplerate.Resampler", return_value=object()
        ), patch("harmonix.input.adapter.sd.check_input_settings", side_effect=ValueError("unsupported format")), patch(
            "harmonix.input.adapter.sd.check_output_settings"
        ), patch("harmonix.input.adapter.sd.Stream") as stream:
            adapter.start_capture()
            with self.assertRaises(CaptureStartError):
                adapter.create_duplex_stream(lambda *args: None)

        self.assertFalse(adapter.is_capturing())
        self.assertIsNone(adapter.native_sample_rate)
        stream.assert_not_called()

    def test_stream_block_partitions_still_emit_only_canonical_2048_frames(self) -> None:
        for stream_block_size in (256, 1024, 4096):
            with self.subTest(stream_block_size=stream_block_size):
                adapter = InputAdapter(frame_size=stream_block_size)
                signal = np.arange(8192, dtype=np.float32)
                frames = []
                for start in range(0, signal.size, stream_block_size):
                    frames.extend(adapter.emit_canonical_frames(signal[start : start + stream_block_size], 44100))
                self.assertEqual([frame.size for frame in frames], [2048] * 4)
                np.testing.assert_array_equal(np.concatenate(frames), signal)

    def test_no_input_device_is_reported_cleanly(self) -> None:
        adapter = InputAdapter(device_index=None)
        with patch.object(InputAdapter, "list_devices", return_value=[]):
            self.assertFalse(adapter.has_input_device())

            with self.assertRaises(NoInputDeviceError):
                adapter.require_selected_device()

    def test_stale_selected_device_is_detected(self) -> None:
        adapter = InputAdapter(device_index=999999)
        with self.assertRaises(StaleSelectedDeviceError):
            adapter.require_selected_device()

    def test_mono_float32_normalization_and_multichannel_downmix(self) -> None:
        frame = np.array(
            [
                [1.0, 0.0],
                [2.0, 0.5],
                [3.0, 1.0],
            ],
            dtype=np.float32,
        )

        mono = InputAdapter.normalize_to_mono_float32(frame)

        self.assertEqual(mono.dtype, np.float32)
        np.testing.assert_allclose(mono, np.array([0.5, 1.25, 2.0], dtype=np.float32))

    def test_native_44100_chunk_is_not_resampled(self) -> None:
        adapter = InputAdapter(device_index=None)

        chunk = np.linspace(0.0, 1.0, 2048, dtype=np.float32)
        out = adapter.process_native_chunk(chunk, sample_rate=44100)

        self.assertEqual(len(out), 2048)
        self.assertEqual(out.dtype, np.float32)

    def test_stateful_resampler_across_chunks_preserves_stream_continuity(self) -> None:
        adapter = InputAdapter(device_index=None)
        native = np.linspace(-1.0, 1.0, 4096, dtype=np.float32)
        chunk_a = native[:2048]
        chunk_b = native[2048:]

        out_a = adapter.process_native_chunk(chunk_a, sample_rate=22050)
        out_b = adapter.process_native_chunk(chunk_b, sample_rate=22050)

        combined = np.concatenate([out_a, out_b])
        self.assertGreater(len(combined), 0)
        self.assertEqual(combined.dtype, np.float32)
        self.assertFalse(np.isnan(combined).any())

    def test_canonical_frame_assembly_uses_2048_samples(self) -> None:
        adapter = InputAdapter(device_index=None)
        full = np.arange(4096, dtype=np.float32) / 4096.0
        frames = adapter.emit_canonical_frames(full, sample_rate=44100)

        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0].shape[0], 2048)
        self.assertEqual(frames[1].shape[0], 2048)

    def test_partial_final_frame_is_retained_or_rejected_without_fabrication(self) -> None:
        adapter = InputAdapter(device_index=None)
        partial = np.arange(1024, dtype=np.float32)

        frames = adapter.emit_canonical_frames(partial, sample_rate=44100)
        self.assertEqual(len(frames), 0)

    def test_queue_full_behaves_deterministically(self) -> None:
        adapter = InputAdapter(device_index=None, queue_size=1)
        adapter.enqueue_native_chunk(np.ones(64, dtype=np.float32))
        with self.assertRaises(QueueOverflowError):
            adapter.enqueue_native_chunk(np.ones(64, dtype=np.float32))

        self.assertTrue(adapter.queue_overflowed)

    def test_unavailable_device_raises_device_error(self) -> None:
        adapter = InputAdapter(device_index=-1)
        with self.assertRaises(DeviceUnavailableError):
            adapter.require_selected_device()

    def test_stream_start_failure_is_reported(self) -> None:
        # With the EC-1.5 fix, a stale device_index triggers the fallback path.
        # When there is no system default, NoInputDeviceError is raised (a more
        # specific DeviceError) — the spec requires a clear, distinct error.
        adapter = InputAdapter(device_index=None)
        with self.assertRaises(NoInputDeviceError):
            adapter.start_capture(device_index=999999)

    def test_clean_shutdown_does_not_crash(self) -> None:
        adapter = InputAdapter(device_index=None)
        adapter.stop_capture()
        self.assertFalse(adapter.is_capturing())

    def test_queue_overflow_error_is_raised_on_explicit_overflow_path(self) -> None:
        adapter = InputAdapter(device_index=None, queue_size=1)
        adapter.enqueue_native_chunk(np.ones(128, dtype=np.float32))
        with self.assertRaises(QueueOverflowError):
            adapter.enqueue_native_chunk(np.ones(128, dtype=np.float32))


    # ------------------------------------------------------------------
    # EC-1.2 — permission denied
    # ------------------------------------------------------------------

    def test_permission_denied_during_start_capture_raises_distinct_error_not_capture_start_error(self) -> None:
        """EC-1.2: an OS PermissionError during capture startup must become PermissionDeniedError."""
        device = DeviceInfo(3, "Mic", 1, 0, 44100, True, "WASAPI")
        adapter = InputAdapter(device_index=3)
        with patch.object(InputAdapter, "list_devices", return_value=[device]), \
             patch("harmonix.input.adapter.samplerate.Resampler", side_effect=PermissionError("mic blocked")):
            with self.assertRaises(PermissionDeniedError) as ctx:
                adapter.start_capture()

        # Must NOT be wrapped in CaptureStartError.
        self.assertNotIsInstance(ctx.exception, CaptureStartError)
        # Original OS error is chained.
        self.assertIsInstance(ctx.exception.__cause__, PermissionError)
        # Resources must be cleaned up.
        self.assertFalse(adapter.is_capturing())
        self.assertIsNone(adapter.native_sample_rate)

    def test_permission_denied_during_start_capture_allows_retry_without_restart(self) -> None:
        """EC-1.2: after a permission-denied error, a subsequent call succeeds when access is restored."""
        device = DeviceInfo(3, "Mic", 1, 0, 44100, True, "WASAPI")
        adapter = InputAdapter(device_index=3)
        call_count = {"n": 0}

        def resampler_side_effect(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise PermissionError("mic blocked")
            # Use real samplerate outside the patch scope.
            return None  # Resampler not needed beyond initialization for this test

        with patch.object(InputAdapter, "list_devices", return_value=[device]):
            with patch("harmonix.input.adapter.samplerate.Resampler", side_effect=resampler_side_effect):
                with self.assertRaises(PermissionDeniedError):
                    adapter.start_capture()
                self.assertFalse(adapter.is_capturing())

                # Retry — must succeed.
                result = adapter.start_capture()

        self.assertTrue(adapter.is_capturing())
        self.assertIsNotNone(result)
        self.assertEqual(result.index, 3)

    def test_permission_denied_during_duplex_stream_creation_raises_distinct_error_not_capture_start_error(self) -> None:
        """EC-1.2: a PermissionError from sd.Stream() must become PermissionDeniedError."""
        input_device = DeviceInfo(2, "Mic", 1, 0, 48000, False, "WASAPI")
        output_device = DeviceInfo(4, "Speakers", 0, 2, 48000, False, "MME")
        adapter = InputAdapter(device_index=2)
        with patch.object(InputAdapter, "list_devices", return_value=[input_device]), \
             patch.object(InputAdapter, "list_output_devices", return_value=[output_device]), \
             patch.object(InputAdapter, "_default_index", side_effect=lambda d: 2 if d == "input" else 4), \
             patch("harmonix.input.adapter.sd.check_input_settings"), \
             patch("harmonix.input.adapter.sd.check_output_settings"), \
             patch("harmonix.input.adapter.sd.Stream", side_effect=PermissionError("device locked")):
            adapter.start_capture()  # succeeds independently
            with self.assertRaises(PermissionDeniedError) as ctx:
                adapter.create_duplex_stream(lambda *a: None)

        # Must NOT be wrapped in CaptureStartError.
        self.assertNotIsInstance(ctx.exception, CaptureStartError)
        self.assertIsInstance(ctx.exception.__cause__, PermissionError)
        # stop_capture() must have been called by create_duplex_stream.
        self.assertFalse(adapter.is_capturing())
        self.assertIsNone(adapter.native_sample_rate)

    def test_permission_denied_on_check_input_settings_raises_distinct_error(self) -> None:
        """EC-1.2: a PermissionError from sd.check_input_settings() also becomes PermissionDeniedError."""
        input_device = DeviceInfo(2, "Mic", 1, 0, 44100, False, "WASAPI")
        output_device = DeviceInfo(4, "Speakers", 0, 2, 44100, False, "MME")
        adapter = InputAdapter(device_index=2)
        with patch.object(InputAdapter, "list_devices", return_value=[input_device]), \
             patch.object(InputAdapter, "list_output_devices", return_value=[output_device]), \
             patch.object(InputAdapter, "_default_index", side_effect=lambda d: 2 if d == "input" else 4), \
             patch("harmonix.input.adapter.sd.check_input_settings", side_effect=PermissionError("access denied")), \
             patch("harmonix.input.adapter.sd.check_output_settings"), \
             patch("harmonix.input.adapter.sd.Stream") as mock_stream:
            with self.assertRaises(PermissionDeniedError):
                adapter.create_duplex_stream(lambda *a: None)
            mock_stream.assert_not_called()

        self.assertFalse(adapter.is_capturing())

    # ------------------------------------------------------------------
    # EC-1.5 — stale saved device fallback
    # ------------------------------------------------------------------

    def test_stale_saved_device_falls_back_to_system_default_and_returns_fallback_device(self) -> None:
        """EC-1.5: when the saved device is stale, start_capture falls back to the OS default
        and returns the fallback DeviceInfo so the caller can update the persisted setting."""
        default_device = DeviceInfo(1, "Built-in Mic", 1, 0, 44100, True, "WASAPI")
        adapter = InputAdapter(device_index=999)  # stale index
        with patch.object(InputAdapter, "list_devices", return_value=[default_device]), \
             patch.object(InputAdapter, "_default_index", side_effect=lambda d: 1 if d == "input" else None), \
             patch("harmonix.input.adapter.samplerate.Resampler", return_value=object()):
            result = adapter.start_capture()

        # Fallback succeeded — capture is active.
        self.assertTrue(adapter.is_capturing())
        self.assertEqual(adapter.native_sample_rate, 44100)
        # The fallback DeviceInfo is returned (caller can update the setting).
        self.assertIsNotNone(result)
        self.assertEqual(result.index, 1)
        self.assertEqual(result.name, "Built-in Mic")

    def test_stale_saved_device_with_no_system_default_raises_no_input_device_error(self) -> None:
        """EC-1.5: if the saved device is stale AND no system default exists, raise NoInputDeviceError."""
        adapter = InputAdapter(device_index=999)
        with patch.object(InputAdapter, "list_devices", return_value=[]), \
             patch.object(InputAdapter, "_default_index", return_value=None):
            with self.assertRaises(NoInputDeviceError) as ctx:
                adapter.start_capture()

        self.assertIn("stale", str(ctx.exception).lower())
        self.assertIn("no system-default", str(ctx.exception).lower())
        self.assertFalse(adapter.is_capturing())
        self.assertIsNone(adapter.native_sample_rate)

    def test_stale_saved_device_with_unavailable_default_raises_no_input_device_error(self) -> None:
        """EC-1.5: if the saved device is stale AND the system default is also unavailable,
        raise NoInputDeviceError — never silently select an unrelated device."""
        adapter = InputAdapter(device_index=999)
        # _default_index returns an index, but that index doesn't appear in list_devices.
        with patch.object(InputAdapter, "list_devices", return_value=[]), \
             patch.object(InputAdapter, "_default_index", side_effect=lambda d: 5 if d == "input" else None):
            with self.assertRaises(NoInputDeviceError) as ctx:
                adapter.start_capture()

        self.assertIn("also unavailable", str(ctx.exception).lower())
        self.assertFalse(adapter.is_capturing())
        self.assertIsNone(adapter.native_sample_rate)

    def test_successful_fallback_returns_distinct_device_info_from_original_saved_index(self) -> None:
        """EC-1.5: the returned DeviceInfo index must be the fallback device, not the stale saved one."""
        fallback_device = DeviceInfo(7, "Webcam Mic", 1, 0, 48000, True, "WASAPI")
        adapter = InputAdapter(device_index=42)  # stale
        with patch.object(InputAdapter, "list_devices", return_value=[fallback_device]), \
             patch.object(InputAdapter, "_default_index", side_effect=lambda d: 7 if d == "input" else None), \
             patch("harmonix.input.adapter.samplerate.Resampler", return_value=object()):
            result = adapter.start_capture()

        self.assertIsNotNone(result)
        self.assertEqual(result.index, 7)  # fallback, not 42
        self.assertNotEqual(result.index, adapter.device_index)

    def test_stale_fallback_is_retryable_without_app_restart(self) -> None:
        """EC-1.5: after a failed attempt (no default available), adapter can be retried
        once the default is restored, without reconstructing the adapter."""
        adapter = InputAdapter(device_index=999)
        default_device = DeviceInfo(1, "Built-in Mic", 1, 0, 44100, True, "WASAPI")

        # First attempt: no default available.
        with patch.object(InputAdapter, "list_devices", return_value=[]), \
             patch.object(InputAdapter, "_default_index", return_value=None):
            with self.assertRaises(NoInputDeviceError):
                adapter.start_capture()
        self.assertFalse(adapter.is_capturing())

        # Second attempt after default becomes available: should succeed.
        with patch.object(InputAdapter, "list_devices", return_value=[default_device]), \
             patch.object(InputAdapter, "_default_index", side_effect=lambda d: 1 if d == "input" else None), \
             patch("harmonix.input.adapter.samplerate.Resampler", return_value=object()):
            result = adapter.start_capture()

        self.assertTrue(adapter.is_capturing())
        self.assertIsNotNone(result)
        self.assertEqual(result.index, 1)

    def test_non_stale_device_error_is_not_treated_as_fallback_trigger(self) -> None:
        """EC-1.5 boundary: a DeviceUnavailableError (not StaleSelectedDeviceError) must NOT
        trigger the fallback path — it must propagate directly.
        Use a patched require_selected_device that raises DeviceUnavailableError directly,
        as occurs when device_index is None and the OS default is negative/unavailable."""
        adapter = InputAdapter(device_index=None)  # uses require_selected_device path
        with patch.object(
            InputAdapter,
            "require_selected_device",
            side_effect=DeviceUnavailableError("device reports invalid rate"),
        ):
            with self.assertRaises(DeviceUnavailableError):
                adapter.start_capture()

        self.assertFalse(adapter.is_capturing())

    def test_stale_input_selector_triggers_fallback_and_sets_flag(self) -> None:
        """EC-1.5: a saved input_selector that is no longer in the device list must
        trigger the system-default fallback and mark fallback_occurred = True."""
        fallback_device = DeviceInfo(2, "Default Mic", 1, 0, 44100, True, "WASAPI")
        stale_selector = format_audio_device_selector("Unplugged Mic", "WASAPI")
        adapter = InputAdapter(input_selector=stale_selector)

        with patch.object(InputAdapter, "list_devices", return_value=[fallback_device]), \
             patch.object(InputAdapter, "_default_index", side_effect=lambda d: 2 if d == "input" else None), \
             patch("harmonix.input.adapter.samplerate.Resampler", return_value=object()):
            result = adapter.start_capture()

        self.assertTrue(adapter.is_capturing())
        self.assertTrue(adapter.fallback_occurred)
        self.assertIsNotNone(result)
        self.assertEqual(result.index, 2)
        self.assertEqual(result.name, "Default Mic")

    def test_valid_device_does_not_set_fallback_flag(self) -> None:
        """When the selected device is available, fallback_occurred must remain False."""
        device = DeviceInfo(1, "Valid Mic", 1, 0, 44100, True, "WASAPI")
        adapter = InputAdapter(device_index=1)

        with patch.object(InputAdapter, "list_devices", return_value=[device]), \
             patch("harmonix.input.adapter.samplerate.Resampler", return_value=object()):
            result = adapter.start_capture()

        self.assertTrue(adapter.is_capturing())
        self.assertFalse(adapter.fallback_occurred)
        self.assertIsNotNone(result)
        self.assertEqual(result.index, 1)


if __name__ == "__main__":
    unittest.main()
