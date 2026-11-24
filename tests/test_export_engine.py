import tempfile
import unittest
from concurrent.futures import CancelledError
import os
from pathlib import Path
from threading import Event, Thread
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile

from harmonix.domain.errors import ExportValidationError, ExportWriteError
from harmonix.domain.models import RecordedBuffers
from harmonix.export.engine import ExportCancellation, ExportEngine


class ExportEngineTests(unittest.TestCase):
    def test_cancelled_export_before_render_preserves_existing_destination(self) -> None:
        recorded = RecordedBuffers(
            original=np.zeros(256, dtype=np.float32),
            a_per_voice={},
            sample_rate=44100,
        )
        cancel_event = Event()
        cancel_event.set()
        with tempfile.TemporaryDirectory() as tempdir:
            destination = Path(tempdir) / "existing.wav"
            existing_contents = b"existing destination"
            destination.write_bytes(existing_contents)

            with self.assertRaises(CancelledError):
                ExportEngine().export(
                    recorded,
                    destination,
                    enabled_voices=(),
                    cancel_event=cancel_event,
                )

            self.assertEqual(destination.read_bytes(), existing_contents)
            self.assertEqual(list(Path(tempdir).glob("harmonix-export-*.tmp")), [])

    def test_cancellation_after_temp_write_cleans_temp_and_preserves_destination(self) -> None:
        recorded = RecordedBuffers(
            original=np.zeros(256, dtype=np.float32),
            a_per_voice={},
            sample_rate=44100,
        )
        cancel_event = ExportCancellation()
        real_write = wavfile.write

        def write_then_cancel(temp_path, sample_rate, audio):
            real_write(temp_path, sample_rate, audio)
            self.assertTrue(cancel_event.cancel())

        with tempfile.TemporaryDirectory() as tempdir:
            destination = Path(tempdir) / "existing.wav"
            existing_contents = b"existing destination"
            destination.write_bytes(existing_contents)

            with patch("harmonix.export.engine.wavfile.write", side_effect=write_then_cancel):
                with self.assertRaises(CancelledError):
                    ExportEngine().export(
                        recorded,
                        destination,
                        enabled_voices=(),
                        cancel_event=cancel_event,
                    )

            self.assertEqual(destination.read_bytes(), existing_contents)
            self.assertEqual(list(Path(tempdir).glob("harmonix-export-*.tmp")), [])

    def test_replacement_winning_cancellation_race_reports_completion(self) -> None:
        recorded = RecordedBuffers(
            original=np.array([0.25, -0.25], dtype=np.float32),
            a_per_voice={},
            sample_rate=44100,
        )
        cancel_event = ExportCancellation()
        real_replace = os.replace
        cancellation_attempting = Event()
        cancel_result = []
        cancellation_threads = []

        def cancel_after_commit():
            cancellation_attempting.set()
            cancel_result.append(cancel_event.cancel())

        def replace_then_cancel(source, destination):
            real_replace(source, destination)
            cancellation_thread = Thread(target=cancel_after_commit)
            cancellation_threads.append(cancellation_thread)
            cancellation_thread.start()
            self.assertTrue(cancellation_attempting.wait(2.0))

        with tempfile.TemporaryDirectory() as tempdir:
            destination = Path(tempdir) / "mix.wav"
            with patch("harmonix.export.engine.os.replace", side_effect=replace_then_cancel):
                output_path = ExportEngine().export(
                    recorded,
                    destination,
                    enabled_voices=(),
                    cancel_event=cancel_event,
                )

            for cancellation_thread in cancellation_threads:
                cancellation_thread.join(2.0)
            self.assertEqual(output_path, str(destination))
            self.assertEqual(cancel_result, [False])
            _, output = wavfile.read(str(destination))
            np.testing.assert_array_equal(output, recorded.original)

    def test_cancellation_between_render_blocks_stops_further_rendering(self) -> None:
        recorded = RecordedBuffers(
            original=np.zeros(5000, dtype=np.float32),
            a_per_voice={},
            sample_rate=44100,
        )
        cancel_event = ExportCancellation()
        progress = []

        def cancel_after_first_block(phase, completed, total):
            progress.append((phase, completed, total))
            if phase == "rendering":
                self.assertTrue(cancel_event.cancel())

        with tempfile.TemporaryDirectory() as tempdir:
            destination = Path(tempdir) / "cancelled.wav"
            with self.assertRaises(CancelledError):
                ExportEngine().export(
                    recorded,
                    destination,
                    enabled_voices=(),
                    progress_callback=cancel_after_first_block,
                    cancel_event=cancel_event,
                )

        self.assertEqual(progress, [("rendering", 2048, 5000)])
        self.assertFalse(destination.exists())

    def test_export_writes_valid_wav_and_combines_selected_voices(self) -> None:
        original = np.array([-0.5, -0.125, 0.25, 0.625], dtype=np.float32)
        voice = np.array([0.25, -0.25, 0.125, 0.125], dtype=np.float32)
        recorded = RecordedBuffers(
            original=original,
            a_per_voice={"3rd": voice, "5th": np.zeros_like(original), "octave": np.zeros_like(original)},
            sample_rate=44100,
        )
        original_before = original.copy()
        voice_before = voice.copy()

        with tempfile.TemporaryDirectory() as tempdir:
            destination = Path(tempdir) / "mix.wav"
            output = ExportEngine().export(recorded, destination, enabled_voices=("3rd",))
            self.assertEqual(output, str(destination))
            sample_rate, data = wavfile.read(str(destination))
            self.assertEqual(sample_rate, 44100)
            np.testing.assert_array_equal(data, original + voice)

        np.testing.assert_array_equal(original, original_before)
        np.testing.assert_array_equal(voice, voice_before)

    def test_export_exactly_sums_multiple_explicitly_selected_voices(self) -> None:
        original = np.array([-0.5, 0.0, 0.25, 0.5], dtype=np.float32)
        third = np.array([0.125, 0.25, -0.125, 0.0], dtype=np.float32)
        fifth = np.array([0.25, -0.125, 0.5, -0.25], dtype=np.float32)
        octave = np.full_like(original, 0.75)
        original_before = original.copy()
        third_before = third.copy()
        fifth_before = fifth.copy()
        recorded = RecordedBuffers(
            original=original,
            a_per_voice={"3rd": third, "5th": fifth, "octave": octave},
            sample_rate=44100,
        )

        with tempfile.TemporaryDirectory() as tempdir:
            destination = Path(tempdir) / "selected.wav"
            ExportEngine().export(recorded, destination, enabled_voices=("5th", "3rd"))
            _, output = wavfile.read(str(destination))

        np.testing.assert_array_equal(output, original + fifth + third)
        np.testing.assert_array_equal(original, original_before)
        np.testing.assert_array_equal(third, third_before)
        np.testing.assert_array_equal(fifth, fifth_before)

    def test_over_range_sample_sums_are_preserved_without_clipping(self) -> None:
        original = np.array([0.75, -0.75], dtype=np.float32)
        third = np.array([0.5, -0.5], dtype=np.float32)
        fifth = np.array([0.25, -0.25], dtype=np.float32)
        original_before = original.copy()
        third_before = third.copy()
        fifth_before = fifth.copy()
        recorded = RecordedBuffers(
            original=original,
            a_per_voice={"3rd": third, "5th": fifth},
            sample_rate=44100,
        )

        with tempfile.TemporaryDirectory() as tempdir:
            destination = Path(tempdir) / "over-range.wav"
            ExportEngine().export(recorded, destination, enabled_voices=("3rd", "5th"))
            _, output = wavfile.read(str(destination))

        expected = np.array([1.5, -1.5], dtype=np.float32)
        np.testing.assert_array_equal(output, expected)
        np.testing.assert_array_equal(original, original_before)
        np.testing.assert_array_equal(third, third_before)
        np.testing.assert_array_equal(fifth, fifth_before)

    def test_explicit_empty_voice_selection_exports_original_only(self) -> None:
        original = np.linspace(-0.8, 0.8, 4096, dtype=np.float32)
        voice = np.full_like(original, 0.25)
        recorded = RecordedBuffers(
            original=original,
            a_per_voice={"3rd": voice},
            sample_rate=44100,
        )

        with tempfile.TemporaryDirectory() as tempdir:
            destination = Path(tempdir) / "original.wav"
            ExportEngine().export(recorded, destination, enabled_voices=())
            sample_rate, output = wavfile.read(str(destination))

        self.assertEqual(sample_rate, 44100)
        np.testing.assert_array_equal(output, original)

    def test_omitted_voice_selection_defaults_to_available_harmony_buffers(self) -> None:
        original = np.full(128, 0.1, dtype=np.float32)
        third = np.full_like(original, 0.2)
        fifth = np.full_like(original, 0.3)
        recorded = RecordedBuffers(
            original=original,
            a_per_voice={"3rd": third, "5th": fifth, "octave": None},
            sample_rate=44100,
        )

        with tempfile.TemporaryDirectory() as tempdir:
            destination = Path(tempdir) / "default.wav"
            ExportEngine().export(recorded, destination)
            _, output = wavfile.read(str(destination))

        np.testing.assert_allclose(output, original + third + fifth)

    def test_progress_reports_rendered_sample_counts_and_non_percentage_write_phase(self) -> None:
        original = np.zeros(5000, dtype=np.float32)
        recorded = RecordedBuffers(original=original, a_per_voice={}, sample_rate=44100)
        progress = []

        with tempfile.TemporaryDirectory() as tempdir:
            destination = Path(tempdir) / "progress.wav"

            def report(phase, completed, total):
                if completed is None:
                    self.assertIsNone(total)
                else:
                    self.assertIsInstance(total, int)
                    self.assertTrue(0 < completed <= total)
                if phase == "completed":
                    self.assertTrue(destination.exists())
                    _, committed_audio = wavfile.read(str(destination))
                    np.testing.assert_array_equal(committed_audio, original)
                progress.append((phase, completed, total))

            ExportEngine().export(recorded, destination, enabled_voices=(), progress_callback=report)

        self.assertEqual(
            progress,
            [
                ("rendering", 2048, 5000),
                ("rendering", 4096, 5000),
                ("rendering", 5000, 5000),
                ("writing", None, None),
                ("completed", 5000, 5000),
            ],
        )
        rendering_counts = [event[1] for event in progress if event[0] == "rendering"]
        self.assertEqual(rendering_counts, sorted(rendering_counts))
        self.assertTrue(all(0 < count <= 5000 for count in rendering_counts))
        self.assertEqual(rendering_counts[-1], 5000)

    def test_callback_failures_are_logged_once_and_do_not_fail_export(self) -> None:
        phases = ("rendering", "writing", "completed")
        for failed_phase in phases:
            with self.subTest(phase=failed_phase), tempfile.TemporaryDirectory() as tempdir:
                destination = Path(tempdir) / f"{failed_phase}.wav"
                calls = []

                def report(phase, completed, total):
                    calls.append(phase)
                    if phase == "writing":
                        self.assertIsNone(completed)
                        self.assertIsNone(total)
                    else:
                        self.assertIsInstance(completed, int)
                        self.assertEqual(total, 4096)
                    if phase == failed_phase:
                        raise RuntimeError(f"{phase} observer failed")

                with self.assertLogs("harmonix.export.engine", level="WARNING") as logs:
                    output = ExportEngine().export(
                        RecordedBuffers(
                            original=np.zeros(4096, dtype=np.float32),
                            a_per_voice={},
                            sample_rate=44100,
                        ),
                        destination,
                        enabled_voices=(),
                        progress_callback=report,
                    )

                self.assertEqual(output, str(destination))
                self.assertTrue(destination.exists())
                self.assertEqual(calls.count(failed_phase), 1)
                self.assertEqual(len(calls), calls.index(failed_phase) + 1)
                self.assertIn(f"failed during {failed_phase}", logs.output[0])
                self.assertTrue(any("Traceback" in line for line in logs.output))

    def test_actual_write_and_replace_failures_remain_export_errors_after_callback_failure(self) -> None:
        for failure_site in ("write", "replace"):
            with self.subTest(failure_site=failure_site), tempfile.TemporaryDirectory() as tempdir:
                destination = Path(tempdir) / "existing.wav"
                existing_contents = b"previous destination"
                destination.write_bytes(existing_contents)
                recorded = RecordedBuffers(
                    original=np.zeros(128, dtype=np.float32),
                    a_per_voice={},
                    sample_rate=44100,
                )
                failed_callback_phases = []

                def fail_progress(phase, completed, total):
                    failed_callback_phases.append(phase)
                    self.assertEqual(phase, "rendering")
                    self.assertGreater(completed, 0)
                    self.assertEqual(total, 128)
                    raise RuntimeError("progress observer failed")

                with self.assertLogs("harmonix.export.engine", level="WARNING"):
                    if failure_site == "write":
                        write_patch = patch(
                            "harmonix.export.engine.wavfile.write",
                            side_effect=OSError("injected writer failure"),
                        )
                        replace_patch = patch("harmonix.export.engine.os.replace")
                    else:
                        write_patch = patch("harmonix.export.engine.wavfile.write", wraps=wavfile.write)
                        replace_patch = patch(
                            "harmonix.export.engine.os.replace",
                            side_effect=PermissionError("injected replacement failure"),
                        )

                    with write_patch, replace_patch, self.assertRaises(ExportWriteError):
                        ExportEngine().export(
                            recorded,
                            destination,
                            enabled_voices=(),
                            progress_callback=fail_progress,
                        )

                self.assertEqual(destination.read_bytes(), existing_contents)
                self.assertEqual(list(Path(tempdir).glob("harmonix-export-*.tmp")), [])
                self.assertEqual(failed_callback_phases, ["rendering"])

    def test_permission_denied_is_specific_and_preserves_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            destination = Path(tempdir) / "existing.wav"
            original = b"previous target bytes"
            destination.write_bytes(original)
            recorded = RecordedBuffers(original=np.zeros(128, dtype=np.float32), a_per_voice={}, sample_rate=44100)
            with patch("harmonix.export.engine.os.replace", side_effect=PermissionError("denied")):
                with self.assertRaisesRegex(ExportWriteError, "permission denied"):
                    ExportEngine().export(recorded, destination, enabled_voices=())
            self.assertEqual(destination.read_bytes(), original)
            self.assertEqual(list(Path(tempdir).glob("harmonix-export-*.tmp")), [])

    def test_export_rejects_non_wav_destination(self) -> None:
        recorded = RecordedBuffers(original=np.zeros(128, dtype=np.float32), a_per_voice={}, sample_rate=44100)
        with tempfile.TemporaryDirectory() as tempdir:
            destination = Path(tempdir) / "mix.mp3"
            with self.assertRaises(ExportValidationError):
                ExportEngine().export(recorded, destination)

    def test_export_rejects_invalid_or_non_finite_audio(self) -> None:
        recorded = RecordedBuffers(original=np.array([0.0, np.nan], dtype=np.float32), a_per_voice={}, sample_rate=44100)
        with tempfile.TemporaryDirectory() as tempdir:
            with self.assertRaises(ExportValidationError):
                ExportEngine().export(recorded, Path(tempdir) / "bad.wav")

    def test_export_rejects_invalid_sample_rate(self) -> None:
        recorded = RecordedBuffers(original=np.zeros(128, dtype=np.float32), a_per_voice={}, sample_rate=0)
        with tempfile.TemporaryDirectory() as tempdir:
            with self.assertRaises(ExportValidationError):
                ExportEngine().export(recorded, Path(tempdir) / "bad.wav")

    def test_export_rejects_inconsistent_selected_voice_buffer(self) -> None:
        recorded = RecordedBuffers(
            original=np.zeros(128, dtype=np.float32),
            a_per_voice={"3rd": np.zeros(64, dtype=np.float32)},
            sample_rate=44100,
        )
        with tempfile.TemporaryDirectory() as tempdir:
            with self.assertRaises(ExportValidationError):
                ExportEngine().export(recorded, Path(tempdir) / "bad.wav", enabled_voices=("3rd",))

    def test_export_rejects_invalid_voice_selections_and_unavailable_buffers(self) -> None:
        recorded = RecordedBuffers(
            original=np.zeros(128, dtype=np.float32),
            a_per_voice={"3rd": np.zeros(128, dtype=np.float32), "5th": None},
            sample_rate=44100,
        )
        invalid_selections = (("alto",), ("5th",), 3)
        with tempfile.TemporaryDirectory() as tempdir:
            for selection in invalid_selections:
                with self.subTest(selection=selection):
                    with self.assertRaises(ExportValidationError):
                        ExportEngine().export(
                            recorded,
                            Path(tempdir) / "bad.wav",
                            enabled_voices=selection,
                        )

    def test_export_rejects_malformed_and_non_finite_harmony_buffers(self) -> None:
        original = np.zeros(128, dtype=np.float32)
        invalid_buffers = [
            ["not", "an", "ndarray"],
            np.zeros((2, 64), dtype=np.float32),
            np.array([object()] * 128, dtype=object),
            np.full(128, np.nan, dtype=np.float32),
        ]
        invalid_buffers.extend(
            np.full(128, value, dtype=np.float32)
            for value in (np.inf, -np.inf)
        )
        with tempfile.TemporaryDirectory() as tempdir:
            for invalid_buffer in invalid_buffers:
                with self.subTest(buffer_type=type(invalid_buffer).__name__):
                    recorded = RecordedBuffers(
                        original=original,
                        a_per_voice={"3rd": invalid_buffer},
                        sample_rate=44100,
                    )
                    with self.assertRaises(ExportValidationError):
                        ExportEngine().export(
                            recorded,
                            Path(tempdir) / "bad.wav",
                            enabled_voices=("3rd",),
                        )

    def test_validation_failures_leave_existing_destinations_unchanged(self) -> None:
        original = np.zeros(128, dtype=np.float32)
        valid_recorded = RecordedBuffers(
            original=original,
            a_per_voice={"3rd": np.zeros(128, dtype=np.float32), "5th": None},
            sample_rate=44100,
        )
        cases = (
            (
                "invalid sample rate",
                RecordedBuffers(original=original, a_per_voice={}, sample_rate=0),
                ("3rd",),
                "existing.wav",
            ),
            ("unsupported voice", valid_recorded, ("alto",), "existing.wav"),
            ("unavailable voice", valid_recorded, ("5th",), "existing.wav"),
            ("malformed selection", valid_recorded, 3, "existing.wav"),
            ("invalid destination format", valid_recorded, (), "existing.mp3"),
        )
        with tempfile.TemporaryDirectory() as tempdir:
            for case_name, recorded, voices, filename in cases:
                with self.subTest(case=case_name):
                    destination = Path(tempdir) / filename
                    existing_contents = f"existing destination: {case_name}".encode()
                    destination.write_bytes(existing_contents)

                    with self.assertRaises(ExportValidationError):
                        ExportEngine().export(recorded, destination, enabled_voices=voices)

                    self.assertEqual(destination.read_bytes(), existing_contents)

    def test_voice_iterator_failure_raises_validation_error_and_preserves_destination(self) -> None:
        original = np.zeros(128, dtype=np.float32)
        recorded = RecordedBuffers(
            original=original,
            a_per_voice={"3rd": np.zeros_like(original)},
            sample_rate=44100,
        )

        def broken_voice_selection():
            yield "3rd"
            raise RuntimeError("selection iterator failed")

        with tempfile.TemporaryDirectory() as tempdir:
            destination = Path(tempdir) / "existing.wav"
            existing_contents = b"previous destination"
            destination.write_bytes(existing_contents)

            with self.assertRaisesRegex(ExportValidationError, "selection could not be read"):
                ExportEngine().export(
                    recorded,
                    destination,
                    enabled_voices=broken_voice_selection(),
                )

            self.assertEqual(destination.read_bytes(), existing_contents)

    def test_export_write_errors_raise_domain_error(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            destination = Path(tempdir) / "missing" / "mix.wav"
            with self.assertRaises(ExportWriteError):
                ExportEngine()._write_wav_safely(destination, 44100, np.zeros(256, dtype=np.float32))

    def test_wav_writer_failure_cleans_temp_and_preserves_existing_destination(self) -> None:
        original = np.full(256, 0.125, dtype=np.float32)
        voice = np.full_like(original, 0.25)
        original_before = original.copy()
        voice_before = voice.copy()
        recorded = RecordedBuffers(
            original=original,
            a_per_voice={"3rd": voice},
            sample_rate=44100,
        )
        with tempfile.TemporaryDirectory() as tempdir:
            destination = Path(tempdir) / "existing.wav"
            existing_contents = b"previous valid export"
            destination.write_bytes(existing_contents)

            def fail_writer(temp_path, _sample_rate, _audio):
                self.assertTrue(Path(temp_path).exists())
                self.assertEqual(_sample_rate, 44100)
                self.assertEqual(_audio.shape, original.shape)
                raise OSError("injected WAV writer failure")

            with patch("harmonix.export.engine.wavfile.write", side_effect=fail_writer):
                with self.assertRaises(ExportWriteError):
                    ExportEngine().export(recorded, destination, enabled_voices=("3rd",))

            self.assertEqual(destination.read_bytes(), existing_contents)
            self.assertEqual(list(Path(tempdir).glob("harmonix-export-*.tmp")), [])
            np.testing.assert_array_equal(recorded.original, original_before)
            np.testing.assert_array_equal(recorded.a_per_voice["3rd"], voice_before)

    def test_failed_atomic_replace_preserves_existing_destination_and_cleans_temp_file(self) -> None:
        original = np.full(256, 0.1, dtype=np.float32)
        voice = np.full_like(original, 0.2)
        original_before = original.copy()
        voice_before = voice.copy()
        recorded = RecordedBuffers(
            original=original,
            a_per_voice={"3rd": voice},
            sample_rate=44100,
        )
        with tempfile.TemporaryDirectory() as tempdir:
            destination = Path(tempdir) / "mix.wav"
            existing_contents = b"previous valid export"
            destination.write_bytes(existing_contents)
            with patch("harmonix.export.engine.os.replace", side_effect=PermissionError("injected replace failure")):
                with self.assertRaises(ExportWriteError):
                    ExportEngine().export(recorded, destination, enabled_voices=("3rd",))

            self.assertEqual(destination.read_bytes(), existing_contents)
            self.assertEqual(list(Path(tempdir).glob("harmonix-export-*.tmp")), [])
            np.testing.assert_array_equal(recorded.original, original_before)
            np.testing.assert_array_equal(recorded.a_per_voice["3rd"], voice_before)


if __name__ == "__main__":
    unittest.main()
