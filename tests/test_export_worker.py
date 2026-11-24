import tempfile
import unittest
from concurrent.futures import CancelledError
from pathlib import Path
from threading import Event, current_thread
from unittest.mock import patch

import numpy as np

from harmonix.domain.errors import ExportValidationError, ExportWriteError
from harmonix.domain.enums import SessionMode
from harmonix.domain.models import AppState, RecordedBuffers, SessionState
from harmonix.export.engine import ExportEngine
from harmonix.export.worker import ExportWorker
from harmonix.session_state.app_state import StateStore
from harmonix.session_state.commands import (
    CancelExport,
    CompleteExport,
    FailExport,
    UpdateExportProgress,
)
from scipy.io import wavfile


def make_store(sample_count: int = 5000) -> tuple[StateStore, np.ndarray, np.ndarray]:
    original = np.full(sample_count, 0.1, dtype=np.float32)
    third = np.full_like(original, 0.2)
    session = SessionState(
        mode=SessionMode.UPLOAD,
        recorded_buffers=RecordedBuffers(
            original=original,
            a_per_voice={"3rd": third, "5th": None, "octave": None},
            sample_rate=44100,
        ),
    )
    session.voices.active["3rd"] = True
    return StateStore(AppState(session=session)), original, third


class ExportWorkerTests(unittest.TestCase):
    def test_worker_exports_in_background_and_publishes_real_progress(self) -> None:
        entered = Event()
        release = Event()
        ui_thread = current_thread()
        export_threads = []

        class BlockingEngine(ExportEngine):
            def export(self, *args, **kwargs):
                export_threads.append(current_thread())
                entered.set()
                if not release.wait(2.0):
                    raise TimeoutError("test did not release export worker")
                return super().export(*args, **kwargs)

        store, original, third = make_store()
        published = []
        publish_event = store.publish_export_event

        def record_event(command):
            published.append(command)
            publish_event(command)

        with tempfile.TemporaryDirectory() as tempdir:
            destination = Path(tempdir) / "worker.wav"
            worker = ExportWorker(store, BlockingEngine())
            with patch.object(store, "publish_export_event", side_effect=record_event):
                request_id = worker.start(destination, enabled_voices=("3rd",))
                self.assertTrue(entered.wait(2.0))
                self.assertTrue(worker.is_alive)
                self.assertEqual(store.ui_projection().session.export_status, "running")
                release.set()
                self.assertTrue(worker.join(2.0))

            self.assertFalse(worker.is_alive)
            state = store.snapshot()
            export = state.session.export
            self.assertEqual(export.request_id, request_id)
            self.assertEqual(export.status, "completed")
            self.assertFalse(export.in_progress)
            self.assertEqual(export.progress_pct, 100.0)
            self.assertEqual(export.last_export_path, str(destination))
            self.assertTrue(destination.exists())
            _, output = wavfile.read(str(destination))
            np.testing.assert_array_equal(output, original + third)
            progress = [event for event in published if isinstance(event, UpdateExportProgress)]
            render_counts = [event.completed_units for event in progress if event.phase == "rendering"]
            self.assertEqual(render_counts, sorted(render_counts))
            self.assertEqual(render_counts[-1], original.size)
            self.assertTrue(any(event.phase == "writing" for event in progress))
            self.assertIsNot(export_threads[0], ui_thread)
            np.testing.assert_array_equal(state.session.recorded_buffers.original, original)
            np.testing.assert_array_equal(state.session.recorded_buffers.a_per_voice["3rd"], third)

    def test_cancelled_export_retry_uses_a_fresh_request_and_cancellation_signal(self) -> None:
        started = Event()
        release = Event()

        class CancelOnceEngine(ExportEngine):
            def __init__(self) -> None:
                self.calls = 0
                self.cancel_signals = []

            def export(self, *args, cancel_event, **kwargs):
                self.calls += 1
                self.cancel_signals.append(cancel_event)
                if self.calls == 1:
                    started.set()
                    if not release.wait(2.0):
                        raise TimeoutError("test did not release export worker")
                    if cancel_event.is_set():
                        raise CancelledError("cancelled first attempt")
                return super().export(*args, cancel_event=cancel_event, **kwargs)

        store, _, _ = make_store(128)
        engine = CancelOnceEngine()
        worker = ExportWorker(store, engine)
        with tempfile.TemporaryDirectory() as tempdir:
            destination = Path(tempdir) / "retry-after-cancel.wav"
            first_request = worker.start(destination, enabled_voices=("3rd",))
            self.assertTrue(started.wait(2.0))
            worker.request_cancel()
            release.set()
            self.assertTrue(worker.join(2.0))
            self.assertEqual(store.snapshot().session.export.status, "cancelled")

            second_request = worker.start(destination, enabled_voices=("3rd",))
            self.assertTrue(worker.join(2.0))

        self.assertNotEqual(first_request, second_request)
        self.assertIsNot(engine.cancel_signals[0], engine.cancel_signals[1])
        self.assertFalse(engine.cancel_signals[1].is_set())
        self.assertEqual(store.snapshot().session.export.status, "completed")

    def test_worker_publishes_exactly_one_terminal_event_per_outcome(self) -> None:
        test_case = self

        for outcome, expected_type in (
            ("completed", CompleteExport),
            ("cancelled", CancelExport),
            ("failed", FailExport),
        ):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as tempdir:
                if outcome == "completed":
                    engine = ExportEngine()
                elif outcome == "cancelled":
                    class CancelEngine:
                        def export(
                            self, buffers, destination, *, enabled_voices, progress_callback, cancel_event
                        ):
                            test_case.assertIsNotNone(buffers.original)
                            test_case.assertIsInstance(destination, Path)
                            test_case.assertEqual(enabled_voices, ("3rd",))
                            test_case.assertIsNotNone(progress_callback)
                            test_case.assertFalse(cancel_event.is_set())
                            raise CancelledError("injected cancellation")

                    engine = CancelEngine()
                else:
                    class FailEngine:
                        def export(
                            self, buffers, destination, *, enabled_voices, progress_callback, cancel_event
                        ):
                            test_case.assertIsNotNone(buffers.original)
                            test_case.assertIsInstance(destination, Path)
                            test_case.assertEqual(enabled_voices, ("3rd",))
                            test_case.assertIsNotNone(progress_callback)
                            test_case.assertFalse(cancel_event.is_set())
                            raise RuntimeError("injected export failure")

                    engine = FailEngine()

                store, _, _ = make_store(128)
                published = []
                publish = store.publish_export_event

                def capture_event(command):
                    published.append(command)
                    publish(command)

                worker = ExportWorker(store, engine)
                with patch.object(store, "publish_export_event", side_effect=capture_event):
                    worker.start(Path(tempdir) / "terminal.wav", enabled_voices=("3rd",))
                    self.assertTrue(worker.join(2.0))

                terminal_events = [
                    event
                    for event in published
                    if isinstance(event, (CancelExport, CompleteExport, FailExport))
                ]
                self.assertEqual(len(terminal_events), 1)
                self.assertIsInstance(terminal_events[0], expected_type)

    def test_terminal_publication_failures_are_retried_for_all_outcomes(self) -> None:
        test_case = self
        cases = (
            ("complete", CompleteExport),
            ("cancel", CancelExport),
            ("fail", FailExport),
        )
        for outcome, expected_type in cases:
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as tempdir:
                if outcome == "complete":
                    engine = ExportEngine()
                elif outcome == "cancel":
                    class CancelEngine:
                        def export(
                            self, buffers, destination, *, enabled_voices, progress_callback, cancel_event
                        ):
                            test_case.assertIsNotNone(buffers.original)
                            test_case.assertIsInstance(destination, Path)
                            test_case.assertEqual(enabled_voices, ("3rd",))
                            test_case.assertIsNotNone(progress_callback)
                            test_case.assertFalse(cancel_event.is_set())
                            raise CancelledError("injected cancellation")

                    engine = CancelEngine()
                else:
                    class FailEngine:
                        def export(
                            self, buffers, destination, *, enabled_voices, progress_callback, cancel_event
                        ):
                            test_case.assertIsNotNone(buffers.original)
                            test_case.assertIsInstance(destination, Path)
                            test_case.assertEqual(enabled_voices, ("3rd",))
                            test_case.assertIsNotNone(progress_callback)
                            test_case.assertFalse(cancel_event.is_set())
                            raise RuntimeError("injected engine failure")

                    engine = FailEngine()

                store, _, _ = make_store(128)
                publish = store.publish_export_event
                terminal_attempts = []

                def fail_once(command):
                    if isinstance(command, expected_type):
                        terminal_attempts.append(command)
                        if len(terminal_attempts) == 1:
                            raise RuntimeError("injected publication failure")
                    publish(command)

                worker = ExportWorker(store, engine)
                with patch.object(store, "publish_export_event", side_effect=fail_once):
                    with self.assertLogs("harmonix.export.worker", level="ERROR"):
                        worker.start(Path(tempdir) / f"{outcome}.wav", enabled_voices=("3rd",))
                        self.assertTrue(worker.join(2.0))

                self.assertEqual(len(terminal_attempts), 2)
                state = store.snapshot().session.export
                self.assertEqual(
                    state.status,
                    {"complete": "completed", "cancel": "cancelled", "fail": "failed"}[outcome],
                )
                self.assertFalse(state.in_progress)

    def test_thread_start_failure_retries_terminal_state_publication(self) -> None:
        store, _, _ = make_store(128)
        publish = store.publish_export_event
        fail_attempts = []

        def fail_once(command):
            if isinstance(command, FailExport):
                fail_attempts.append(command)
                if len(fail_attempts) == 1:
                    raise RuntimeError("injected failure publication error")
            publish(command)

        worker = ExportWorker(store)
        with tempfile.TemporaryDirectory() as tempdir:
            with patch.object(store, "publish_export_event", side_effect=fail_once):
                with patch(
                    "harmonix.export.worker.Thread.start",
                    side_effect=RuntimeError("injected thread-start error"),
                ):
                    with self.assertLogs("harmonix.export.worker", level="ERROR"):
                        with self.assertRaisesRegex(RuntimeError, "thread-start error"):
                            worker.start(Path(tempdir) / "start-failure.wav", enabled_voices=("3rd",))

        self.assertEqual(len(fail_attempts), 2)
        self.assertTrue(worker.join(0.01))
        state = store.snapshot().session.export
        self.assertEqual(state.status, "failed")
        self.assertFalse(state.in_progress)

    def test_full_render_progress_is_not_committed_export_completion(self) -> None:
        store, original, third = make_store(128)
        publish = store.publish_export_event
        render_complete = Event()
        release_render = Event()
        terminal_events = []
        destination_contents = b"previous destination"

        def pause_after_full_render(command):
            if isinstance(command, CompleteExport):
                sample_rate, audio = wavfile.read(str(destination))
                self.assertEqual(sample_rate, 44100)
                np.testing.assert_array_equal(audio, original + third)
                terminal_events.append(command)
            publish(command)
            if (
                isinstance(command, UpdateExportProgress)
                and command.phase == "rendering"
                and command.completed_units == command.total_units
            ):
                render_complete.set()
                if not release_render.wait(2.0):
                    raise TimeoutError("test did not release export after full rendering")

        worker = ExportWorker(store)
        with tempfile.TemporaryDirectory() as tempdir:
            destination = Path(tempdir) / "precommit.wav"
            destination.write_bytes(destination_contents)
            with patch.object(store, "publish_export_event", side_effect=pause_after_full_render):
                worker.start(destination, enabled_voices=("3rd",))
                self.assertTrue(render_complete.wait(2.0))

                projection = store.ui_projection().session
                self.assertEqual(projection.export_status, "running")
                self.assertEqual(projection.export_phase, "rendering")
                self.assertEqual(projection.export_completed_units, original.size)
                self.assertEqual(projection.export_total_units, original.size)
                self.assertEqual(projection.export_progress_pct, 100.0)
                self.assertEqual(destination.read_bytes(), destination_contents)
                self.assertEqual(terminal_events, [])

                release_render.set()
                self.assertTrue(worker.join(2.0))

            completed = store.ui_projection().session
            self.assertEqual(completed.export_status, "completed")
            self.assertIsNone(completed.export_phase)
            self.assertEqual(completed.export_progress_pct, 100.0)
            self.assertEqual(len(terminal_events), 1)

    def test_shutdown_timeout_reports_live_worker_then_terminates(self) -> None:
        entered = Event()
        release = Event()
        test_case = self

        class CooperativeBlockedEngine:
            def export(self, _buffers, _destination, *, cancel_event, **_kwargs):
                test_case.assertIsNotNone(_buffers.original)
                test_case.assertIsInstance(_destination, Path)
                test_case.assertEqual(_kwargs["enabled_voices"], ("3rd",))
                test_case.assertIsNotNone(_kwargs["progress_callback"])
                entered.set()
                if not release.wait(2.0):
                    raise TimeoutError("test did not release export worker")
                if cancel_event.is_set():
                    raise CancelledError("shutdown cancellation")
                raise AssertionError("shutdown did not request cancellation")

        store, _, _ = make_store(128)
        worker = ExportWorker(store, CooperativeBlockedEngine())
        with tempfile.TemporaryDirectory() as tempdir:
            worker.start(Path(tempdir) / "shutdown.wav", enabled_voices=("3rd",))
            self.assertTrue(entered.wait(2.0))
            self.assertFalse(worker.shutdown(timeout=0.01))
            self.assertTrue(worker.is_alive)
            release.set()
            self.assertTrue(worker.join(2.0))

        self.assertFalse(worker.is_alive)
        self.assertEqual(store.snapshot().session.export.status, "cancelled")

    def test_state_export_source_keeps_audio_out_of_ui_projection(self) -> None:
        store, original, third = make_store(32)

        source = store.export_source()
        projection = store.ui_projection()

        self.assertIs(source.recorded_buffers.original, original)
        self.assertIs(source.recorded_buffers.a_per_voice["3rd"], third)
        self.assertEqual(source.active_voices, ("3rd",))
        self.assertFalse(hasattr(projection.session, "recorded_buffers"))
        self.assertTrue(projection.session.has_recorded_audio)

    def test_worker_cancellation_preserves_session_and_reports_cancelled(self) -> None:
        started = Event()
        self_outer = self

        class WaitingEngine:
            def export(self, _buffers, _destination, *, cancel_event, **_kwargs):
                self_outer.assertIsNotNone(_buffers.original)
                self_outer.assertIsInstance(_destination, Path)
                self_outer.assertEqual(_kwargs["enabled_voices"], ("3rd",))
                started.set()
                if not release.wait(2.0):
                    raise TimeoutError("test did not release export worker")
                if cancel_event.is_set():
                    raise CancelledError("cancelled by test")
                raise AssertionError("worker continued without cancellation")

        store, original, third = make_store(128)
        worker = ExportWorker(store, WaitingEngine())
        release = Event()
        with tempfile.TemporaryDirectory() as tempdir:
            destination = Path(tempdir) / "cancelled.wav"
            worker.start(destination, enabled_voices=("3rd",))
            self.assertTrue(started.wait(2.0))
            worker.request_cancel()
            release.set()
            self.assertTrue(worker.join(2.0))

        session = store.snapshot().session
        self.assertFalse(worker.is_alive)
        self.assertEqual(session.export.status, "cancelled")
        self.assertTrue(session.export.cancellation_requested)
        self.assertFalse(session.export.in_progress)
        self.assertIsNone(session.export.last_export_path)
        np.testing.assert_array_equal(session.recorded_buffers.original, original)
        np.testing.assert_array_equal(session.recorded_buffers.a_per_voice["3rd"], third)

    def test_worker_failure_preserves_session_and_allows_retry(self) -> None:
        class FailOnceEngine(ExportEngine):
            def __init__(self):
                self.calls = 0

            def export(self, *args, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise ExportWriteError("injected first-attempt failure")
                return super().export(*args, **kwargs)

        store, original, third = make_store(128)
        engine = FailOnceEngine()
        worker = ExportWorker(store, engine)
        with tempfile.TemporaryDirectory() as tempdir:
            destination = Path(tempdir) / "retry.wav"
            first_request = worker.start(destination, enabled_voices=("3rd",))
            worker.join(2.0)

            failed = store.snapshot().session
            self.assertEqual(failed.export.request_id, first_request)
            self.assertEqual(failed.export.status, "failed")
            self.assertIn("ExportWriteError", failed.export.error)
            np.testing.assert_array_equal(failed.recorded_buffers.original, original)
            np.testing.assert_array_equal(failed.recorded_buffers.a_per_voice["3rd"], third)

            second_request = worker.start(destination, enabled_voices=("3rd",))
            worker.join(2.0)

        completed = store.snapshot().session
        self.assertNotEqual(first_request, second_request)
        self.assertEqual(completed.export.request_id, second_request)
        self.assertEqual(completed.export.status, "completed")
        self.assertEqual(engine.calls, 2)

    def test_worker_rejects_concurrent_start(self) -> None:
        entered = Event()
        release = Event()

        class BlockingEngine(ExportEngine):
            def export(self, *args, **kwargs):
                entered.set()
                release.wait(2.0)
                return super().export(*args, **kwargs)

        store, _, _ = make_store(128)
        worker = ExportWorker(store, BlockingEngine())
        with tempfile.TemporaryDirectory() as tempdir:
            destination = Path(tempdir) / "first.wav"
            worker.start(destination, enabled_voices=("3rd",))
            self.assertTrue(entered.wait(2.0))
            with self.assertRaisesRegex(RuntimeError, "already running"):
                worker.start(Path(tempdir) / "second.wav", enabled_voices=("3rd",))
            release.set()
            worker.join(2.0)

        self.assertFalse(worker.is_alive)

    def test_worker_without_recorded_session_fails_before_starting(self) -> None:
        store = StateStore(AppState())
        worker = ExportWorker(store)

        with tempfile.TemporaryDirectory() as tempdir:
            with self.assertRaisesRegex(ExportValidationError, "recorded audio"):
                worker.start(Path(tempdir) / "missing.wav", enabled_voices=())

        self.assertIsNone(store.snapshot().session)

    def test_thread_start_failure_is_terminal_and_join_reports_stopped(self) -> None:
        store, _, _ = make_store(128)
        worker = ExportWorker(store)

        with tempfile.TemporaryDirectory() as tempdir:
            with patch("harmonix.export.worker.Thread.start", side_effect=RuntimeError("thread start failed")):
                with self.assertRaisesRegex(RuntimeError, "thread start failed"):
                    worker.start(Path(tempdir) / "start-failure.wav", enabled_voices=("3rd",))

        self.assertTrue(worker.join(0.01))
        self.assertFalse(worker.is_alive)
        export = store.snapshot().session.export
        self.assertEqual(export.status, "failed")
        self.assertFalse(export.in_progress)


if __name__ == "__main__":
    unittest.main()
