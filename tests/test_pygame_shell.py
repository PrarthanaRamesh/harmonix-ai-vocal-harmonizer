"""Milestone 2 shell and state-binding tests."""

import os
import sys
import tempfile
from pathlib import Path
from threading import Event
from types import ModuleType
import unittest
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame

from harmonix.config.validation import format_audio_device_selector
from harmonix.domain.enums import DefaultKeyBehavior, Screen, SessionMode, StylePreset
from harmonix.domain.models import AppState, GlobalSettings, KeyState, PlaybackState, RecordedBuffers, RecentSessionSummary, SessionState
from harmonix.key_scale_resolver import KeyScale
from harmonix.domain.errors import NoInputDeviceError, PermissionDeniedError, PersistenceLockError, PersistenceReadError, PersistenceWriteError
from harmonix.session_state.app_state import PlaybackSource, StateStore
from harmonix.session_state.commands import CompleteUploadAnalysis, EndSession, Navigate, StartUploadAnalysis
from harmonix.persistence.store import RecentSessionMetadata
from harmonix.export.engine import ExportEngine
from harmonix.export.worker import ExportWorker
from harmonix.input.adapter import DeviceInfo, InputAdapter
from harmonix.ui.pygame_app import PygameApp
from harmonix.ui.screens.shell import ShellLayout, render_shell
from harmonix.ui.state_binding import build_view_model, downsample_waveform, upload_progress_label
from harmonix.visualizer_bridge import LiveTelemetryFrame


class PygameShellTests(unittest.TestCase):
    def tearDown(self) -> None:
        pygame.quit()

    def test_view_model_reads_canonical_state(self) -> None:
        store = StateStore()
        store.dispatch(Navigate(Screen.SETTINGS))

        view_model = build_view_model(store.ui_projection())

        self.assertEqual(view_model.screen_name, "settings")
        self.assertFalse(view_model.has_active_session)

    def test_projection_and_playback_source_do_not_copy_audio_buffers(self) -> None:
        buffers = RecordedBuffers(
            np.ones(64, dtype=np.float32),
            {"3rd": np.zeros(64, dtype=np.float32)},
            44100,
        )
        session = SessionState(mode=SessionMode.UPLOAD, recorded_buffers=buffers)
        session.playback = PlaybackState(available_voices=("3rd",))
        store = StateStore(AppState(session=session))

        with patch("harmonix.session_state.app_state.deepcopy", side_effect=AssertionError("must not copy audio")):
            projection = store.ui_projection()
            source = store.playback_source()

        self.assertTrue(projection.session.has_recorded_audio)
        self.assertIsInstance(source, PlaybackSource)
        self.assertIs(source.recorded_buffers, buffers)
        self.assertEqual(source.rendered_voices, ("3rd",))

    def test_unknown_and_underreported_progress_never_claims_false_percent(self) -> None:
        self.assertEqual(upload_progress_label(12, None), "12 units")
        self.assertEqual(upload_progress_label(12, 5), "12 units")
        self.assertEqual(upload_progress_label(3, 5), "3 / 5 units")

    def test_waveform_reduction_is_deterministic_and_bounded(self) -> None:
        audio = np.array([-1.0, 0.25, 0.5, -0.75, 0.1, 0.8], dtype=np.float32)

        result = downsample_waveform(audio, point_count=3)

        np.testing.assert_allclose(result, ((-1.0, 0.25), (-0.75, 0.5), (0.1, 0.8)))
        self.assertEqual(len(downsample_waveform(audio, point_count=100)), audio.size)

    def test_navigation_uses_authoritative_screen_command(self) -> None:
        store = StateStore()
        app = PygameApp(store, window_size=(480, 320))
        app.initialize()
        app.render()

        app._activate("choose_upload")

        self.assertEqual(store.ui_projection().current_screen, Screen.UPLOAD_SELECT.value)
        app.shutdown()

    def test_all_defined_screens_render_headlessly(self) -> None:
        pygame.init()
        surface = pygame.Surface((640, 420))
        for screen in Screen:
            with self.subTest(screen=screen):
                store = StateStore()
                store.dispatch(Navigate(screen))
                layout = render_shell(
                    surface,
                    build_view_model(store.ui_projection()),
                    focused="",
                    hovered=None,
                    upload_voices=("3rd",),
                    upload_path_entry="",
                    upload_start_error=None,
                )
                self.assertIn("nav_settings", layout.controls)
                if screen is Screen.SETTINGS:
                    self.assertTrue({
                        "setting:key:auto", "setting:key:last_used",
                        "setting:style:pop", "setting:style:choir", "setting:style:barbershop",
                        "setting:frame:down", "setting:frame:up", "setting:reduced_motion",
                    }.issubset(layout.controls))
                    self.assertTrue({
                        "setting:input_device:previous", "setting:input_device:next", "setting:input_device:default",
                        "setting:output_device:previous", "setting:output_device:next", "setting:output_device:default",
                        "refresh_audio_devices",
                    }.issubset(layout.controls))
                    self.assertNotIn("setting:style:barbershop", layout.enabled)

    def test_playback_actions_queue_controller_intents_without_mutating_engine(self) -> None:
        app = PygameApp()
        app._layout = ShellLayout(
            {"play": pygame.Rect(0, 0, 20, 20), "seek": pygame.Rect(0, 0, 100, 10), "voice:3rd": pygame.Rect(0, 0, 20, 20)},
            {"play", "seek", "voice:3rd"},
        )
        session = SessionState(mode=SessionMode.UPLOAD)
        session.playback = PlaybackState(available_voices=("3rd",), duration_seconds=10.0)
        app.state_store = StateStore(AppState(session=session))

        app._activate("play")
        app._activate("voice:3rd")
        app._seek_at(50)

        self.assertEqual(app._playback_requests.get_nowait(), ("play",))
        self.assertEqual(app._playback_requests.get_nowait(), ("voices", ("3rd",)))
        action, seconds = app._playback_requests.get_nowait()
        self.assertEqual(action, "seek")
        self.assertAlmostEqual(seconds, 5.0)
        self.assertFalse(hasattr(app, "engine"))

    def test_disabled_controls_are_not_keyboard_or_mouse_activatable(self) -> None:
        pygame.init()
        surface = pygame.Surface((640, 420))
        store = StateStore()
        store.dispatch(Navigate(Screen.LIVE_CONFIGURE))
        layout = render_shell(
            surface,
            build_view_model(store.ui_projection()),
            focused="start_live",
            hovered=None,
            upload_voices=(),
            upload_path_entry="",
            upload_start_error=None,
        )

        self.assertIn("start_live", layout.controls)
        self.assertNotIn("start_live", layout.enabled)

    def test_shell_initializes_renders_and_shuts_down(self) -> None:
        app = PygameApp(window_size=(320, 240))

        app.initialize()
        app.render()

        self.assertTrue(app.initialized)
        self.assertIsNotNone(pygame.display.get_surface())

        app.shutdown()

        self.assertFalse(app.initialized)
        self.assertFalse(app.running)

    def test_quit_event_stops_test_run_and_releases_pygame(self) -> None:
        app = PygameApp(window_size=(320, 240))
        app.initialize()
        pygame.event.post(pygame.event.Event(pygame.QUIT))

        app.run(max_frames=1)

        self.assertFalse(app.initialized)
        self.assertFalse(app.running)

    def test_lifetime_lock_is_acquired_before_pygame_and_released_after_cleanup(self) -> None:
        events = []

        class LockStore:
            def acquire_lifetime_lock(self):
                events.append("acquire")

            def load_settings(self):
                events.append("load_settings")
                return GlobalSettings(default_key_behavior=DefaultKeyBehavior.LAST_USED)

            def load_recent_sessions(self):
                events.append("load_recent_sessions")
                return []

            settings_warning = None

            def release_lifetime_lock(self):
                events.append("release")

        state = StateStore()
        app = PygameApp(state_store=state, persistence_store=LockStore())
        real_init = pygame.init
        real_quit = pygame.quit

        def init():
            events.append("pygame.init")
            self.assertEqual(state.snapshot().settings.default_key_behavior, DefaultKeyBehavior.LAST_USED)
            real_init()

        def quit():
            events.append("pygame.quit")
            real_quit()

        with patch("harmonix.ui.pygame_app.pygame.init", side_effect=init), patch(
            "harmonix.ui.pygame_app.pygame.quit", side_effect=quit
        ):
            app.initialize()
            app.shutdown()
            app.shutdown()

        self.assertLess(events.index("acquire"), events.index("pygame.init"))
        self.assertLess(events.index("load_settings"), events.index("pygame.init"))
        self.assertLess(events.index("load_recent_sessions"), events.index("pygame.init"))
        self.assertEqual(events.count("load_settings"), 1)
        self.assertEqual(events.count("load_recent_sessions"), 1)
        self.assertLess(events.index("pygame.quit"), events.index("release"))
        self.assertEqual(events.count("release"), 1)

    def test_lock_acquisition_failure_never_initializes_or_releases(self) -> None:
        class LockStore:
            def acquire_lifetime_lock(self):
                raise PersistenceLockError("owned elsewhere")

            def release_lifetime_lock(self):
                raise AssertionError("must not release another owner's lock")

        app = PygameApp(persistence_store=LockStore())
        with patch("harmonix.ui.pygame_app.pygame.init") as init:
            with self.assertRaises(PersistenceLockError):
                app.initialize()
        init.assert_not_called()

    def test_partial_pygame_initialization_cleans_up_before_releasing_lock(self) -> None:
        events = []

        class LockStore:
            def acquire_lifetime_lock(self):
                events.append("acquire")

            def load_settings(self):
                events.append("load_settings")
                return GlobalSettings()

            def load_recent_sessions(self):
                events.append("load_recent_sessions")
                return []

            settings_warning = None

            def release_lifetime_lock(self):
                events.append("release")

        app = PygameApp(persistence_store=LockStore())
        with patch("harmonix.ui.pygame_app.pygame.init", side_effect=lambda: events.append("init")), patch(
            "harmonix.ui.pygame_app.pygame.display.set_mode", side_effect=RuntimeError("window failure")
        ), patch("harmonix.ui.pygame_app.pygame.quit", side_effect=lambda: events.append("quit")):
            with self.assertRaisesRegex(RuntimeError, "window failure"):
                app.initialize()

        self.assertLess(events.index("acquire"), events.index("init"))
        self.assertLess(events.index("load_settings"), events.index("init"))
        self.assertLess(events.index("quit"), events.index("release"))
        self.assertEqual(events.count("release"), 1)

    def test_runtime_thread_start_failure_cleans_pygame_before_releasing_lock(self) -> None:
        events = []

        class LockStore:
            def acquire_lifetime_lock(self):
                events.append("acquire")

            def load_settings(self):
                events.append("load_settings")
                return GlobalSettings()

            def load_recent_sessions(self):
                events.append("load_recent_sessions")
                return []

            settings_warning = None

            def release_lifetime_lock(self):
                events.append("release")

        class StartFailureThread:
            ident = None

            def __init__(self, *args, **kwargs):
                pass

            def start(self):
                events.append("thread.start")
                raise RuntimeError("thread start failed")

            def join(self):
                raise AssertionError("unstarted thread must not be joined")

        app = PygameApp(persistence_store=LockStore())
        with patch("harmonix.ui.pygame_app.Thread", StartFailureThread), patch(
            "harmonix.ui.pygame_app.pygame.init", side_effect=lambda: events.append("pygame.init")
        ), patch("harmonix.ui.pygame_app.pygame.display.set_mode", return_value=object()), patch(
            "harmonix.ui.pygame_app.pygame.display.set_caption"
        ), patch("harmonix.ui.pygame_app.pygame.quit", side_effect=lambda: events.append("pygame.quit")):
            with self.assertRaisesRegex(RuntimeError, "thread start failed"):
                app.initialize()

        self.assertLess(events.index("acquire"), events.index("pygame.init"))
        self.assertLess(events.index("load_settings"), events.index("pygame.init"))
        self.assertLess(events.index("thread.start"), events.index("pygame.quit"))
        self.assertLess(events.index("pygame.quit"), events.index("release"))
        self.assertEqual(events.count("release"), 1)
        self.assertIsNone(app._runtime_thread)
        self.assertFalse(app.initialized)

    def test_settings_load_failure_releases_lock_without_initializing_pygame(self) -> None:
        events = []

        class LoadFailureStore:
            settings_warning = None

            def acquire_lifetime_lock(self):
                events.append("acquire")

            def load_settings(self):
                events.append("load")
                raise PersistenceReadError("load failed")

            def release_lifetime_lock(self):
                events.append("release")

        app = PygameApp(persistence_store=LoadFailureStore())
        with patch("harmonix.ui.pygame_app.pygame.init") as init:
            with self.assertRaises(PersistenceReadError):
                app.initialize()

        init.assert_not_called()
        self.assertEqual(events, ["acquire", "load", "release"])

    def test_settings_edits_persist_before_dispatch_and_retry_keeps_canonical_state(self) -> None:
        events = []

        class FakePersistence:
            settings_warning = None
            fail = True

            def save_settings(self, candidate):
                events.append(("save", candidate.default_key_behavior, candidate.default_style_preset, state.snapshot().settings.default_key_behavior))
                if self.fail:
                    raise PersistenceWriteError("disk full")

        persistence = FakePersistence()
        state = StateStore()
        app = PygameApp(state, persistence_store=persistence)

        app._save_setting("default_key_behavior", DefaultKeyBehavior.LAST_USED)
        self.assertEqual(state.snapshot().settings.default_key_behavior, DefaultKeyBehavior.AUTO)
        self.assertEqual(app._pending_setting, ("default_key_behavior", DefaultKeyBehavior.LAST_USED))
        app._save_setting("default_style_preset", StylePreset.CHOIR)
        self.assertEqual(len(events), 1)
        self.assertEqual(state.snapshot().settings.default_style_preset, StylePreset.POP)
        self.assertEqual(events[0][3], DefaultKeyBehavior.AUTO)

        persistence.fail = False
        app._save_setting(*app._pending_setting, retry=True)
        self.assertEqual(state.snapshot().settings.default_key_behavior, DefaultKeyBehavior.LAST_USED)
        self.assertEqual(app._pending_setting, None)
        self.assertIsNone(app._settings_error)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[1][3], DefaultKeyBehavior.AUTO)

    def test_device_choice_uses_canonical_save_flow_and_explicit_system_default(self) -> None:
        class Persistence:
            settings_warning = None

            def __init__(self):
                self.saved = []

            def save_settings(self, candidate):
                self.saved.append(candidate)

        persistence = Persistence()
        state = StateStore()
        app = PygameApp(state, persistence_store=persistence)
        mic = DeviceInfo(3, "USB Mic", 1, 0, 48000, True, "Windows WASAPI")
        speaker = DeviceInfo(7, "USB Speakers", 0, 2, 48000, True, "Windows WASAPI")
        with patch.object(InputAdapter, "list_devices", return_value=[mic]), patch.object(
            InputAdapter, "list_output_devices", return_value=[speaker]
        ):
            app._refresh_audio_devices()
            app._layout = ShellLayout(
                {"setting:input_device:next": pygame.Rect(0, 0, 20, 20), "setting:output_device:default": pygame.Rect(0, 0, 20, 20)},
                {"setting:input_device:next", "setting:output_device:default"},
            )
            app._activate("setting:input_device:next")
            app._activate("setting:output_device:default")

        self.assertEqual(state.snapshot().settings.audio_input_device, "USB Mic, Windows WASAPI")
        self.assertIsNone(state.snapshot().settings.audio_output_device)
        self.assertEqual(persistence.saved[-1].audio_input_device, "USB Mic, Windows WASAPI")
        self.assertIsNone(persistence.saved[-1].audio_output_device)

    def test_ambiguous_device_choice_does_not_change_or_persist_canonical_setting(self) -> None:
        class Persistence:
            settings_warning = None
            saves = 0

            def save_settings(self, candidate):
                self.saves += 1

        persistence = Persistence()
        state = StateStore()
        app = PygameApp(state, persistence_store=persistence)
        app._audio_input_devices = [
            DeviceInfo(1, "USB Mic", 1, 0, 48000, False, "WASAPI"),
            DeviceInfo(9, "USB Mic", 1, 0, 48000, False, "WASAPI"),
        ]
        app._layout = ShellLayout(
            {"setting:input_device:next": pygame.Rect(0, 0, 20, 20)},
            {"setting:input_device:next"},
        )
        app._activate("setting:input_device:next")

        self.assertIsNone(state.snapshot().settings.audio_input_device)
        self.assertEqual(persistence.saves, 0)
        self.assertIn("ambiguous", app._audio_device_error.lower())

    def test_invalid_setting_never_persists_or_dispatches(self) -> None:
        class FakePersistence:
            settings_warning = None

            def save_settings(self, candidate):
                raise AssertionError("invalid settings must not persist")

        state = StateStore()
        app = PygameApp(state, persistence_store=FakePersistence())
        app._save_setting("latency_buffer_frame_size", True)

        self.assertEqual(state.snapshot().settings.latency_buffer_frame_size, 2048)
        self.assertIsNotNone(app._settings_error)
        self.assertIsNone(app._pending_setting)

    def test_unreadable_save_retry_stays_fail_closed_and_settings_edits_disable(self) -> None:
        class UnreadablePersistence:
            settings_warning = "settings could not be read"
            attempts = 0

            def save_settings(self, candidate):
                self.attempts += 1
                raise PersistenceReadError("refusing to replace unreadable settings")

        persistence = UnreadablePersistence()
        state = StateStore()
        app = PygameApp(state, persistence_store=persistence)
        app._save_setting("reduced_motion", True)
        self.assertEqual(state.snapshot().settings.reduced_motion, False)
        self.assertEqual(app._pending_setting, ("reduced_motion", True))

        app._save_setting(*app._pending_setting, retry=True)
        self.assertEqual(persistence.attempts, 2)
        self.assertEqual(state.snapshot().settings.reduced_motion, False)
        self.assertEqual(app._pending_setting, ("reduced_motion", True))
        self.assertEqual(app._settings_warning, persistence.settings_warning)

        pygame.init()
        surface = pygame.Surface((900, 620))
        state.dispatch(Navigate(Screen.SETTINGS))
        layout = render_shell(
            surface,
            build_view_model(state.ui_projection()),
            focused="",
            hovered=None,
            upload_voices=(),
            upload_path_entry="",
            upload_start_error=None,
            settings_warning=app._settings_warning,
            settings_error=app._settings_error,
            settings_retry_pending=True,
        )
        self.assertNotIn("setting:reduced_motion", layout.enabled)
        self.assertIn("settings_retry", layout.enabled)
        self.assertFalse(any("recover" in name for name in layout.controls))


    def test_last_used_upload_uses_only_the_canonical_manual_key(self):
        class Worker:
            args = None

            def start(self, path, voices, *, manual_key=None, style_preset=None):
                self.args = (path, tuple(voices), manual_key, style_preset)

        settings = GlobalSettings(default_key_behavior=DefaultKeyBehavior.LAST_USED)
        store = StateStore(AppState(settings=settings, last_manual_key=KeyState("D", "minor", False)))
        worker = Worker()
        app = PygameApp(state_store=store, upload_worker=worker)
        app._start_upload("fresh.wav")
        self.assertEqual(worker.args[2], KeyScale("D", "minor"))

    def test_new_upload_voices_follow_approved_style_defaults(self):
        settings = GlobalSettings(default_style_preset=StylePreset.CHOIR)
        app = PygameApp(state_store=StateStore(AppState(settings=settings)))
        app._navigate(Screen.UPLOAD_SELECT)
        self.assertEqual(app._upload_voices, {"3rd", "5th", "octave"})

    def test_barbershop_cannot_be_selected_or_start_a_new_upload(self):
        settings = GlobalSettings(default_style_preset=StylePreset.BARBERSHOP)
        app = PygameApp(state_store=StateStore(AppState(settings=settings)))
        app._navigate(Screen.UPLOAD_SELECT)
        app._start_upload("fresh.wav")
        self.assertIn("Choose Pop or Choir", app._upload_start_error)

    def test_successive_configure_key_changes_keep_controls_open_and_only_latest_buffers_playable(self):
        pygame.init()
        original = np.zeros(16, dtype=np.float32)
        initial_buffers = RecordedBuffers(
            original=original,
            a_per_voice={"3rd": np.zeros_like(original)},
            sample_rate=44100,
        )
        session = SessionState(
            mode=SessionMode.UPLOAD,
            source_file="vocal.wav",
            key=KeyState("C", "major", False),
            detected_key=KeyState("C", "major"),
            recorded_buffers=initial_buffers,
        )
        session.voices.active["3rd"] = True
        session.playback.available_voices = ("3rd",)
        session.upload_analysis.status = "completed"
        session.upload_analysis.request_id = "initial-request"
        state = StateStore(AppState(current_screen=Screen.UPLOAD_CONFIGURE, session=session))

        class QueuedWorker:
            def __init__(self):
                self.requests = []

            def start(self, path, voices, *, manual_key=None, style_preset=None, keep_configure_open=False, continuation_request_id=None):
                request_id = f"request-{len(self.requests) + 1}"
                requested_key = KeyState(manual_key.root, manual_key.scale_type, False)
                self.requests.append((request_id, requested_key, continuation_request_id))
                state.dispatch(
                    StartUploadAnalysis(
                        request_id,
                        path,
                        tuple(voices),
                        style_preset,
                        requested_key,
                        keep_configure_open,
                        continuation_request_id,
                    )
                )
                return request_id

        worker = QueuedWorker()
        app = PygameApp(state_store=state, upload_worker=worker)

        self.assertIsNotNone(state.playback_source())
        app._select_upload_key("key:root:next")
        first_request, first_key, first_continuation = worker.requests[-1]
        self.assertEqual(first_continuation, "initial-request")
        self.assertEqual(first_key, KeyState("C#", "major", False))
        self.assertEqual(state.ui_projection().current_screen, Screen.UPLOAD_CONFIGURE.value)
        self.assertIsNone(state.playback_source())

        layout = render_shell(
            pygame.Surface((900, 620)),
            build_view_model(state.ui_projection()),
            focused="",
            hovered=None,
            upload_voices=("3rd",),
            upload_path_entry="",
            upload_start_error=None,
        )
        self.assertIn("key:root:next", layout.enabled)
        self.assertNotIn("open_playback", layout.enabled)

        app._select_upload_key("key:root:next")
        latest_request, latest_key, latest_continuation = worker.requests[-1]
        self.assertEqual(latest_continuation, first_request)
        self.assertEqual(latest_key, KeyState("D", "major", False))
        self.assertEqual(state.snapshot().last_manual_key, latest_key)
        self.assertIsNone(state.playback_source())

        stale_buffers = RecordedBuffers(original=np.ones(16, dtype=np.float32), a_per_voice={}, sample_rate=44100)
        state.dispatch(
            CompleteUploadAnalysis(
                first_request,
                stale_buffers,
                first_key,
                16 / 44100,
                KeyState("C", "major"),
            )
        )
        self.assertIsNone(state.playback_source())

        latest_buffers = RecordedBuffers(
            original=np.ones(16, dtype=np.float32),
            a_per_voice={"3rd": np.full(16, 0.5, dtype=np.float32)},
            sample_rate=44100,
        )
        state.dispatch(
            CompleteUploadAnalysis(
                latest_request,
                latest_buffers,
                latest_key,
                16 / 44100,
                KeyState("C", "major"),
            )
        )
        committed = state.snapshot().session
        self.assertEqual(committed.key, latest_key)
        self.assertEqual(committed.detected_key, KeyState("C", "major"))
        self.assertIsNone(committed.pending_key)
        self.assertIs(state.playback_source().recorded_buffers, latest_buffers)

    def test_retry_upload_continues_the_current_failed_request(self):
        session = SessionState(mode=SessionMode.UPLOAD, source_file="retry.wav")
        session.upload_analysis.request_id = "failed-request"
        session.upload_analysis.status = "failed"
        state = StateStore(AppState(session=session))

        class Worker:
            continuation = None

            def start(self, path, voices, *, manual_key=None, style_preset=None, continuation_request_id=None):
                self.continuation = continuation_request_id
                state.dispatch(
                    StartUploadAnalysis(
                        "retry-request",
                        path,
                        tuple(voices),
                        style_preset or StylePreset.POP,
                        continuation_request_id=continuation_request_id,
                    )
                )

        worker = Worker()
        PygameApp(state_store=state, upload_worker=worker)._start_upload("retry.wav", retry=True)

        self.assertEqual(worker.continuation, "failed-request")
        self.assertEqual(state.snapshot().session.upload_analysis.request_id, "retry-request")

    def test_auto_upload_ignores_last_manual_key(self):
        class Worker:
            key = object()

            def start(self, _path, _voices, *, manual_key=None, style_preset=None):
                self.key = manual_key

        store = StateStore(AppState(last_manual_key=KeyState("D", "minor", False)))
        worker = Worker()
        PygameApp(state_store=store, upload_worker=worker)._start_upload("fresh.wav")
        self.assertIsNone(worker.key)

    def test_end_session_persists_before_clear_and_failure_can_retry(self):
        events = []
        session = SessionState(
            mode=SessionMode.UPLOAD,
            source_file="song.wav",
            key=KeyState("G", "minor", False),
            style_preset=StylePreset.CHOIR,
        )
        session.voices.active.update({"3rd": False, "5th": True, "octave": True})
        session.recorded_buffers = RecordedBuffers(
            original=np.zeros(64, dtype=np.float32),
            a_per_voice={"3rd": np.ones(64, dtype=np.float32), "5th": np.ones(64, dtype=np.float32), "octave": np.ones(64, dtype=np.float32)},
            sample_rate=44100,
        )
        session.playback.available_voices = ("3rd", "5th", "octave")
        state = StateStore(AppState(
            current_screen=Screen.UPLOAD_PLAYBACK,
            session=session,
            settings=GlobalSettings(default_key_behavior=DefaultKeyBehavior.LAST_USED),
        ))

        class Persistence:
            fail = True

            def load_recent_sessions(self):
                return []

            def save_recent_sessions(self, records):
                events.append(("save", state.snapshot().session is not None, records[0].id))
                if self.fail:
                    raise PersistenceWriteError("disk full")
                return list(records)

        persistence = Persistence()
        app = PygameApp(state_store=state, persistence_store=persistence)
        original_settings = state.snapshot().settings
        pygame.init()
        app._layout = render_shell(
            pygame.Surface((900, 620)),
            build_view_model(state.ui_projection()),
            focused="",
            hovered=None,
            upload_voices=(),
            upload_path_entry="",
            upload_start_error=None,
        )
        self.assertIn("end_session", app._layout.enabled)

        app._activate("end_session")

        self.assertIsNotNone(state.snapshot().session)
        self.assertEqual(state.snapshot().settings, original_settings)
        self.assertEqual(app._session_end_error, "disk full")
        pending_id = app._pending_session_end[0].id
        self.assertTrue(events[-1][1])

        persistence.fail = False
        app._layout = render_shell(
            pygame.Surface((900, 620)),
            build_view_model(state.ui_projection()),
            focused="",
            hovered=None,
            upload_voices=(),
            upload_path_entry="",
            upload_start_error=None,
            session_end_error=app._session_end_error,
            session_end_pending=True,
        )
        self.assertIn("retry_session_end", app._layout.enabled)
        app._activate("retry_session_end")

        self.assertEqual(events[-1][2], pending_id)
        self.assertIsNotNone(state.snapshot().session)
        self.assertEqual(state.snapshot().settings, original_settings)
        self.assertEqual(state.snapshot().recent_sessions[0].last_used_key_root, "G")
        self.assertEqual(dict(state.snapshot().recent_sessions[0].voice_toggles), {"3rd": False, "5th": True, "octave": True})
        self.assertEqual(state.snapshot().session.export.selected_voices, ("5th", "octave"))
        self.assertEqual(state.ui_projection().current_screen, Screen.REVIEW_EXPORT.value)

    def test_review_export_selection_is_independent_and_failures_keep_session_for_retry(self):
        buffers = RecordedBuffers(
            original=np.ones(8, dtype=np.float32),
            a_per_voice={"3rd": np.zeros(8, dtype=np.float32)},
            sample_rate=44100,
        )
        session = SessionState(mode=SessionMode.UPLOAD, source_file="song.wav", recorded_buffers=buffers)
        session.playback.available_voices = ("3rd",)
        session.voices.active["3rd"] = True
        from harmonix.session_state.commands import (
            CancelExport, FailExport, RequestExportCancellation, SetExportVoices, StartExport,
        )
        state = StateStore(AppState(current_screen=Screen.REVIEW_EXPORT, session=session))
        state.dispatch(SetExportVoices(("3rd",)))
        app = PygameApp(state_store=state)
        pygame.init()
        app._layout = render_shell(
            pygame.Surface((900, 620)), build_view_model(state.ui_projection()), focused="", hovered=None,
            upload_voices=(), upload_path_entry="", upload_start_error=None,
        )
        self.assertIn("export_voice:3rd", app._layout.enabled)
        app._activate("export_voice:3rd")
        self.assertEqual(state.snapshot().session.export.selected_voices, ())
        self.assertTrue(state.snapshot().session.voices.active["3rd"])

        state.dispatch(SetExportVoices(("3rd",)))
        state.publish_export_event(StartExport("export-1"))
        state.publish_export_event(FailExport("export-1", "ExportWriteError", "permission denied"))
        after_failure = state.snapshot().session
        np.testing.assert_array_equal(after_failure.recorded_buffers.original, buffers.original)
        np.testing.assert_array_equal(after_failure.recorded_buffers.a_per_voice["3rd"], buffers.a_per_voice["3rd"])
        self.assertEqual(after_failure.export.selected_voices, ("3rd",))
        self.assertEqual(after_failure.export.status, "failed")
        state.publish_export_event(StartExport("export-2"))
        state.publish_export_event(RequestExportCancellation("export-2"))
        state.publish_export_event(CancelExport("export-2"))
        after_cancel = state.snapshot().session
        np.testing.assert_array_equal(after_cancel.recorded_buffers.original, buffers.original)
        self.assertEqual(after_cancel.export.selected_voices, ("3rd",))
        self.assertEqual(after_cancel.export.status, "cancelled")
        app._layout = render_shell(
            pygame.Surface((900, 620)), build_view_model(state.ui_projection()), focused="", hovered=None,
            upload_voices=(), upload_path_entry="", upload_start_error=None,
        )
        self.assertIn("export_start", app._layout.enabled)
        self.assertIn("discard_session", app._layout.enabled)
        app._layout = render_shell(
            pygame.Surface((900, 620)), build_view_model(state.ui_projection()), focused="", hovered=None,
            upload_voices=(), upload_path_entry="", upload_start_error=None,
            discard_confirmation_pending=True,
        )
        self.assertEqual(app._layout.enabled, {"discard_confirm", "discard_cancel"})

    def test_shutdown_defers_pygame_cleanup_and_lock_release_until_export_worker_stops(self):
        class Persistence:
            released = False

            def release_lifetime_lock(self):
                self.released = True

        class Export:
            is_alive = True

            def shutdown(self, timeout=None):
                self.timeout = timeout
                return False

        persistence = Persistence()
        app = PygameApp(persistence_store=persistence, export_worker=Export())
        app._lifetime_lock_acquired = True
        app._pygame_init_started = True
        app._initialized = True
        with patch("harmonix.ui.pygame_app.pygame.quit") as pygame_quit:
            with self.assertRaisesRegex(RuntimeError, "worker has not terminated"):
                app.shutdown()
        pygame_quit.assert_not_called()
        self.assertFalse(persistence.released)
        self.assertTrue(app._lifetime_lock_acquired)
        self.assertTrue(app.initialized)

    def test_discard_cancel_retains_session_and_confirmed_discard_clears_it(self):
        buffers = RecordedBuffers(
            original=np.ones(16, dtype=np.float32),
            a_per_voice={"3rd": np.zeros(16, dtype=np.float32)},
            sample_rate=44100,
        )
        session = SessionState(mode=SessionMode.UPLOAD, source_file="song.wav", recorded_buffers=buffers)
        session.playback.available_voices = ("3rd",)
        state = StateStore(AppState(current_screen=Screen.REVIEW_EXPORT, session=session))
        app = PygameApp(state_store=state)

        def render_discard_dialog():
            app._layout = render_shell(
                pygame.Surface((900, 620)), build_view_model(state.ui_projection()),
                focused="", hovered=None, upload_voices=(), upload_path_entry="",
                upload_start_error=None, discard_confirmation_pending=True,
            )

        pygame.init()
        app._layout = render_shell(
            pygame.Surface((900, 620)), build_view_model(state.ui_projection()),
            focused="", hovered=None, upload_voices=(), upload_path_entry="", upload_start_error=None,
        )
        app._activate("discard_session")
        render_discard_dialog()
        self.assertEqual(app._layout.enabled, {"discard_confirm", "discard_cancel"})
        app._activate("discard_cancel")
        self.assertIsNotNone(state.snapshot().session)
        np.testing.assert_array_equal(state.snapshot().session.recorded_buffers.original, buffers.original)
        self.assertEqual(state.ui_projection().current_screen, Screen.REVIEW_EXPORT.value)

        app._layout = render_shell(
            pygame.Surface((900, 620)), build_view_model(state.ui_projection()),
            focused="", hovered=None, upload_voices=(), upload_path_entry="", upload_start_error=None,
        )
        app._activate("discard_session")
        render_discard_dialog()
        app._activate("discard_confirm")
        self.assertIsNone(state.snapshot().session)
        self.assertEqual(state.ui_projection().current_screen, Screen.HOME.value)

    def test_successful_export_sets_feedback_and_committed_path_before_clearing(self):
        from harmonix.session_state.commands import SetExportVoices

        pygame.init()
        buffers = RecordedBuffers(
            original=np.linspace(-0.5, 0.5, 64, dtype=np.float32),
            a_per_voice={"3rd": np.ones(64, dtype=np.float32) * 0.1},
            sample_rate=44100,
        )
        session = SessionState(mode=SessionMode.UPLOAD, source_file="song.wav", recorded_buffers=buffers)
        session.playback.available_voices = ("3rd",)
        state = StateStore(AppState(current_screen=Screen.REVIEW_EXPORT, session=session))
        state.dispatch(SetExportVoices(("3rd",)))
        worker = ExportWorker(state)
        app = PygameApp(state_store=state, export_worker=worker)

        with tempfile.TemporaryDirectory() as tempdir:
            destination = Path(tempdir) / "committed.wav"
            worker.start(destination, enabled_voices=("3rd",))
            self.assertTrue(worker.join(timeout=3.0))
            self.assertTrue(destination.is_file())
            self.assertEqual(state.ui_projection().session.export_last_export_path, str(destination))

            clear_session = app._clear_session_to_home

            def assert_feedback_then_clear():
                self.assertEqual(app._export_toast_path, str(destination))
                self.assertIn(destination.name, app._export_toast)
                self.assertEqual(state.ui_projection().session.export_status, "completed")
                self.assertTrue(destination.is_file())
                clear_session()

            app._clear_session_to_home = assert_feedback_then_clear
            app._finish_export_if_complete()

            self.assertIsNone(state.snapshot().session)
            self.assertEqual(state.ui_projection().current_screen, Screen.HOME.value)
            app._layout = render_shell(
                pygame.Surface((900, 620)), build_view_model(state.ui_projection()),
                focused="", hovered=None, upload_voices=(), upload_path_entry="",
                upload_start_error=None, export_toast=app._export_toast,
            )
            self.assertIn("show_export_folder", app._layout.enabled)
            with patch("os.startfile", create=True) as startfile:
                app._activate("show_export_folder")
            startfile.assert_called_once_with(str(destination.parent))

    def test_cancelled_save_dialog_keeps_review_state_and_does_not_start_worker(self):
        buffers = RecordedBuffers(
            original=np.ones(16, dtype=np.float32),
            a_per_voice={"3rd": np.zeros(16, dtype=np.float32)},
            sample_rate=44100,
        )
        session = SessionState(mode=SessionMode.UPLOAD, source_file="song.wav", recorded_buffers=buffers)
        session.export.selected_voices = ("3rd",)
        state = StateStore(AppState(current_screen=Screen.REVIEW_EXPORT, session=session))

        class ExportStub:
            starts = []

            def start(self, destination, *, enabled_voices):
                self.starts.append((destination, enabled_voices))

        worker = ExportStub()
        app = PygameApp(state_store=state, export_worker=worker)
        tk_module = ModuleType("tkinter")
        dialog_module = ModuleType("tkinter.filedialog")

        class Root:
            def withdraw(self):
                pass

            def destroy(self):
                pass

        tk_module.Tk = Root
        dialog_module.asksaveasfilename = lambda **_kwargs: ""
        tk_module.filedialog = dialog_module
        with patch.dict(sys.modules, {"tkinter": tk_module, "tkinter.filedialog": dialog_module}):
            app._export_dialog_pending = True
            app._open_export_dialog()
            app._collect_runtime_events()

        self.assertFalse(app._export_dialog_pending)
        self.assertEqual(worker.starts, [])
        snapshot = state.snapshot()
        self.assertIsNotNone(snapshot.session)
        np.testing.assert_array_equal(snapshot.session.recorded_buffers.original, buffers.original)
        self.assertEqual(snapshot.session.export.selected_voices, ("3rd",))
        self.assertEqual(snapshot.current_screen, Screen.REVIEW_EXPORT)

    def test_confirmed_close_waits_for_active_export_termination_before_cleanup_and_lock_release(self):
        class ControlledEngine(ExportEngine):
            def __init__(self):
                self.entered = Event()
                self.release = Event()
                self.finished = Event()

            def export(self, recorded_buffers, destination, *, enabled_voices, cancel_event, **kwargs):
                self.entered.set()
                try:
                    if not self.release.wait(timeout=5.0):
                        raise TimeoutError("controlled export was not released by the test")
                    self._raise_if_cancelled(cancel_event)
                    return super().export(
                        recorded_buffers, destination, enabled_voices=enabled_voices,
                        cancel_event=cancel_event, **kwargs,
                    )
                finally:
                    self.finished.set()

        pygame.init()
        events = []

        class Persistence:
            def release_lifetime_lock(self):
                events.append(("lock_release", worker.is_alive))

        buffers = RecordedBuffers(
            original=np.ones(64, dtype=np.float32),
            a_per_voice={"3rd": np.zeros(64, dtype=np.float32)},
            sample_rate=44100,
        )
        session = SessionState(mode=SessionMode.UPLOAD, source_file="song.wav", recorded_buffers=buffers)
        session.playback.available_voices = ("3rd",)
        state = StateStore(AppState(current_screen=Screen.REVIEW_EXPORT, session=session))
        engine = ControlledEngine()
        worker = ExportWorker(state, engine)
        app = PygameApp(state_store=state, persistence_store=Persistence(), export_worker=worker)
        app._lifetime_lock_acquired = True
        app._pygame_init_started = True
        app._initialized = True
        app._running = True

        with tempfile.TemporaryDirectory() as tempdir:
            worker.start(Path(tempdir) / "cancelled.wav", enabled_voices=())
            self.assertTrue(engine.entered.wait(timeout=2.0))
            app.handle_event(pygame.event.Event(pygame.QUIT))
            self.assertTrue(app._close_confirmation_pending)
            app._layout = render_shell(
                pygame.Surface((900, 620)), build_view_model(state.ui_projection()),
                focused="", hovered=None, upload_voices=(), upload_path_entry="",
                upload_start_error=None, close_confirmation_pending=True,
            )
            app._activate("close_export_confirm")
            self.assertTrue(state.snapshot().session.export.cancellation_requested)
            self.assertTrue(worker.is_alive)
            self.assertTrue(app.running)

            engine.release.set()
            self.assertTrue(worker.join(timeout=3.0))
            self.assertTrue(engine.finished.is_set())
            self.assertEqual(state.ui_projection().session.export_status, "cancelled")
            app._poll_close_after_export()
            self.assertFalse(app.running)

            def pygame_cleanup():
                events.append(("pygame_cleanup", worker.is_alive))

            with patch("harmonix.ui.pygame_app.pygame.quit", side_effect=pygame_cleanup):
                app.shutdown()

        self.assertFalse(worker.is_alive)
        self.assertEqual(events, [("pygame_cleanup", False), ("lock_release", False)])

    def test_recent_upload_restoration_uses_saved_values_not_global_defaults(self):
        pygame.init()
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tempdir:
            source = Path(tempdir) / "restore.wav"
            source.write_bytes(b"audio")
            record = RecentSessionSummary(
                id="upload-1",
                display_name="restore.wav",
                mode=SessionMode.UPLOAD,
                source_file_path=str(source),
                last_used_key_root="F#",
                last_used_key_scale_type="minor",
                voice_toggles=(("3rd", False), ("5th", True), ("octave", True)),
                style_preset=StylePreset.CHOIR,
                timestamp="2025-01-01T00:00:00+00:00",
            )
            state = StateStore(AppState(
                settings=GlobalSettings(default_key_behavior=DefaultKeyBehavior.AUTO, default_style_preset=StylePreset.POP),
                recent_sessions=[record],
                last_manual_key=KeyState("D", "major", False),
            ))

            class Worker:
                args = None

                def start(self, path, voices, *, manual_key=None, style_preset=None, keep_configure_open=False):
                    self.args = (path, tuple(voices), manual_key, style_preset, keep_configure_open)
                    requested = KeyState(manual_key.root, manual_key.scale_type, False) if manual_key else None
                    state.dispatch(StartUploadAnalysis("restored", path, tuple(voices), style_preset, requested, keep_configure_open))

            worker = Worker()
            app = PygameApp(state_store=state, upload_worker=worker)
            app._layout = render_shell(
                pygame.Surface((900, 620)),
                build_view_model(state.ui_projection()),
                focused="",
                hovered=None,
                upload_voices=(),
                upload_path_entry="",
                upload_start_error=None,
            )
            self.assertIn("recent:upload-1", app._layout.enabled)
            app._activate("recent:upload-1")

            self.assertEqual(worker.args, (str(source), ("5th", "octave"), KeyScale("F#", "minor"), StylePreset.CHOIR, True))
            restored = state.snapshot().session
            self.assertEqual(restored.style_preset, StylePreset.CHOIR)
            self.assertEqual(restored.pending_key, KeyState("F#", "minor", False))
            self.assertEqual(restored.voices.active, {"3rd": False, "5th": True, "octave": True})
            self.assertEqual(state.snapshot().last_manual_key, KeyState("D", "major", False))
            self.assertEqual(state.ui_projection().current_screen, Screen.UPLOAD_CONFIGURE.value)

    def test_live_recent_entry_is_visible_but_not_activatable(self):
        pygame.init()
        recent_live = RecentSessionSummary(
            id="live-1",
            display_name="Live session — today",
            mode=SessionMode.LIVE,
            source_file_path=None,
            last_used_key_root=None,
            last_used_key_scale_type=None,
            voice_toggles=(),
            style_preset=None,
            timestamp="2025-01-01T00:00:00+00:00",
        )
        state = StateStore(AppState(recent_sessions=[recent_live]))

        class Worker:
            starts = 0

            def start(self, *args, **kwargs):
                self.starts += 1

        worker = Worker()
        app = PygameApp(state_store=state, upload_worker=worker)
        layout = render_shell(
            pygame.Surface((900, 620)),
            build_view_model(state.ui_projection()),
            focused="",
            hovered=None,
            upload_voices=(),
            upload_path_entry="",
            upload_start_error=None,
        )
        app._layout = layout
        self.assertIn("recent:live-1", layout.controls)
        self.assertNotIn("recent:live-1", layout.enabled)
        app._activate("recent:live-1")
        self.assertEqual(worker.starts, 0)
        self.assertEqual(state.ui_projection().current_screen, Screen.HOME.value)

    def test_home_navigation_preserves_session_and_allows_resume_before_new_upload(self):
        pygame.init()
        session = SessionState(
            mode=SessionMode.UPLOAD,
            source_file="current.wav",
            key=KeyState("A", "major", False),
        )
        state = StateStore(AppState(current_screen=Screen.UPLOAD_PLAYBACK, session=session))
        app = PygameApp(state_store=state)

        app._navigate(Screen.HOME)

        self.assertIsNotNone(state.snapshot().session)
        app._layout = render_shell(
            pygame.Surface((900, 620)),
            build_view_model(state.ui_projection()),
            focused="",
            hovered=None,
            upload_voices=(),
            upload_path_entry="",
            upload_start_error=None,
        )
        self.assertIn("resume_session", app._layout.enabled)
        app._activate("resume_session")
        self.assertEqual(state.ui_projection().current_screen, Screen.UPLOAD_CONFIGURE.value)

        app._start_upload("next.wav")
        self.assertEqual(state.snapshot().session.source_file, "current.wav")
        self.assertIn("End the current session", app._upload_start_error)

    def test_live_start_and_stop_transition_preserves_buffers_and_navigates_to_review_export(self):
        class DummyPersistence:
            settings_warning = None
            def acquire_lifetime_lock(self): pass
            def release_lifetime_lock(self): pass
            def load_settings(self): return GlobalSettings()
            def load_recent_sessions(self): return []

        class FakeLiveCoordinator:
            def __init__(self, visualizer_bridge=None):
                self.visualizer_bridge = visualizer_bridge
                self.active_voices = set()
                self.is_alive = False
                self.started = False
                self.stopped = False
                self.request_id = ""
                self.buffers = RecordedBuffers(
                    original=np.array([0.1, -0.1], dtype=np.float32),
                    a_per_voice={"3rd": np.array([0.2, -0.2], dtype=np.float32), "5th": np.array([0.3, -0.3], dtype=np.float32), "octave": None},
                    sample_rate=44100,
                )

            def set_active_voices(self, voices):
                self.active_voices = set(voices)

            def start(self, device_index=None, request_id=None):
                self.started = True
                self.is_alive = True
                if request_id:
                    self.request_id = request_id

            def stop(self, timeout=1.0):
                self.stopped = True
                self.is_alive = False
                return True

            def shutdown(self, timeout=None):
                self.stopped = True
                self.is_alive = False
                return True

            def recorded_buffers(self):
                return self.buffers

        state = StateStore()
        coordinator = FakeLiveCoordinator()
        app = PygameApp(state_store=state, live_coordinator=coordinator, persistence_store=DummyPersistence())
        app.initialize()
        app._audio_input_devices = [
            DeviceInfo(index=0, name="Mic", host_api_name="MME", max_input_channels=1, max_output_channels=0, default_samplerate=44100.0, is_default=True)
        ]

        app._navigate(Screen.LIVE_CONFIGURE)
        self.assertEqual(state.ui_projection().current_screen, Screen.LIVE_CONFIGURE.value)
        app.render()

        app._activate("start_live")
        self.assertTrue(coordinator.started)
        self.assertEqual(state.ui_projection().current_screen, Screen.LIVE_SESSION.value)
        self.assertTrue(state.snapshot().session.playback.is_recording)

        app.render()
        app._activate("stop_live")
        self.assertTrue(coordinator.stopped)
        self.assertEqual(state.ui_projection().current_screen, Screen.REVIEW_EXPORT.value)
        self.assertFalse(state.snapshot().session.playback.is_recording)
        np.testing.assert_array_equal(state.snapshot().session.recorded_buffers.original, coordinator.buffers.original)
        self.assertIs(state.export_source().recorded_buffers, coordinator.buffers)
        self.assertEqual(state.snapshot().session.export.selected_voices, ("3rd", "5th"))

        app.shutdown()

    def test_live_voice_selection_in_review_export(self):
        class DummyPersistence:
            settings_warning = None
            def acquire_lifetime_lock(self): pass
            def release_lifetime_lock(self): pass
            def load_settings(self): return GlobalSettings()
            def load_recent_sessions(self): return []

        buffers = RecordedBuffers(
            original=np.array([0.1, -0.1], dtype=np.float32),
            a_per_voice={"3rd": np.array([0.2, -0.2], dtype=np.float32), "5th": np.array([0.3, -0.3], dtype=np.float32), "octave": None},
            sample_rate=44100,
        )
        session = SessionState(
            mode=SessionMode.LIVE,
            live_request_id="live-export-test",
            recorded_buffers=buffers,
            playback=PlaybackState(available_voices=("3rd", "5th"), is_playing=False, is_recording=False),
        )
        session.export.selected_voices = ("3rd", "5th")
        state = StateStore(AppState(current_screen=Screen.REVIEW_EXPORT, session=session))
        app = PygameApp(state_store=state, persistence_store=DummyPersistence())
        app.initialize()
        app.render()

        app._activate("export_voice:5th")
        self.assertEqual(state.snapshot().session.export.selected_voices, ("3rd",))

        app.render()
        app._activate("export_voice:3rd")
        self.assertEqual(state.snapshot().session.export.selected_voices, ())

        app.render()
        app._activate("export_voice:3rd")
        self.assertEqual(state.snapshot().session.export.selected_voices, ("3rd",))

        app.shutdown()

    def test_live_telemetry_propagation_to_state(self):
        class DummyPersistence:
            settings_warning = None
            def acquire_lifetime_lock(self): pass
            def release_lifetime_lock(self): pass
            def load_settings(self): return GlobalSettings()
            def load_recent_sessions(self): return []

        class FakeLiveCoordinator:
            def __init__(self, visualizer_bridge=None):
                self.visualizer_bridge = visualizer_bridge
                self.is_alive = True

            def start(self, device_index=None, request_id=None):
                pass

            def stop(self, timeout=1.0):
                self.is_alive = False
                return True

            def shutdown(self, timeout=None):
                self.is_alive = False
                return True

            def set_active_voices(self, voices):
                pass

        session = SessionState(mode=SessionMode.LIVE, live_request_id="live-tel-test")
        state = StateStore(AppState(current_screen=Screen.LIVE_SESSION, session=session))
        coordinator = FakeLiveCoordinator()
        app = PygameApp(state_store=state, live_coordinator=coordinator, persistence_store=DummyPersistence())
        app._live_request_id = "live-tel-test"
        app.initialize()

        frame = LiveTelemetryFrame(
            request_id="live-tel-test",
            input_level=0.35,
            current_f0=440.0,
            confidence=0.92,
            tracking_state="tracking",
            in_tune=True,
            detected_key=KeyScale("G", "major"),
            effective_key=KeyScale("G", "major"),
            physical_latency_ms=30.0,
            pipeline_latency_ms=12.0,
            latency_provenance="pipeline_callback_to_dac",
        )
        app.visualizer_bridge.publish(frame)

        app.render()

        projection = state.ui_projection().session
        self.assertEqual(projection.current_f0, 440.0)
        self.assertTrue(projection.in_tune)
        self.assertEqual(projection.detected_key_root, "G")
        self.assertEqual(projection.physical_latency_ms, 30.0)
        self.assertEqual(projection.pipeline_latency_ms, 12.0)

        app.shutdown()

    def test_live_worker_shutdown_safety_before_lock_release(self):
        events = []

        class LockStore:
            def acquire_lifetime_lock(self):
                events.append("lock_acquired")

            def load_settings(self):
                return GlobalSettings()

            def load_recent_sessions(self):
                return []

            settings_warning = None

            def release_lifetime_lock(self):
                events.append("lock_released")

        class StubLiveCoordinator:
            def __init__(self, should_fail=False):
                self.is_alive = True
                self.should_fail = should_fail

            def shutdown(self, timeout=None):
                events.append(("shutdown_call", timeout))
                if self.should_fail:
                    return False
                self.is_alive = False
                return True

        stuck_coordinator = StubLiveCoordinator(should_fail=True)
        app_fail = PygameApp(persistence_store=LockStore(), live_coordinator=stuck_coordinator)
        app_fail.initialize()
        events.clear()

        with self.assertRaises(RuntimeError):
            app_fail.shutdown()

        self.assertIn(("shutdown_call", None), events)
        self.assertNotIn("lock_released", events)

        clean_coordinator = StubLiveCoordinator(should_fail=False)
        app_clean = PygameApp(persistence_store=LockStore(), live_coordinator=clean_coordinator)
        app_clean.initialize()
        events.clear()

        app_clean.shutdown()
        self.assertEqual(events, [("shutdown_call", None), "lock_released"])
        self.assertFalse(clean_coordinator.is_alive)

    def test_live_start_fallback_updates_global_settings(self):
        """EC-1.5: When live start triggers device fallback, GlobalSettings.audio_input_device
        must be persisted with the validated fallback device selector."""
        saved_settings = []

        class StubStore:
            settings_warning = None
            def acquire_lifetime_lock(self): pass
            def release_lifetime_lock(self): pass
            def load_settings(self): return GlobalSettings(audio_input_device="Stale Mic, WASAPI")
            def load_recent_sessions(self): return []
            def save_settings(self, settings):
                saved_settings.append(settings)
                return settings

        class FallbackCoordinator:
            def __init__(self):
                self.is_alive = True
                self.visualizer_bridge = None
                class Adapter:
                    fallback_occurred = True
                self.input_adapter = Adapter()

            def set_active_voices(self, voices): pass
            def start(self, device_index=None, request_id=None):
                return DeviceInfo(3, "Default Mic", 1, 0, 44100, True, "WASAPI")
            def stop(self, timeout=1.0):
                self.is_alive = False
                return True
            def shutdown(self, timeout=None):
                self.is_alive = False
                return True

        store = StateStore(AppState(settings=GlobalSettings(audio_input_device="Stale Mic, WASAPI")))
        app = PygameApp(state_store=store, persistence_store=StubStore(), live_coordinator=FallbackCoordinator())
        app._start_live()

        self.assertIsNone(app._live_start_error)
        self.assertIsNotNone(store.snapshot().session)
        self.assertEqual(store.snapshot().settings.audio_input_device, "Default Mic, WASAPI")
        self.assertTrue(len(saved_settings) > 0)
        self.assertEqual(saved_settings[-1].audio_input_device, "Default Mic, WASAPI")
        self.assertEqual(app._audio_input_selector, "Default Mic, WASAPI")

    def test_live_start_fallback_failure_preserves_old_setting(self):
        """EC-1.5: If fallback fails during live start, the old saved setting must be preserved
        and the app must remain cleanly retryable."""
        saved_settings = []

        class StubStore:
            settings_warning = None
            def acquire_lifetime_lock(self): pass
            def release_lifetime_lock(self): pass
            def load_settings(self): return GlobalSettings(audio_input_device="Stale Mic, WASAPI")
            def load_recent_sessions(self): return []
            def save_settings(self, settings):
                saved_settings.append(settings)
                return settings

        class FailingFallbackCoordinator:
            def __init__(self):
                self.is_alive = False
                self.visualizer_bridge = None
                class Adapter:
                    fallback_occurred = False
                self.input_adapter = Adapter()

            def set_active_voices(self, voices): pass
            def start(self, device_index=None, request_id=None):
                raise NoInputDeviceError("Saved input device is stale and no system-default input device is available.")
            def stop(self, timeout=1.0): return True
            def shutdown(self, timeout=None): return True

        store = StateStore(AppState(settings=GlobalSettings(audio_input_device="Stale Mic, WASAPI")))
        app = PygameApp(state_store=store, persistence_store=StubStore(), live_coordinator=FailingFallbackCoordinator())
        app._start_live()

        self.assertIsNotNone(app._live_start_error)
        self.assertIn("no system-default", app._live_start_error)
        self.assertIsNone(store.snapshot().session)
        self.assertEqual(store.snapshot().settings.audio_input_device, "Stale Mic, WASAPI")
        self.assertEqual(len(saved_settings), 0)

    def test_live_start_permission_denied_preserves_setting_and_surfaces_error(self):
        """EC-1.2: OS permission-denied error propagates distinctly, preserves settings,
        and leaves the app retryable."""
        class PermissionDeniedCoordinator:
            def __init__(self):
                self.is_alive = False
                self.visualizer_bridge = None
                class Adapter:
                    fallback_occurred = False
                self.input_adapter = Adapter()

            def set_active_voices(self, voices): pass
            def start(self, device_index=None, request_id=None):
                raise PermissionDeniedError("Microphone access was denied by the operating system")
            def stop(self, timeout=1.0): return True
            def shutdown(self, timeout=None): return True

        store = StateStore(AppState(settings=GlobalSettings(audio_input_device="Valid Mic, WASAPI")))
        app = PygameApp(state_store=store, live_coordinator=PermissionDeniedCoordinator())
        app._start_live()

        self.assertIsNotNone(app._live_start_error)
        self.assertIn("denied", app._live_start_error.lower())
        self.assertIsNone(store.snapshot().session)
        self.assertEqual(store.snapshot().settings.audio_input_device, "Valid Mic, WASAPI")

    def test_live_start_fallback_persistence_failure_retains_error_and_does_not_falsely_claim_persisted(self):
        """Requirement 6: If settings persistence fails after live audio startup, report the error,
        retain the running stream without rollback disruption, and do not falsely claim persisted."""
        class FailingPersistenceStore:
            settings_warning = None
            def acquire_lifetime_lock(self): pass
            def release_lifetime_lock(self): pass
            def load_settings(self): return GlobalSettings(audio_input_device="Stale Mic, WASAPI")
            def load_recent_sessions(self): return []
            def save_settings(self, settings):
                raise PersistenceWriteError("disk full or unwritable")

        class FallbackCoordinator:
            def __init__(self):
                self.is_alive = True
                self.visualizer_bridge = None
                class Adapter:
                    fallback_occurred = True
                self.input_adapter = Adapter()

            def set_active_voices(self, voices): pass
            def start(self, device_index=None, request_id=None):
                return DeviceInfo(3, "Default Mic", 1, 0, 44100, True, "WASAPI")
            def stop(self, timeout=1.0):
                self.is_alive = False
                return True
            def shutdown(self, timeout=None):
                self.is_alive = False
                return True

        store = StateStore(AppState(settings=GlobalSettings(audio_input_device="Stale Mic, WASAPI")))
        app = PygameApp(state_store=store, persistence_store=FailingPersistenceStore(), live_coordinator=FallbackCoordinator())
        app._start_live()

        # Audio stream started successfully and session is live
        self.assertIsNone(app._live_start_error)
        self.assertIsNotNone(store.snapshot().session)
        # Persistence error is reported explicitly
        self.assertIsNotNone(app._settings_error)
        # GlobalSettings in state store was NOT updated (truthful persistence state)
        self.assertEqual(store.snapshot().settings.audio_input_device, "Stale Mic, WASAPI")


if __name__ == "__main__":
    unittest.main()
