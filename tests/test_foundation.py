"""Milestone 1 tests using the Python standard library test harness."""

import unittest
import tempfile
from unittest.mock import patch

from harmonix.config.models import MAX_RECENT_SESSIONS, MAX_UPLOAD_DURATION_SECONDS
from harmonix.config.validation import (
    format_audio_device_selector,
    resolve_audio_device_selector,
    validate_frame_size,
    validate_settings,
    validate_setting_update,
    validate_upload_metadata,
)
from harmonix.config.upload_presets import UPLOAD_PRESET_VOICES
from harmonix.domain.enums import DefaultKeyBehavior, Screen, SessionMode, StylePreset
from harmonix.domain.errors import ConfigurationError, DeviceUnavailableError, StateError
from harmonix.domain.models import AppState, GlobalSettings, KeyState, RecordedBuffers, RecentSessionSummary, SessionState
from harmonix.session_state.app_state import StateStore
from harmonix.session_state.commands import (
    CancelExport,
    ClearSession,
    CompleteExport,
    FailExport,
    Navigate,
    RequestExportCancellation,
    StartExport,
    UpdateExportProgress,
    UpdateGlobalSetting,
    SelectUploadKey,
    ReplaceRecentSessions,
    SetExportVoices,
    StartUploadAnalysis,
    LiveTelemetryUpdated,
    StartLiveSession,
    StopLiveSession,
)
from harmonix.key_scale_resolver import KeyScale
from harmonix.visualizer_bridge import LiveTelemetryFrame, VisualizerBridge
from harmonix.persistence.store import JSONPersistenceStore


class FoundationTests(unittest.TestCase):
    def test_live_telemetry_is_canonical_and_rejects_stale_request(self) -> None:
        session = SessionState(mode=SessionMode.LIVE, live_request_id="live-current")
        store = StateStore(AppState(session=session))
        frame = LiveTelemetryUpdated(
            request_id="live-current",
            input_level=0.25,
            current_f0=440.0,
            confidence=0.91,
            tracking_state="tracking",
            in_tune=True,
            detected_key=KeyState("C", "major"),
            effective_key=KeyState("G", "major", False),
            physical_latency_ms=None,
            pipeline_latency_ms=42.0,
            latency_provenance="pipeline_callback_to_dac",
        )

        store.dispatch(frame)
        projection = store.ui_projection().session
        self.assertTrue(projection.mic_connected)
        self.assertEqual(projection.input_level, 0.25)
        self.assertEqual(projection.current_f0, 440.0)
        self.assertEqual(projection.tracking_confidence, 0.91)
        self.assertEqual(projection.tracking_state, "tracking")
        self.assertTrue(projection.in_tune)
        self.assertEqual(projection.latency_provenance, "pipeline_callback_to_dac")
        self.assertIsNone(projection.physical_latency_ms)
        self.assertEqual(projection.pipeline_latency_ms, 42.0)
        self.assertEqual(store.snapshot().session.detected_key, KeyState("C", "major"))
        self.assertEqual(store.snapshot().session.key, KeyState("G", "major", False))

        before = store.snapshot().session
        store.dispatch(
            LiveTelemetryUpdated(
                **{**frame.__dict__, "request_id": "stale-live-request", "current_f0": 220.0}
            )
        )
        self.assertEqual(store.snapshot().session, before)

    def test_visualizer_bridge_coalesces_snapshots_without_blocking_reader(self) -> None:
        bridge = VisualizerBridge()
        self.assertIsNone(bridge.read_after(0))
        first = LiveTelemetryFrame(
            request_id="live-1", input_level=0.1, current_f0=None, confidence=0.0,
            tracking_state="idle", in_tune=False, detected_key=None, effective_key=None,
            physical_latency_ms=None, pipeline_latency_ms=None, latency_provenance="unavailable",
        )
        latest = LiveTelemetryFrame(
            request_id="live-1", input_level=0.5, current_f0=440.0, confidence=0.8,
            tracking_state="tracking", in_tune=True, detected_key=KeyScale("C", "major"),
            effective_key=KeyScale("C", "major"), physical_latency_ms=None,
            pipeline_latency_ms=12.0, latency_provenance="pipeline_callback_to_dac",
        )

        self.assertEqual(bridge.publish(first), 1)
        self.assertEqual(bridge.publish(latest), 2)
        result = bridge.read_after(1)
        self.assertEqual(result, (2, latest))
        self.assertIsNone(bridge.read_after(2))

    def test_export_voice_selection_is_canonical_separate_from_playback_and_survives_start(self) -> None:
        session = SessionState(mode=SessionMode.UPLOAD)
        session.recorded_buffers = RecordedBuffers(original="original", a_per_voice={"3rd": "third"}, sample_rate=44100)
        session.voices.active["3rd"] = False
        store = StateStore(AppState(session=session))

        store.dispatch(SetExportVoices(("3rd",)))
        self.assertEqual(store.snapshot().session.export.selected_voices, ("3rd",))
        self.assertFalse(store.snapshot().session.voices.active["3rd"])

        store.publish_export_event(StartExport("wav-request"))
        state = store.snapshot().session
        self.assertTrue(state.export.in_progress)
        self.assertEqual(state.export.selected_voices, ("3rd",))
        self.assertFalse(state.voices.active["3rd"])
        self.assertEqual(store.ui_projection().session.export_selected_voices, ("3rd",))

        with self.assertRaises(StateError):
            store.dispatch(SetExportVoices(()))

    def test_app_state_has_specified_foundation_defaults(self) -> None:
        state = AppState()

        self.assertEqual(state.current_screen, Screen.HOME)
        self.assertIsNone(state.session)
        self.assertEqual(state.recent_sessions, [])
        self.assertEqual(MAX_RECENT_SESSIONS, 5)

    def test_session_contains_specified_nested_state(self) -> None:
        state = SessionState(mode=SessionMode.LIVE)

        self.assertEqual(set(state.voices.active), {"3rd", "5th", "octave"})
        self.assertFalse(state.playback.is_playing)
        self.assertFalse(state.audio_status.mic_connected)
        self.assertFalse(state.export.in_progress)

    def test_state_store_serializes_commands_and_returns_snapshots(self) -> None:
        store = StateStore()

        changed = store.dispatch(Navigate(Screen.SETTINGS))
        changed.current_screen = Screen.LIVE_SESSION

        self.assertEqual(store.snapshot().current_screen, Screen.SETTINGS)
        self.assertEqual(store.dispatch(ClearSession()).session, None)

    def test_upload_continuation_requires_current_request_and_matching_source(self) -> None:
        session = SessionState(mode=SessionMode.UPLOAD, source_file="active.wav")
        session.key = KeyState("D", "minor", False)
        session.upload_analysis.request_id = "current-request"
        session.upload_analysis.status = "completed"
        old_buffers = RecordedBuffers(original="previous-buffer", a_per_voice={}, sample_rate=44100)
        session.recorded_buffers = old_buffers
        store = StateStore(AppState(session=session))

        rejected = (
            StartUploadAnalysis("missing", "active.wav", ("3rd",)),
            StartUploadAnalysis("stale", "active.wav", ("3rd",), continuation_request_id="older-request"),
            StartUploadAnalysis("mismatched", "other.wav", ("3rd",), continuation_request_id="current-request"),
        )
        for command in rejected:
            with self.subTest(request_id=command.request_id), self.assertRaises(StateError):
                store.dispatch(command)
            unchanged = store.snapshot().session
            self.assertEqual(unchanged.source_file, "active.wav")
            self.assertEqual(unchanged.upload_analysis.request_id, "current-request")
            self.assertEqual(unchanged.key, KeyState("D", "minor", False))
            self.assertEqual(unchanged.recorded_buffers.original, old_buffers.original)

        store.dispatch(
            StartUploadAnalysis(
                "continued-request",
                "active.wav",
                ("3rd",),
                continuation_request_id="current-request",
            )
        )
        self.assertEqual(store.snapshot().session.upload_analysis.request_id, "continued-request")
        self.assertIsNone(store.snapshot().session.recorded_buffers.original)

        live_store = StateStore(AppState(session=SessionState(mode=SessionMode.LIVE)))
        with self.assertRaises(StateError):
            live_store.dispatch(StartUploadAnalysis("fresh", "unrelated.wav", ("3rd",)))
        self.assertEqual(live_store.snapshot().session.mode, SessionMode.LIVE)

    def test_settings_frame_size_defaults_and_enforces_documented_range(self) -> None:
        settings = GlobalSettings()
        self.assertEqual(settings.latency_buffer_frame_size, 2048)
        validate_settings(settings)
        validate_frame_size(256)
        validate_frame_size(4096)
        for invalid in (255, 4097, 256.0, True, False, None):
            with self.subTest(invalid=invalid), self.assertRaises(ConfigurationError):
                validate_frame_size(invalid)

    def test_only_approved_settings_values_are_accepted(self) -> None:
        for value in DefaultKeyBehavior:
            validate_setting_update("default_key_behavior", value)
        for value in StylePreset:
            validate_setting_update("default_style_preset", value)
        validate_setting_update("reduced_motion", True)
        validate_setting_update("reduced_motion", False)
        for value in ("guess", None):
            with self.subTest(value=value), self.assertRaises(ConfigurationError):
                validate_setting_update("default_key_behavior", value)
        for value in ("rock", None):
            with self.subTest(value=value), self.assertRaises(ConfigurationError):
                validate_setting_update("default_style_preset", value)
        with self.assertRaises(ConfigurationError):
            validate_setting_update("reduced_motion", 1)
        with self.assertRaises(ConfigurationError):
            validate_setting_update("audio_input_device", "device")

    def test_device_selector_is_validated_persisted_through_canonical_flow(self) -> None:
        input_selector = format_audio_device_selector("USB Mic", "Windows WASAPI")
        output_selector = format_audio_device_selector("Speakers", "Windows WASAPI")
        for field, selector in (
            ("audio_input_device", input_selector),
            ("audio_output_device", output_selector),
        ):
            validate_setting_update(field, selector)

        initial = GlobalSettings(
            default_key_behavior=DefaultKeyBehavior.LAST_USED,
            default_style_preset=StylePreset.CHOIR,
            latency_buffer_frame_size=1024,
            reduced_motion=True,
        )
        state = StateStore(AppState(settings=initial))
        changed = state.dispatch(UpdateGlobalSetting("audio_input_device", input_selector))
        self.assertEqual(changed.settings.audio_input_device, input_selector)
        self.assertEqual(changed.settings.audio_output_device, initial.audio_output_device)
        self.assertEqual(changed.settings.default_key_behavior, initial.default_key_behavior)
        self.assertEqual(changed.settings.default_style_preset, initial.default_style_preset)
        self.assertEqual(changed.settings.latency_buffer_frame_size, initial.latency_buffer_frame_size)
        self.assertEqual(changed.settings.reduced_motion, initial.reduced_motion)

        before_invalid_update = state.snapshot()
        with self.assertRaises(ConfigurationError):
            state.dispatch(UpdateGlobalSetting("audio_output_device", "Speakers"))
        self.assertEqual(state.snapshot(), before_invalid_update)

        changed = state.dispatch(UpdateGlobalSetting("audio_output_device", output_selector))
        self.assertEqual(changed.settings.audio_input_device, input_selector)
        self.assertEqual(changed.settings.audio_output_device, output_selector)
        self.assertEqual(changed.settings.default_key_behavior, initial.default_key_behavior)
        self.assertEqual(changed.settings.default_style_preset, initial.default_style_preset)
        self.assertEqual(changed.settings.latency_buffer_frame_size, initial.latency_buffer_frame_size)
        self.assertEqual(changed.settings.reduced_motion, initial.reduced_motion)

        with tempfile.TemporaryDirectory() as tempdir:
            persistence = JSONPersistenceStore(tempdir)
            persistence.save_settings(changed.settings)
            loaded = persistence.load_settings()
        self.assertEqual(loaded.audio_input_device, input_selector)
        self.assertEqual(loaded.audio_output_device, output_selector)
        self.assertEqual(loaded.default_key_behavior, initial.default_key_behavior)
        self.assertEqual(loaded.default_style_preset, initial.default_style_preset)
        self.assertEqual(loaded.latency_buffer_frame_size, initial.latency_buffer_frame_size)
        self.assertEqual(loaded.reduced_motion, initial.reduced_motion)

        system_default = state.dispatch(UpdateGlobalSetting("audio_output_device", None))
        self.assertEqual(system_default.settings.audio_input_device, input_selector)
        self.assertIsNone(system_default.settings.audio_output_device)
        with tempfile.TemporaryDirectory() as tempdir:
            persistence = JSONPersistenceStore(tempdir)
            persistence.save_settings(system_default.settings)
            loaded_default = persistence.load_settings()
        self.assertEqual(loaded_default.audio_input_device, input_selector)
        self.assertIsNone(loaded_default.audio_output_device)

    def test_audio_device_selector_rejects_invalid_values(self) -> None:
        for value in ("", "USB Mic", "USB Mic, ", ", Windows WASAPI", " USB Mic, Windows WASAPI", 3, True):
            with self.subTest(value=value), self.assertRaises(ConfigurationError):
                validate_setting_update("audio_input_device", value)
        with self.assertRaises(ConfigurationError):
            format_audio_device_selector("USB Mic", "")

    def test_device_selector_requires_exact_unique_name_and_host_api_match(self) -> None:
        selector = format_audio_device_selector("USB Mic", "Windows WASAPI")
        device = {"index": 5, "name": "USB Mic", "host_api_name": "Windows WASAPI"}
        self.assertIs(resolve_audio_device_selector(selector, [device]), device)

        with self.assertRaisesRegex(ConfigurationError, "ambiguous"):
            resolve_audio_device_selector(selector, [device, {**device, "index": 9}])
        with self.assertRaises(DeviceUnavailableError):
            resolve_audio_device_selector(selector, [{**device, "host_api_name": "MME"}])

    def test_none_is_explicit_system_default_for_audio_device_selection(self) -> None:
        validate_setting_update("audio_input_device", None)
        validate_setting_update("audio_output_device", None)
        self.assertIsNone(resolve_audio_device_selector(None, []))
        state = StateStore(AppState(settings=GlobalSettings(audio_input_device="USB Mic, Windows WASAPI")))
        changed = state.dispatch(UpdateGlobalSetting("audio_input_device", None))
        self.assertIsNone(changed.settings.audio_input_device)

    def test_only_approved_upload_preset_toggle_mappings_exist(self) -> None:
        self.assertEqual(UPLOAD_PRESET_VOICES[StylePreset.POP], {"3rd", "5th"})
        self.assertEqual(UPLOAD_PRESET_VOICES[StylePreset.CHOIR], {"3rd", "5th", "octave"})
        self.assertNotIn(StylePreset.BARBERSHOP, UPLOAD_PRESET_VOICES)

    def test_settings_reducer_changes_only_the_requested_canonical_field(self) -> None:
        session = SessionState(mode=SessionMode.UPLOAD)
        prior = RecentSessionSummary(
            id="prior",
            display_name="prior.wav",
            mode=SessionMode.UPLOAD,
            source_file_path="prior.wav",
            last_used_key_root=None,
            last_used_key_scale_type=None,
            voice_toggles=(),
            style_preset=None,
            timestamp="2025-01-01T00:00:00+00:00",
        )
        store = StateStore(AppState(session=session, recent_sessions=[prior]))

        changed = store.dispatch(UpdateGlobalSetting("default_style_preset", StylePreset.CHOIR))

        self.assertEqual(changed.settings.default_style_preset, StylePreset.CHOIR)
        self.assertEqual(changed.settings.default_key_behavior, DefaultKeyBehavior.AUTO)
        self.assertEqual(changed.recent_sessions, [prior])
        self.assertIsNotNone(changed.session)
        self.assertEqual(store.snapshot().settings.default_style_preset, StylePreset.CHOIR)
        loaded = store.dispatch(ReplaceRecentSessions((prior,)))
        self.assertEqual(loaded.recent_sessions, [prior])

    def test_last_manual_key_is_canonical_in_memory_state_and_survives_session_clear(self) -> None:
        store = StateStore()
        selected = KeyState(root="D", scale_type="minor", auto_detect=False)
        changed = store.dispatch(SelectUploadKey(selected))
        self.assertEqual(changed.last_manual_key, selected)
        store.dispatch(ClearSession())
        self.assertEqual(store.snapshot().last_manual_key, selected)

    def test_recent_metadata_projection_does_not_copy_recorded_audio(self) -> None:
        session = SessionState(mode=SessionMode.UPLOAD, source_file="song.wav")
        session.key = KeyState("G", "minor", False)
        session.recorded_buffers.original = object()
        store = StateStore(AppState(session=session))

        with patch("harmonix.session_state.app_state.deepcopy", side_effect=AssertionError("must not copy audio")):
            metadata = store.session_metadata_source()

        self.assertEqual(metadata.source_file, "song.wav")
        self.assertEqual((metadata.key_root, metadata.key_scale_type), ("G", "minor"))

    def test_upload_constraints_are_explicit(self) -> None:
        validate_upload_metadata("wav", MAX_UPLOAD_DURATION_SECONDS)

        with self.assertRaises(ConfigurationError):
            validate_upload_metadata("wav", MAX_UPLOAD_DURATION_SECONDS + 1)

        with self.assertRaises(ConfigurationError):
            validate_upload_metadata("ogg", 1)

    def test_export_reducer_tracks_progress_lifecycle_and_rejects_stale_events(self) -> None:
        original = [0.1, 0.2]
        recorded = RecordedBuffers(original=original, a_per_voice={}, sample_rate=44100)
        store = StateStore(AppState(session=SessionState(mode=SessionMode.UPLOAD, recorded_buffers=recorded)))

        store.publish_export_event(StartExport("first"))
        store.publish_export_event(UpdateExportProgress("first", "rendering", 2, 4))
        self.assertEqual(store.snapshot().session.export.progress_pct, 50.0)
        store.publish_export_event(UpdateExportProgress("first", "rendering", 1, 4))
        self.assertEqual(store.snapshot().session.export.completed_units, 2)
        store.publish_export_event(UpdateExportProgress("first", "rendering", 99, 4))
        self.assertEqual(store.snapshot().session.export.completed_units, 4)
        self.assertEqual(store.snapshot().session.export.progress_pct, 100.0)

        store.publish_export_event(UpdateExportProgress("first", "writing", None, None))
        writing = store.snapshot().session.export
        self.assertEqual(writing.phase, "writing")
        self.assertEqual(writing.completed_units, 4)
        self.assertEqual(writing.total_units, 4)
        self.assertIsNone(writing.progress_pct)

        store.publish_export_event(CompleteExport("first", "C:/exports/first.wav"))
        complete = store.snapshot().session.export
        self.assertEqual(complete.status, "completed")
        self.assertFalse(complete.in_progress)
        self.assertEqual(complete.completed_units, 4)
        self.assertEqual(complete.total_units, 4)
        self.assertEqual(complete.last_export_path, "C:/exports/first.wav")
        self.assertEqual(complete.progress_pct, 100.0)

        store.publish_export_event(StartExport("second"))
        store.publish_export_event(UpdateExportProgress("first", "rendering", 4, 4))
        store.publish_export_event(CancelExport("first"))
        store.publish_export_event(FailExport("first", "stale", "stale failure"))
        store.publish_export_event(CompleteExport("first", "C:/exports/stale.wav"))
        current = store.snapshot().session.export
        self.assertEqual(current.request_id, "second")
        self.assertEqual(current.status, "running")
        self.assertIsNone(current.last_export_path)

        store.publish_export_event(UpdateExportProgress("second", "rendering", 3, 4))
        store.publish_export_event(RequestExportCancellation("second"))
        cancelling = store.snapshot().session.export
        self.assertEqual(cancelling.completed_units, 3)
        self.assertEqual(cancelling.total_units, 4)
        self.assertIsNone(cancelling.progress_pct)
        store.publish_export_event(CancelExport("second"))
        cancelled = store.snapshot().session.export
        self.assertEqual(cancelled.status, "cancelled")
        self.assertEqual(cancelled.completed_units, 3)
        self.assertEqual(cancelled.total_units, 4)
        self.assertIsNone(cancelled.progress_pct)
        self.assertTrue(cancelled.cancellation_requested)
        self.assertFalse(cancelled.in_progress)
        self.assertEqual(store.snapshot().session.recorded_buffers.original, original)

    def test_export_failure_preserves_audio_and_rejects_start_without_session(self) -> None:
        original = [0.25, -0.25]
        voices = {"3rd": [0.5, -0.5]}
        session = SessionState(
            mode=SessionMode.UPLOAD,
            recorded_buffers=RecordedBuffers(original=original, a_per_voice=voices, sample_rate=44100),
        )
        store = StateStore(AppState(session=session))
        store.publish_export_event(StartExport("failed-request"))
        store.publish_export_event(UpdateExportProgress("failed-request", "rendering", 2, 4))
        store.publish_export_event(FailExport("failed-request", "ExportWriteError", "disk unavailable"))

        failed = store.snapshot().session
        self.assertEqual(failed.export.status, "failed")
        self.assertIn("disk unavailable", failed.export.error)
        self.assertEqual(failed.export.completed_units, 2)
        self.assertEqual(failed.export.total_units, 4)
        self.assertIsNone(failed.export.progress_pct)
        self.assertEqual(failed.recorded_buffers.original, original)
        self.assertEqual(failed.recorded_buffers.a_per_voice["3rd"], voices["3rd"])

        no_session_store = StateStore()
        with self.assertRaises(StateError):
            no_session_store.publish_export_event(StartExport("orphan"))

    def test_start_live_session_creates_canonical_session_and_rejects_active_upload(self) -> None:
        store = StateStore()
        store.dispatch(StartLiveSession("live-test-1", selected_voices=("3rd", "octave"), style_preset=StylePreset.POP))
        snapshot = store.snapshot()
        self.assertEqual(snapshot.current_screen, Screen.LIVE_SESSION)
        self.assertIsNotNone(snapshot.session)
        self.assertEqual(snapshot.session.mode, SessionMode.LIVE)
        self.assertEqual(snapshot.session.live_request_id, "live-test-1")
        self.assertEqual(snapshot.session.voices.active, {"3rd": True, "5th": False, "octave": True})
        self.assertTrue(snapshot.session.playback.is_recording)

        upload_session = SessionState(mode=SessionMode.UPLOAD, source_file="song.wav")
        upload_store = StateStore(AppState(session=upload_session))
        with self.assertRaises(StateError):
            upload_store.dispatch(StartLiveSession("live-conflict"))

    def test_stop_live_session_preserves_recorded_buffers_and_initializes_export_defaults(self) -> None:
        session = SessionState(mode=SessionMode.LIVE, live_request_id="live-active")
        session.voices.active = {"3rd": True, "5th": False, "octave": True}
        store = StateStore(AppState(session=session))

        original = [0.1, -0.1]
        buffers = RecordedBuffers(
            original=original,
            a_per_voice={"3rd": [0.2, -0.2], "5th": [0.3, -0.3], "octave": None},
            sample_rate=44100,
        )
        changed = store.dispatch(StopLiveSession(buffers, request_id="live-active"))
        stopped = changed.session
        self.assertEqual(store.snapshot().current_screen, Screen.REVIEW_EXPORT)
        self.assertEqual(stopped.recorded_buffers, buffers)
        self.assertIs(store.export_source().recorded_buffers, buffers)
        # octave has None array, so only 3rd and 5th are available
        self.assertEqual(stopped.playback.available_voices, ("3rd", "5th"))
        # 3rd was active and available; 5th was inactive; octave was missing
        self.assertEqual(stopped.export.selected_voices, ("3rd",))
        self.assertFalse(stopped.playback.is_recording)

        # Original-only export when no harmony voices were active
        session_no_harmony = SessionState(mode=SessionMode.LIVE, live_request_id="live-no-harmony")
        session_no_harmony.voices.active = {"3rd": False, "5th": False, "octave": False}
        store_no_harmony = StateStore(AppState(session=session_no_harmony))
        store_no_harmony.dispatch(StopLiveSession(buffers, request_id="live-no-harmony"))
        self.assertEqual(store_no_harmony.snapshot().session.export.selected_voices, ())

        # Stale request is ignored
        before = store.snapshot()
        store.dispatch(StopLiveSession(buffers, request_id="stale-req"))
        self.assertEqual(store.snapshot().session.export.selected_voices, before.session.export.selected_voices)

        # Stopping without active live session raises StateError
        no_session = StateStore()
        with self.assertRaises(StateError):
            no_session.dispatch(StopLiveSession(buffers))

    def test_set_export_voices_permits_stopped_live_session_and_validates_voices(self) -> None:
        session = SessionState(mode=SessionMode.LIVE, live_request_id="live-done")
        session.playback.is_recording = False
        buffers = RecordedBuffers(
            original=[0.1, -0.1],
            a_per_voice={"3rd": [0.2, -0.2], "5th": [0.3, -0.3], "octave": None},
            sample_rate=44100,
        )
        session.recorded_buffers = buffers
        store = StateStore(AppState(session=session))

        # Can select valid available voices
        store.dispatch(SetExportVoices(("3rd", "5th")))
        self.assertEqual(store.snapshot().session.export.selected_voices, ("3rd", "5th"))

        # Can select single voice
        store.dispatch(SetExportVoices(("3rd",)))
        self.assertEqual(store.snapshot().session.export.selected_voices, ("3rd",))

        # Can select empty (original-only export)
        store.dispatch(SetExportVoices(()))
        self.assertEqual(store.snapshot().session.export.selected_voices, ())

        # Selecting unavailable voice raises StateError
        with self.assertRaises(StateError):
            store.dispatch(SetExportVoices(("octave",)))

        # Selecting during active recording raises StateError
        recording_session = SessionState(mode=SessionMode.LIVE, live_request_id="live-rec")
        recording_session.playback.is_recording = True
        recording_session.recorded_buffers = buffers
        recording_store = StateStore(AppState(session=recording_session))
        with self.assertRaises(StateError):
            recording_store.dispatch(SetExportVoices(("3rd",)))

        # No active session raises StateError
        empty_store = StateStore()
        with self.assertRaises(StateError):
            empty_store.dispatch(SetExportVoices(("3rd",)))


if __name__ == "__main__":
    unittest.main()
