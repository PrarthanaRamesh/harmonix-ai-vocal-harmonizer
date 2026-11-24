"""Pygame lifecycle, navigation, and nonblocking workflow orchestration."""

from __future__ import annotations

from queue import Empty, Queue
from dataclasses import replace
from datetime import datetime, timezone
from threading import Event, Thread
from uuid import uuid4
from pathlib import Path

import pygame

from ..config.validation import format_audio_device_selector, resolve_audio_device_selector, validate_setting_update
from ..config.upload_presets import UPLOAD_PRESET_VOICES
from ..domain.enums import DefaultKeyBehavior, Screen, StylePreset
from ..domain.errors import ConfigurationError, DeviceUnavailableError, PersistenceError, PersistenceWriteError
from ..domain.enums import SessionMode
from ..domain.models import GlobalSettings, KeyState
from ..export.worker import ExportWorker
from ..input.adapter import DeviceInfo, InputAdapter
from ..key_scale_resolver import KeyScale, _ROOTS
from ..live_audio import LiveAudioCoordinator
from ..playback_engine import PlaybackController
from ..persistence.store import JSONPersistenceStore, RecentSessionMetadata
from ..session_state.app_state import StateStore
from ..session_state.commands import (
    ClearSession,
    EndSession,
    Navigate,
    PlaybackFailed,
    ReplaceGlobalSettings,
    ReplaceRecentSessions,
    SelectUploadKey,
    SetExportVoices,
    StartLiveSession,
    StopLiveSession,
    LiveTelemetryUpdated,
    UpdateGlobalSetting,
)
from ..upload.analysis import UploadAnalysisWorker
from ..visualizer_bridge import VisualizerBridge
from . import theme
from .screens.shell import ShellLayout, render_shell
from .state_binding import build_view_model, downsample_waveform

UPLOAD_VOICES = ("3rd", "5th", "octave")
SCREEN_BY_ACTION = {
    "nav_live": Screen.LIVE_CONFIGURE,
    "choose_live": Screen.LIVE_CONFIGURE,
    "nav_upload": Screen.UPLOAD_SELECT,
    "choose_upload": Screen.UPLOAD_SELECT,
    "nav_home": Screen.HOME,
    "nav_settings": Screen.SETTINGS,
}


class PygameApp:
    """Own Pygame resources and transient worker/controller runtime objects."""

    def __init__(
        self,
        state_store: StateStore | None = None,
        window_size: tuple[int, int] = theme.WINDOW_SIZE,
        *,
        upload_worker: UploadAnalysisWorker | None = None,
        playback_controller_factory=None,
        persistence_store: JSONPersistenceStore | None = None,
        export_worker: ExportWorker | None = None,
        live_coordinator_factory=None,
        live_coordinator: LiveAudioCoordinator | None = None,
        visualizer_bridge: VisualizerBridge | None = None,
    ) -> None:
        self.state_store = state_store or StateStore()
        self.window_size = window_size
        self.upload_worker = upload_worker or UploadAnalysisWorker(self.state_store)
        self.playback_controller_factory = playback_controller_factory or PlaybackController
        self.persistence_store = persistence_store or JSONPersistenceStore()
        self.export_worker = export_worker or ExportWorker(self.state_store)
        self.visualizer_bridge = visualizer_bridge or VisualizerBridge()
        self.live_coordinator_factory = live_coordinator_factory
        self.live_coordinator: LiveAudioCoordinator | None = live_coordinator
        if self.live_coordinator is not None and getattr(self.live_coordinator, "visualizer_bridge", None) is None:
            self.live_coordinator.visualizer_bridge = self.visualizer_bridge
        self._live_telemetry_sequence = 0
        self._live_request_id: str | None = None
        self._live_start_error: str | None = None
        self._running = False
        self._initialized = False
        self._pygame_init_started = False
        self._lifetime_lock_acquired = False
        self._settings_loaded = False
        self._recent_sessions_loaded = False
        self._recent_metadata: list[RecentSessionMetadata] = []
        self._pending_session_end: tuple[RecentSessionMetadata, list[RecentSessionMetadata]] | None = None
        self._session_end_error: str | None = None
        self._discard_confirmation_pending = False
        self._close_confirmation_pending = False
        self._close_after_export = False
        self._export_dialog_pending = False
        self._export_completion_handled_request: str | None = None
        self._export_toast: str | None = None
        self._export_toast_path: str | None = None
        self._export_ui_error: str | None = None
        self._shutdown_complete = False
        self._settings_warning: str | None = None
        self._settings_error: str | None = None
        self._pending_setting: tuple[str, object] | None = None
        self._audio_input_devices: list[DeviceInfo] = []
        self._audio_output_devices: list[DeviceInfo] = []
        self._audio_device_error: str | None = None
        self._audio_devices_loaded = False
        self._audio_input_selector: str | None = None
        self._audio_output_selector: str | None = None
        self._surface: pygame.Surface | None = None
        self._layout = ShellLayout({}, set())
        self._focused = "choose_upload"
        self._hovered: str | None = None
        self._upload_voices = set(UPLOAD_VOICES)
        self._path_entry = ""
        self._upload_start_error: str | None = None
        self._live_key: KeyScale | None = None
        self._playback_requests: Queue[tuple[object, ...]] = Queue()
        self._runtime_events: Queue[tuple[str, object]] = Queue()
        self._runtime_stop = Event()
        self._runtime_thread: Thread | None = None
        self._waveform: tuple[tuple[float, float], ...] = ()
        self._waveform_source = None
        self._waveform_generation = 0

    @property
    def running(self) -> bool:
        return self._running

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(self) -> None:
        if self._initialized:
            return
        self._shutdown_complete = False
        self.persistence_store.acquire_lifetime_lock()
        self._lifetime_lock_acquired = True
        try:
            if not self._settings_loaded:
                settings = self.persistence_store.load_settings()
                self.state_store.dispatch(ReplaceGlobalSettings(settings))
                self._audio_input_selector = settings.audio_input_device
                self._audio_output_selector = settings.audio_output_device
                self._settings_warning = self.persistence_store.settings_warning
                self._settings_loaded = True
            if not self._recent_sessions_loaded:
                self._recent_metadata = self.persistence_store.load_recent_sessions()
                self.state_store.dispatch(
                    ReplaceRecentSessions(tuple(item.to_summary() for item in self._recent_metadata))
                )
                self._recent_sessions_loaded = True
            self._pygame_init_started = True
            pygame.init()
            self._surface = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption(theme.WINDOW_TITLE)
            self._runtime_stop.clear()
            self._runtime_thread = Thread(target=self._playback_runtime, name="harmonix-ui-playback-control", daemon=True)
            self._runtime_thread.start()
            self._initialized = True
        except BaseException:
            self.shutdown()
            raise

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self._request_close()
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self._request_close()
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_HOME:
            self._navigate(Screen.HOME)
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
            self._cycle_focus(1)
        elif event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_SPACE):
            self._activate(self._focused)
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_BACKSPACE and self._path_entry:
            self._path_entry = self._path_entry[:-1]
        elif event.type == pygame.TEXTINPUT and self.state_store.ui_projection().current_screen == Screen.UPLOAD_SELECT.value:
            self._path_entry += event.text
        elif event.type == pygame.MOUSEMOTION:
            self._hovered = self._control_at(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            name = self._control_at(event.pos)
            if name is not None:
                self._focused = name
                if name == "seek":
                    self._seek_at(event.pos[0])
                else:
                    self._activate(name)
        elif event.type == getattr(pygame, "DROPFILE", -1):
            self._start_upload(getattr(event, "file", ""))

    def render(self) -> None:
        if not self._initialized or self._surface is None:
            raise RuntimeError("PygameApp must be initialized before rendering")
        self._collect_runtime_events()
        self._poll_live_telemetry()
        self._finish_export_if_complete()
        self._poll_close_after_export()
        projection = self.state_store.ui_projection()
        if projection.current_screen in {Screen.UPLOAD_CONFIGURE.value, Screen.UPLOAD_PLAYBACK.value}:
            self._ensure_playback_source()
        view_model = build_view_model(projection, self._waveform)
        self._layout = render_shell(
            self._surface,
            view_model,
            focused=self._focused,
            hovered=self._hovered,
            upload_voices=tuple(voice for voice in UPLOAD_VOICES if voice in self._upload_voices),
            upload_path_entry=self._path_entry,
            upload_start_error=self._upload_start_error,
            settings_warning=self._settings_warning,
            settings_error=self._settings_error,
            settings_retry_pending=self._pending_setting is not None,
            input_devices=tuple(self._audio_input_devices),
            output_devices=tuple(self._audio_output_devices),
            input_device_selector=self._audio_input_selector,
            output_device_selector=self._audio_output_selector,
            audio_device_error=self._audio_device_error,
            audio_devices_loaded=self._audio_devices_loaded,
            session_end_error=self._session_end_error,
            session_end_pending=self._pending_session_end is not None,
            discard_confirmation_pending=self._discard_confirmation_pending,
            close_confirmation_pending=self._close_confirmation_pending,
            export_toast=self._export_toast,
            export_ui_error=self._export_ui_error,
            live_start_error=self._live_start_error,
            live_key=self._live_key,
        )
        pygame.display.flip()

    def run(self, max_frames: int | None = None) -> None:
        try:
            self.initialize()
            self._running = True
            clock = pygame.time.Clock()
            frames = 0
            while self._running and (max_frames is None or frames < max_frames):
                for event in pygame.event.get():
                    self.handle_event(event)
                self.render()
                clock.tick(60)
                frames += 1
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        if self._shutdown_complete:
            return
        if self._lifetime_lock_acquired:
            if not self.export_worker.shutdown(timeout=None):
                raise RuntimeError("WAV export worker has not terminated; application cleanup is deferred")
            if self.live_coordinator is not None and self.live_coordinator.is_alive:
                if not self.live_coordinator.shutdown(timeout=None):
                    raise RuntimeError("Live audio worker has not terminated; application cleanup is deferred")
            try:
                self.upload_worker.request_cancel()
            finally:
                self._playback_requests.put(("shutdown",))
                self._runtime_stop.set()
                runtime_thread = self._runtime_thread
                try:
                    if runtime_thread is not None and runtime_thread.ident is not None:
                        runtime_thread.join()
                finally:
                    self._runtime_thread = None
                    try:
                        if self._pygame_init_started:
                            pygame.quit()
                    finally:
                        self._surface = None
                        self._initialized = False
                        self._pygame_init_started = False
                        self._running = False
                        self._lifetime_lock_acquired = False
                        self.persistence_store.release_lifetime_lock()
        self._running = False
        self._shutdown_complete = True

    def _navigate(self, screen: Screen) -> None:
        previous_screen = self.state_store.ui_projection().current_screen
        if screen is Screen.UPLOAD_SELECT and previous_screen != Screen.UPLOAD_SELECT.value:
            preset = self.state_store.snapshot().settings.default_style_preset
            if preset in UPLOAD_PRESET_VOICES:
                self._upload_voices = set(UPLOAD_PRESET_VOICES[preset])
        if (
            self.state_store.ui_projection().current_screen == Screen.UPLOAD_PLAYBACK.value
            and screen is not Screen.UPLOAD_PLAYBACK
        ):
            self._playback_requests.put(("close",))
            self._controller_source_identity = None
        self.state_store.dispatch(Navigate(screen))
        if screen is not Screen.UPLOAD_SELECT:
            self._path_entry = ""

    def _activate(self, name: str | None) -> None:
        if not name or name not in self._layout.controls or name not in self._layout.enabled:
            return
        if self._discard_confirmation_pending and name not in {"discard_confirm", "discard_cancel"}:
            return
        if name in SCREEN_BY_ACTION:
            self._navigate(SCREEN_BY_ACTION[name])
        elif name == "start_live":
            self._start_live()
        elif name == "stop_live":
            self._stop_live()
        elif name == "resume_session":
            session = self.state_store.session_metadata_source()
            if session is not None:
                target = Screen.LIVE_SESSION if session.mode == SessionMode.LIVE.value else Screen.UPLOAD_CONFIGURE
                self._navigate(target)
        elif name.startswith("upload_voice:"):
            voice = name.partition(":")[2]
            if voice in self._upload_voices:
                self._upload_voices.remove(voice)
            else:
                self._upload_voices.add(voice)
        elif name == "browse":
            if self._path_entry.strip():
                self._start_upload(self._path_entry.strip())
            else:
                Thread(target=self._open_file_dialog, name="harmonix-file-dialog", daemon=True).start()
        elif name == "cancel_upload":
            self.upload_worker.request_cancel()
        elif name == "retry":
            session = self.state_store.ui_projection().session
            if session is not None and session.source_file:
                self._start_upload(session.source_file, retry=True)
        elif name == "open_configure":
            self._navigate(Screen.UPLOAD_CONFIGURE)
        elif name == "open_playback":
            self._navigate(Screen.UPLOAD_PLAYBACK)
        elif name.startswith("key:"):
            self._select_upload_key(name)
        elif name in {"play", "play_again"}:
            self._playback_requests.put(("play",))
        elif name == "pause":
            self._playback_requests.put(("pause",))
        elif name.startswith("voice:"):
            self._toggle_rendered_voice(name.partition(":")[2])
        elif name.startswith("setting:key:"):
            value = DefaultKeyBehavior(name.rsplit(":", 1)[1])
            self._save_setting("default_key_behavior", value)
        elif name.startswith("setting:style:"):
            value = StylePreset(name.rsplit(":", 1)[1])
            self._save_setting("default_style_preset", value)
        elif name == "setting:frame:down":
            current = self.state_store.snapshot().settings.latency_buffer_frame_size
            self._save_setting("latency_buffer_frame_size", max(256, current - 256))
        elif name == "setting:frame:up":
            current = self.state_store.snapshot().settings.latency_buffer_frame_size
            self._save_setting("latency_buffer_frame_size", min(4096, current + 256))
        elif name == "refresh_audio_devices":
            self._refresh_audio_devices()
        elif name.startswith("setting:input_device:"):
            self._select_audio_device("input", name.rsplit(":", 1)[1])
        elif name.startswith("setting:output_device:"):
            self._select_audio_device("output", name.rsplit(":", 1)[1])
        elif name == "setting:reduced_motion":
            current = self.state_store.snapshot().settings.reduced_motion
            self._save_setting("reduced_motion", not current)
        elif name == "settings_retry":
            if self._pending_setting is not None:
                self._save_setting(*self._pending_setting, retry=True)
        elif name == "end_session":
            self._end_session(EndSession())
        elif name == "retry_session_end":
            self._end_session(EndSession(), retry=True)
        elif name.startswith("export_voice:"):
            projection = self.state_store.ui_projection().session
            if projection is not None and not projection.export_in_progress:
                selected = set(projection.export_selected_voices)
                voice = name.partition(":")[2]
                if voice in selected:
                    selected.remove(voice)
                else:
                    selected.add(voice)
                self.state_store.dispatch(SetExportVoices(tuple(v for v in UPLOAD_VOICES if v in selected)))
        elif name == "export_start":
            if not self._export_dialog_pending:
                self._export_dialog_pending = True
                Thread(target=self._open_export_dialog, name="harmonix-export-save-dialog", daemon=True).start()
        elif name == "export_cancel":
            self.export_worker.request_cancel()
        elif name == "export_retry":
            if not self._export_dialog_pending:
                self._export_dialog_pending = True
                Thread(target=self._open_export_dialog, name="harmonix-export-save-dialog", daemon=True).start()
        elif name == "discard_session":
            self._discard_confirmation_pending = True
        elif name == "discard_confirm":
            self._discard_confirmation_pending = False
            self._clear_session_to_home()
        elif name == "discard_cancel":
            self._discard_confirmation_pending = False
        elif name == "close_export_confirm":
            self._close_confirmation_pending = False
            self._close_after_export = True
            self.export_worker.request_cancel()
            self._poll_close_after_export()
        elif name == "close_export_cancel":
            self._close_confirmation_pending = False
        elif name.startswith("live_key:"):
            self._activate_live_key(name)
        elif name == "show_export_folder":
            self._show_export_folder()
        elif name.startswith("recent:"):
            self._restore_recent_session(name.partition(":")[2])

    def _session_metadata(self) -> RecentSessionMetadata | None:
        source = self.state_store.session_metadata_source()
        if source is None:
            return None
        timestamp = datetime.now(timezone.utc).isoformat()
        mode = source.mode
        source_file = source.source_file
        display_name = Path(source_file).name if mode == SessionMode.UPLOAD.value and source_file else f"Live session — {timestamp}"
        toggles = source.voice_toggles
        voices = tuple(voice for voice, enabled in toggles if enabled)
        return RecentSessionMetadata(
            source_file=source_file,
            key_root=source.key_root,
            key_scale_type=source.key_scale_type,
            voices=voices,
            style_preset=source.style_preset,
            last_used_at=timestamp,
            id=uuid4().hex,
            display_name=display_name,
            mode=mode,
            voice_toggles=toggles,
        )

    def _end_session(self, command: EndSession, *, retry: bool = False) -> None:
        del command  # EndSession is an application intent, not a reducer-side I/O command.
        if retry:
            if self._pending_session_end is None:
                return
            metadata, candidate = self._pending_session_end
        else:
            if self.state_store.session_metadata_source() is None:
                return
            metadata = self._session_metadata()
            if metadata is None:
                return
            candidate = [*self._recent_metadata, metadata]
            self._pending_session_end = (metadata, candidate)
        try:
            saved = self.persistence_store.save_recent_sessions(candidate)
            if not any(item.id == metadata.id for item in saved):
                raise PersistenceWriteError("session metadata was not retained by the recent-session store")
        except PersistenceError as error:
            self._session_end_error = str(error)
            return
        self._recent_metadata = saved
        self.state_store.dispatch(ReplaceRecentSessions(tuple(item.to_summary() for item in saved)))
        current = self.state_store.snapshot().session
        if current is not None and current.mode is SessionMode.UPLOAD:
            selected = tuple(
                voice
                for voice, enabled in current.voices.active.items()
                if enabled and current.recorded_buffers.a_per_voice.get(voice) is not None
            )
            self.state_store.dispatch(SetExportVoices(selected))
            self.state_store.dispatch(Navigate(Screen.REVIEW_EXPORT))
        elif current is not None and current.mode is SessionMode.LIVE and self.live_coordinator is not None and self.live_coordinator.is_alive:
            self._stop_live()
        else:
            self.state_store.dispatch(ClearSession())
            self.state_store.dispatch(Navigate(Screen.HOME))
        self._pending_session_end = None
        self._session_end_error = None

    def _clear_session_to_home(self) -> None:
        self.state_store.dispatch(ClearSession())
        self.state_store.dispatch(Navigate(Screen.HOME))
        self._controller_source_identity = None
        self._waveform_source = None
        self._waveform = ()

    def _open_export_dialog(self) -> None:
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            try:
                path = filedialog.asksaveasfilename(
                    defaultextension=".wav",
                    filetypes=(("WAV audio", "*.wav"),),
                )
            finally:
                root.destroy()
            self._runtime_events.put(("export_destination", path))
        except Exception as error:
            self._runtime_events.put(("export_dialog_error", str(error)))

    def _start_export(self, destination: str) -> None:
        if not destination:
            return
        try:
            projection = self.state_store.ui_projection().session
            if projection is None:
                return
            self.export_worker.start(destination, enabled_voices=projection.export_selected_voices)
            self._export_completion_handled_request = None
            self._export_toast = None
            self._export_ui_error = None
        except Exception as error:
            self._export_ui_error = str(error)

    def _finish_export_if_complete(self) -> None:
        projection = self.state_store.ui_projection()
        session = projection.session
        if session is None or session.export_status != "completed":
            return
        path = session.export_last_export_path
        self._export_toast_path = path
        self._export_toast = f"Export saved: {Path(path).name}" if path else "Export saved."
        self._clear_session_to_home()

    def _request_close(self) -> None:
        if self.export_worker.is_alive:
            self._close_confirmation_pending = True
            return
        if self.live_coordinator is not None and self.live_coordinator.is_alive:
            self._stop_live()
            if self.live_coordinator.is_alive:
                self._close_confirmation_pending = True
                return
        self._running = False

    def _poll_close_after_export(self) -> None:
        if self._close_after_export and not self.export_worker.is_alive:
            self._close_after_export = False
            self._running = False

    def _show_export_folder(self) -> None:
        path = self._export_toast_path
        if not path:
            return
        try:
            import os

            os.startfile(str(Path(path).parent))
        except Exception as error:
            self._export_toast = f"Export saved; could not open folder: {error}"

    def _restore_recent_session(self, session_id: str) -> None:
        if self._pending_session_end is not None or self.state_store.session_metadata_source() is not None:
            return
        summary = next((item for item in self.state_store.snapshot().recent_sessions if item.id == session_id), None)
        if summary is None or summary.mode is not SessionMode.UPLOAD or not summary.source_file_path:
            return
        if not Path(summary.source_file_path).exists():
            self._upload_start_error = "The recording for this recent session is unavailable."
            return
        try:
            style_preset = StylePreset(summary.style_preset) if summary.style_preset else StylePreset.POP
        except ValueError:
            self._upload_start_error = "This saved style is not supported by the current Upload flow."
            return
        if style_preset is StylePreset.BARBERSHOP:
            self._upload_start_error = "This saved style cannot be restored until its mapping is approved."
            return
        toggles = dict(summary.voice_toggles)
        selected_voices = tuple(voice for voice in UPLOAD_VOICES if toggles.get(voice, False))
        self._upload_voices = set(selected_voices)
        manual_key = None
        if summary.last_used_key_root and summary.last_used_key_scale_type:
            manual_key = KeyScale(summary.last_used_key_root, summary.last_used_key_scale_type)
        self._upload_start_error = None
        self._navigate(Screen.UPLOAD_CONFIGURE)
        try:
            self.upload_worker.start(
                summary.source_file_path,
                selected_voices,
                manual_key=manual_key,
                style_preset=style_preset,
                keep_configure_open=True,
            )
        except Exception as error:
            self._upload_start_error = str(error)

    def _save_setting(self, field: str, value: object, *, retry: bool = False) -> None:
        if self._pending_setting is not None and not retry:
            return
        try:
            if field == "default_style_preset" and value is StylePreset.BARBERSHOP:
                raise ConfigurationError("Barbershop cannot be selected until its voice mapping is approved.")
            validate_setting_update(field, value)
            current = self.state_store.snapshot().settings
            candidate = replace(current, **{field: value})
            self.persistence_store.save_settings(candidate)
            self.state_store.dispatch(UpdateGlobalSetting(field, value))
        except (ConfigurationError, PersistenceError) as error:
            self._settings_error = str(error)
            if isinstance(error, PersistenceError):
                self._pending_setting = (field, value)
            self._settings_warning = self.persistence_store.settings_warning
            return
        self._settings_error = None
        self._settings_warning = self.persistence_store.settings_warning
        self._pending_setting = None
        if field == "audio_input_device":
            self._audio_input_selector = value
        elif field == "audio_output_device":
            self._audio_output_selector = value

    def _refresh_audio_devices(self) -> None:
        try:
            self._audio_input_devices = InputAdapter.list_devices()
            self._audio_output_devices = InputAdapter.list_output_devices()
            self._audio_device_error = None
            self._audio_devices_loaded = True
            for selector, devices, direction in (
                (self._audio_input_selector, self._audio_input_devices, "input"),
                (self._audio_output_selector, self._audio_output_devices, "output"),
            ):
                if selector is not None:
                    try:
                        resolve_audio_device_selector(selector, devices)
                    except Exception as error:
                        self._audio_device_error = f"Saved {direction} selection: {error}"
                        break
        except Exception as error:
            self._audio_device_error = f"Could not enumerate audio devices: {error}"
            self._audio_devices_loaded = False

    def _select_audio_device(self, direction: str, choice: str) -> None:
        field = f"audio_{direction}_device"
        if choice == "default":
            # System default is an explicit user selection, never an automatic fallback.
            self._save_setting(field, None)
            self._audio_device_error = None if self._settings_error is None else self._settings_error
            return
        try:
            candidates = self._audio_input_devices if direction == "input" else self._audio_output_devices
            if choice in {"previous", "next"}:
                if not candidates:
                    raise ConfigurationError(f"No {direction} devices are available; refresh the device list and retry.")
                current_selector = self._audio_input_selector if direction == "input" else self._audio_output_selector
                current = None
                if current_selector is not None:
                    try:
                        current = resolve_audio_device_selector(current_selector, candidates)
                    except (ConfigurationError, DeviceUnavailableError):
                        current = None
                else:
                    current = next((item for item in candidates if item.is_default), None)
                current_position = next((i for i, item in enumerate(candidates) if current is item), -1)
                step = -1 if choice == "previous" else 1
                device = candidates[(current_position + step) % len(candidates)]
            else:
                index = int(choice)
                device = next(item for item in candidates if item.index == index)
            selector = format_audio_device_selector(device.name, device.host_api_name)
            # Validate against the current enumeration before persisting; ambiguous selectors fail closed.
            resolve_audio_device_selector(selector, candidates)
        except Exception as error:
            self._audio_device_error = str(error)
            return
        self._save_setting(field, selector)
        self._audio_device_error = None if self._settings_error is None else self._settings_error

    def _activate_live_key(self, name: str) -> None:
        """Cycle live key root, toggle scale, or revert to AUTO for the Live Configure screen."""
        if name == "live_key:auto":
            self._live_key = None
            return
        # Derive a base root/scale to mutate from the current selection or a sensible default.
        if self._live_key is not None:
            root = self._live_key.root
            scale = self._live_key.scale_type
        else:
            root = "C"
            scale = "major"
        if name == "live_key:scale":
            scale = "minor" if scale == "major" else "major"
        elif name == "live_key:root:prev":
            root = _ROOTS[(_ROOTS.index(root) - 1) % len(_ROOTS)]
        elif name == "live_key:root:next":
            root = _ROOTS[(_ROOTS.index(root) + 1) % len(_ROOTS)]
        self._live_key = KeyScale(root, scale)

    def _get_or_create_live_coordinator(self) -> LiveAudioCoordinator:
        if self.live_coordinator is not None:
            if getattr(self.live_coordinator, "visualizer_bridge", None) is None:
                self.live_coordinator.visualizer_bridge = self.visualizer_bridge
            return self.live_coordinator
        if self.live_coordinator_factory is not None:
            coordinator = self.live_coordinator_factory(self.state_store, self.visualizer_bridge)
            self.live_coordinator = coordinator
            return coordinator
        settings = self.state_store.snapshot().settings
        adapter = InputAdapter(
            input_selector=self._audio_input_selector,
            output_selector=self._audio_output_selector,
            frame_size=settings.latency_buffer_frame_size,
        )
        coordinator = LiveAudioCoordinator(
            adapter,
            visualizer_bridge=self.visualizer_bridge,
        )
        self.live_coordinator = coordinator
        return coordinator

    def _start_live(self) -> None:
        if self._pending_session_end is not None:
            self._live_start_error = "Save or retry ending the current session before starting Live singing."
            return
        if self.state_store.session_metadata_source() is not None:
            self._live_start_error = "End the current session before starting Live singing."
            return
        try:
            self._live_start_error = None
            coordinator = self._get_or_create_live_coordinator()
            request_id = uuid4().hex
            snapshot = self.state_store.snapshot()
            style_preset = snapshot.settings.default_style_preset
            if style_preset is StylePreset.BARBERSHOP:
                raise ConfigurationError("Barbershop is not supported for Live sessions; choose Pop or Choir.")
            selected_voices = tuple(UPLOAD_PRESET_VOICES.get(style_preset, ("3rd", "5th")))
            coordinator.set_active_voices(selected_voices)
            requested_key = None
            if self._live_key is not None:
                # User explicitly selected a key on Live Configure: apply it
                # authoritatively to the resolver before the stream starts.
                coordinator.key_resolver.set_manual_key(self._live_key.root, self._live_key.scale_type)
                coordinator.key_resolver.apply_processing_boundary()
                requested_key = self._live_key
            elif snapshot.settings.default_key_behavior is DefaultKeyBehavior.LAST_USED and snapshot.last_manual_key is not None:
                requested_key = snapshot.last_manual_key
                coordinator.key_resolver.set_manual_key(requested_key.root, requested_key.scale_type)
                coordinator.key_resolver.apply_processing_boundary()
            active_device = coordinator.start(request_id=request_id)
            self._live_request_id = request_id
            self.state_store.dispatch(
                StartLiveSession(
                    request_id,
                    selected_voices=selected_voices,
                    style_preset=style_preset,
                    requested_key=requested_key,
                )
            )
            if getattr(coordinator.input_adapter, "fallback_occurred", False) and active_device is not None:
                try:
                    new_selector = format_audio_device_selector(active_device.name, active_device.host_api_name) if active_device.host_api_name else None
                    self._save_setting("audio_input_device", new_selector)
                except Exception as setting_error:
                    self._settings_error = f"Fallback device activated, but failed to persist setting: {setting_error}"
        except Exception as error:
            self._live_start_error = str(error)

    def _stop_live(self) -> None:
        coordinator = self.live_coordinator
        if coordinator is None:
            return
        stopped = coordinator.stop(timeout=1.0)
        if not stopped:
            stopped = coordinator.shutdown(timeout=2.0)
        if not stopped and coordinator.is_alive:
            self._live_start_error = "Live worker has not terminated cleanly."
            return
        buffers = coordinator.recorded_buffers()
        request_id = self._live_request_id
        self._live_request_id = None
        self.state_store.dispatch(StopLiveSession(buffers, request_id=request_id))

    def _poll_live_telemetry(self) -> None:
        result = self.visualizer_bridge.read_after(self._live_telemetry_sequence)
        if result is None:
            return
        self._live_telemetry_sequence, frame = result
        projection = self.state_store.ui_projection().session
        if projection is None or projection.mode != SessionMode.LIVE.value:
            return
        detected_key = KeyState(root=frame.detected_key.root, scale_type=frame.detected_key.scale_type) if frame.detected_key else None
        effective_key = KeyState(root=frame.effective_key.root, scale_type=frame.effective_key.scale_type, auto_detect=False) if frame.effective_key else None
        self.state_store.dispatch(
            LiveTelemetryUpdated(
                request_id=frame.request_id,
                input_level=frame.input_level,
                current_f0=frame.current_f0,
                confidence=frame.confidence,
                tracking_state=frame.tracking_state,
                in_tune=frame.in_tune,
                detected_key=detected_key,
                effective_key=effective_key,
                physical_latency_ms=frame.physical_latency_ms,
                pipeline_latency_ms=frame.pipeline_latency_ms,
                latency_provenance=frame.latency_provenance,
                mic_connected=frame.mic_connected,
                error_message=frame.error_message,
            )
        )

    def _start_upload(self, path: str, *, retry: bool = False) -> None:
        if not path:
            return
        if self._pending_session_end is not None:
            self._upload_start_error = "Save or retry ending the current session before starting another Upload."
            return
        if self.state_store.session_metadata_source() is not None and not retry:
            self._upload_start_error = "End the current session before starting another Upload."
            return
        try:
            self._upload_start_error = None
            snapshot = self.state_store.snapshot()
            style_preset = snapshot.session.style_preset if retry and snapshot.session is not None else snapshot.settings.default_style_preset
            if style_preset is StylePreset.BARBERSHOP:
                raise ConfigurationError("Choose Pop or Choir before starting a new Upload session; Barbershop is not defined yet.")
            if not retry and self.state_store.ui_projection().current_screen != Screen.UPLOAD_SELECT.value:
                self._upload_voices = set(UPLOAD_PRESET_VOICES[style_preset])
            manual_key = None
            if snapshot.settings.default_key_behavior is DefaultKeyBehavior.LAST_USED and snapshot.last_manual_key is not None:
                manual_key = KeyScale(snapshot.last_manual_key.root, snapshot.last_manual_key.scale_type)
            continuation_request_id = (
                snapshot.session.upload_analysis.request_id
                if retry and snapshot.session is not None
                else None
            )
            self.upload_worker.start(
                path,
                tuple(voice for voice in UPLOAD_VOICES if voice in self._upload_voices),
                manual_key=manual_key,
                style_preset=style_preset,
                **({"continuation_request_id": continuation_request_id} if continuation_request_id else {}),
            )
            self._path_entry = ""
        except Exception as error:
            self._upload_start_error = str(error)

    def _select_upload_key(self, action: str) -> None:
        if self._pending_session_end is not None:
            return
        session = self.state_store.snapshot().session
        if session is None or not session.source_file or not session.key.root:
            return
        continuation_request_id = session.upload_analysis.request_id
        if not continuation_request_id:
            self._upload_start_error = "The active Upload request is unavailable; key rendering was not started."
            return
        base_key = session.pending_key or session.key
        root = base_key.root
        if root is None:
            return
        scale = base_key.scale_type or "major"
        if action == "key:scale":
            scale = "minor" if scale == "major" else "major"
        else:
            direction = -1 if action == "key:root:prev" else 1
            root = _ROOTS[(_ROOTS.index(root) + direction) % len(_ROOTS)]
        selected = KeyState(root=root, scale_type=scale, auto_detect=False)
        self._playback_requests.put(("close",))
        self._controller_source_identity = None
        self.state_store.dispatch(SelectUploadKey(selected))
        try:
            self.upload_worker.start(
                session.source_file,
                tuple(voice for voice, enabled in session.voices.active.items() if enabled),
                manual_key=KeyScale(root, scale),
                style_preset=session.style_preset,
                keep_configure_open=True,
                continuation_request_id=continuation_request_id,
            )
        except Exception as error:
            self._upload_start_error = str(error)

    def _open_file_dialog(self) -> None:
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            try:
                path = filedialog.askopenfilename(
                    filetypes=(("Audio files", "*.wav *.mp3 *.flac"), ("All files", "*.*"))
                )
            finally:
                root.destroy()
            if path:
                self._runtime_events.put(("file_selected", path))
        except Exception as error:
            self._runtime_events.put(("upload_start_error", str(error)))

    def _toggle_rendered_voice(self, voice: str) -> None:
        if self._pending_session_end is not None:
            return
        projection = self.state_store.ui_projection().session
        if projection is None or voice not in projection.available_voices:
            return
        active = dict(projection.voices)
        enabled = {name for name in projection.available_voices if active.get(name, False)}
        if voice in enabled:
            enabled.remove(voice)
        else:
            enabled.add(voice)
        self._playback_requests.put(("voices", tuple(sorted(enabled))))

    def _seek_at(self, mouse_x: int) -> None:
        rect = self._layout.controls.get("seek")
        session = self.state_store.ui_projection().session
        if rect is None or session is None or session.playback_duration_seconds <= 0:
            return
        fraction = min(1.0, max(0.0, (mouse_x - rect.left) / max(1, rect.width)))
        self._playback_requests.put(("seek", fraction * session.playback_duration_seconds))

    def _control_at(self, position: tuple[int, int]) -> str | None:
        for name, rect in self._layout.controls.items():
            if name in self._layout.enabled and rect.collidepoint(position):
                return name
        return None

    def _cycle_focus(self, direction: int) -> None:
        enabled = tuple(name for name in self._layout.controls if name in self._layout.enabled)
        if not enabled:
            return
        try:
            index = enabled.index(self._focused)
        except ValueError:
            index = -1
        self._focused = enabled[(index + direction) % len(enabled)]

    def _ensure_playback_source(self) -> None:
        source = self.state_store.playback_source()
        if source is None:
            return
        identity = id(source.recorded_buffers.original)
        if self._waveform_source is not source.recorded_buffers.original:
            self._waveform_source = source.recorded_buffers.original
            self._waveform = ()
            self._waveform_generation += 1
            generation = self._waveform_generation
            audio = source.recorded_buffers.original
            Thread(target=self._compute_waveform, args=(generation, audio), name="harmonix-waveform", daemon=True).start()
        if identity != getattr(self, "_controller_source_identity", None):
            self._controller_source_identity = identity
            self._playback_requests.put(("create", source))

    def _compute_waveform(self, generation: int, audio) -> None:
        waveform = downsample_waveform(audio)
        self._runtime_events.put(("waveform", (generation, waveform)))

    def _collect_runtime_events(self) -> None:
        while True:
            try:
                event, value = self._runtime_events.get_nowait()
            except Empty:
                return
            if event == "waveform":
                generation, waveform = value
                if generation == self._waveform_generation:
                    self._waveform = waveform
            elif event == "upload_start_error":
                self._upload_start_error = str(value)
            elif event == "file_selected":
                self._start_upload(str(value))
            elif event == "export_destination":
                self._export_dialog_pending = False
                self._start_export(str(value or ""))
            elif event == "export_dialog_error":
                self._export_dialog_pending = False
                self._export_ui_error = str(value)

    def _playback_runtime(self) -> None:
        controller = None
        while not self._runtime_stop.is_set():
            try:
                request = self._playback_requests.get(timeout=0.01)
            except Empty:
                request = None
            if request:
                action, *arguments = request
                if action == "shutdown":
                    break
                if action == "create":
                    source = arguments[0]
                    try:
                        if controller is not None:
                            controller.close()
                        controller = self.playback_controller_factory(
                            self.state_store,
                            source.recorded_buffers,
                            source.rendered_voices,
                            initially_enabled=source.initially_enabled,
                        )
                    except Exception as error:
                        self.state_store.publish_playback_event(
                            PlaybackFailed(type(error).__name__, str(error), 0.0)
                        )
                elif controller is not None:
                    try:
                        if action == "play":
                            controller.play()
                        elif action == "pause":
                            controller.pause()
                        elif action == "seek":
                            controller.seek(arguments[0])
                        elif action == "voices":
                            controller.set_voices(arguments[0])
                        elif action == "close":
                            controller.close()
                            controller = None
                    except Exception as error:
                        self.state_store.publish_playback_event(
                            PlaybackFailed(type(error).__name__, str(error), controller.engine.position_seconds)
                        )
            if controller is not None:
                controller.process_events()
        if controller is not None:
            controller.close()
