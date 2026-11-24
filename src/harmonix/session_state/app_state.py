"""Thread-safe state store with one command application boundary."""

from copy import deepcopy
from dataclasses import dataclass
from threading import RLock

from ..domain.enums import SessionMode
from ..domain.models import AppState, RecordedBuffers, RecentSessionSummary
from .commands import (
    CancelExport,
    CancelUploadAnalysis,
    ClearSession,
    CompleteExport,
    CompleteUploadAnalysis,
    FailExport,
    FailUploadAnalysis,
    Navigate,
    PlaybackEngineUpdate,
    PlaybackFailed,
    PlaybackPause,
    PlaybackPlay,
    PlaybackRecovered,
    PlaybackSeek,
    PlaybackSeekApplied,
    PlaybackSetVoices,
    RequestExportCancellation,
    RequestUploadCancellation,
    ReplaceRecentSessions,
    ReplaceGlobalSettings,
    StartExport,
    SetExportVoices,
    StartUploadAnalysis,
    SelectUploadKey,
    UpdateExportProgress,
    UpdateGlobalSetting,
    UpdateUploadAnalysisProgress,
    LiveTelemetryUpdated,
)
from .reducers import reduce_state


@dataclass(frozen=True)
class PlaybackSource:
    recorded_buffers: RecordedBuffers
    rendered_voices: tuple[str, ...]
    initially_enabled: tuple[str, ...]


@dataclass(frozen=True)
class ExportSource:
    recorded_buffers: RecordedBuffers
    active_voices: tuple[str, ...]


@dataclass(frozen=True)
class SessionMetadataSource:
    mode: str
    source_file: str | None
    key_root: str | None
    key_scale_type: str | None
    voice_toggles: tuple[tuple[str, bool], ...]
    style_preset: str


@dataclass(frozen=True)
class SessionUIProjection:
    mode: str
    source_file: str | None
    key_root: str | None
    key_scale_type: str | None
    voices: tuple[tuple[str, bool], ...]
    available_voices: tuple[str, ...]
    upload_status: str
    upload_phase: str | None
    upload_completed_units: int
    upload_total_units: int | None
    upload_cancellation_requested: bool
    upload_error_code: str | None
    upload_error_message: str | None
    playback_is_playing: bool
    playback_position_seconds: float
    playback_duration_seconds: float
    playback_pending_seek_seconds: float | None
    playback_ended: bool
    playback_error_code: str | None
    playback_error_message: str | None
    has_recorded_audio: bool
    export_in_progress: bool
    export_status: str
    export_phase: str | None
    export_completed_units: int
    export_total_units: int | None
    export_progress_pct: float | None
    export_error: str | None
    export_last_export_path: str | None
    export_selected_voices: tuple[str, ...] = ()
    detected_key_root: str | None = None
    detected_key_scale_type: str | None = None
    pending_key_root: str | None = None
    pending_key_scale_type: str | None = None
    mic_connected: bool = False
    input_level: float = 0.0
    current_f0: float | None = None
    tracking_confidence: float = 0.0
    tracking_state: str = "idle"
    in_tune: bool = False
    audio_error_message: str | None = None
    physical_latency_ms: float | None = None
    pipeline_latency_ms: float | None = None
    latency_provenance: str = "unavailable"


@dataclass(frozen=True)
class AppUIProjection:
    current_screen: str
    recent_sessions: tuple[RecentSessionSummary, ...]
    reduced_motion: bool
    default_key_behavior: str
    default_style_preset: str
    latency_buffer_frame_size: int
    session: SessionUIProjection | None


class StateStore:
    """Own the canonical AppState and serialize foundational commands."""

    def __init__(self, initial_state: AppState | None = None) -> None:
        self._state = initial_state or AppState()
        self._lock = RLock()

    def snapshot(self) -> AppState:
        with self._lock:
            return deepcopy(self._state)

    def ui_projection(self) -> AppUIProjection:
        """Return only small render metadata; large audio arrays are never copied."""
        with self._lock:
            state = self._state
            session = state.session
            projected_session = None
            if session is not None:
                playback = session.playback
                upload = session.upload_analysis
                export = session.export
                projected_session = SessionUIProjection(
                    mode=session.mode.value,
                    source_file=session.source_file,
                    key_root=session.key.root,
                    key_scale_type=session.key.scale_type,
                    detected_key_root=session.detected_key.root,
                    detected_key_scale_type=session.detected_key.scale_type,
                    pending_key_root=session.pending_key.root if session.pending_key else None,
                    pending_key_scale_type=session.pending_key.scale_type if session.pending_key else None,
                    voices=tuple(sorted(session.voices.active.items())),
                    available_voices=tuple(playback.available_voices),
                    upload_status=upload.status,
                    upload_phase=upload.phase,
                    upload_completed_units=upload.completed_units,
                    upload_total_units=upload.total_units,
                    upload_cancellation_requested=upload.cancellation_requested,
                    upload_error_code=upload.error_code,
                    upload_error_message=upload.error_message,
                    playback_is_playing=playback.is_playing,
                    playback_position_seconds=playback.position_seconds,
                    playback_duration_seconds=playback.duration_seconds,
                    playback_pending_seek_seconds=playback.pending_seek_seconds,
                    playback_ended=playback.ended,
                    playback_error_code=playback.error_code,
                    playback_error_message=playback.error_message,
                    has_recorded_audio=session.recorded_buffers.original is not None,
                    export_in_progress=export.in_progress,
                    export_status=export.status,
                    export_phase=export.phase,
                    export_completed_units=export.completed_units,
                    export_total_units=export.total_units,
                    export_progress_pct=export.progress_pct,
                    export_error=export.error,
                    export_last_export_path=export.last_export_path,
                    export_selected_voices=export.selected_voices,
                    mic_connected=session.audio_status.mic_connected,
                    input_level=session.audio_status.input_level,
                    current_f0=session.audio_status.current_f0,
                    tracking_confidence=session.audio_status.tracking_confidence,
                    tracking_state=session.audio_status.tracking_state,
                    in_tune=session.audio_status.in_tune,
                    audio_error_message=session.audio_status.error_message,
                    physical_latency_ms=session.latency.current_ms,
                    pipeline_latency_ms=session.latency.pipeline_ms,
                    latency_provenance=session.latency.provenance,
                )
            return AppUIProjection(
                current_screen=state.current_screen.value,
                recent_sessions=tuple(state.recent_sessions),
                reduced_motion=state.settings.reduced_motion,
                default_key_behavior=state.settings.default_key_behavior.value,
                default_style_preset=state.settings.default_style_preset.value,
                latency_buffer_frame_size=state.settings.latency_buffer_frame_size,
                session=projected_session,
            )

    def playback_source(self) -> PlaybackSource | None:
        """Expose completed immutable-by-contract buffers without copying their arrays."""
        with self._lock:
            session = self._state.session
            if session is None or session.mode is not SessionMode.UPLOAD:
                return None
            buffers = session.recorded_buffers
            if buffers.original is None or buffers.sample_rate is None:
                return None
            rendered = tuple(session.playback.available_voices)
            enabled = tuple(voice for voice, active in session.voices.active.items() if active and voice in rendered)
            return PlaybackSource(buffers, rendered, enabled)

    def export_source(self) -> ExportSource | None:
        """Expose completed recording buffers by reference for background export."""
        with self._lock:
            session = self._state.session
            if (
                session is None
                or session.recorded_buffers.original is None
                or session.recorded_buffers.sample_rate is None
            ):
                return None
            active = tuple(
                voice
                for voice, is_active in session.voices.active.items()
                if is_active and session.recorded_buffers.a_per_voice.get(voice) is not None
            )
            return ExportSource(session.recorded_buffers, active)

    def session_metadata_source(self) -> SessionMetadataSource | None:
        """Read only session metadata under lock, without copying recorded audio buffers."""
        with self._lock:
            session = self._state.session
            if session is None:
                return None
            key = session.pending_key or session.key
            return SessionMetadataSource(
                mode=session.mode.value,
                source_file=session.source_file,
                key_root=key.root,
                key_scale_type=key.scale_type,
                voice_toggles=tuple(sorted((voice, bool(enabled)) for voice, enabled in session.voices.active.items())),
                style_preset=session.style_preset.value,
            )

    def dispatch(
        self,
        command: Navigate
        | ClearSession
        | ReplaceRecentSessions
        | ReplaceGlobalSettings
        | UpdateGlobalSetting
        | LiveTelemetryUpdated
        | StartUploadAnalysis
        | SelectUploadKey
        | UpdateUploadAnalysisProgress
        | RequestUploadCancellation
        | CompleteUploadAnalysis
        | CancelUploadAnalysis
        | FailUploadAnalysis
        | StartExport
        | SetExportVoices
        | UpdateExportProgress
        | RequestExportCancellation
        | CancelExport
        | CompleteExport
        | FailExport
        | PlaybackPlay
        | PlaybackPause
        | PlaybackSeek
        | PlaybackSetVoices
        | PlaybackEngineUpdate
        | PlaybackSeekApplied
        | PlaybackFailed
        | PlaybackRecovered,
    ) -> AppState:
        with self._lock:
            self._state = reduce_state(self._state, command)
            return deepcopy(self._state)

    def publish_export_event(
        self,
        command: StartExport
        | UpdateExportProgress
        | RequestExportCancellation
        | CancelExport
        | CompleteExport
        | FailExport,
    ) -> None:
        """Apply export worker events without deep-copying recorded audio buffers."""
        with self._lock:
            self._state = reduce_state(self._state, command)

    def publish_playback_event(
        self,
        command: PlaybackEngineUpdate | PlaybackSeekApplied | PlaybackFailed | PlaybackRecovered,
    ) -> None:
        """Apply engine telemetry without copying large recorded audio buffers."""
        with self._lock:
            self._state = reduce_state(self._state, command)

    def dispatch_playback_intent(self, command: PlaybackPlay | PlaybackPause | PlaybackSeek | PlaybackSetVoices) -> None:
        """Apply a playback intent without copying large immutable audio buffers."""
        with self._lock:
            self._state = reduce_state(self._state, command)


def make_default_capture_state() -> dict[str, object]:
    """Expose the M3 shared input-device state contract without duplicating authoritative state."""
    return {
        "selected_input_device": None,
        "input_device_error": None,
        "input_capture_state": "idle",
        "input_retry_count": 0,
        "mic_connected": False,
    }
