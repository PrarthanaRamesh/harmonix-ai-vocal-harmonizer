"""Pure foundational state transitions."""

from dataclasses import replace
import math

from ..config.validation import validate_setting_update
from ..domain.errors import StateError
from ..domain.enums import Screen, SessionMode
from ..domain.models import (
    AppState,
    ExportState,
    KeyState,
    PlaybackState,
    RecordedBuffers,
    SessionState,
    UploadAnalysisState,
    VoiceState,
)
from .commands import (
    CancelUploadAnalysis,
    CancelExport,
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
    StartLiveSession,
    StopLiveSession,
    StartUploadAnalysis,
    SelectUploadKey,
    UpdateExportProgress,
    UpdateGlobalSetting,
    LiveTelemetryUpdated,
    UpdateUploadAnalysisProgress,
)


def reduce_state(
    state: AppState,
    command: Navigate
    | ClearSession
    | ReplaceRecentSessions
    | ReplaceGlobalSettings
    | UpdateGlobalSetting
    | LiveTelemetryUpdated
    | StartLiveSession
    | StopLiveSession
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
    """Return a new state for a command without implementing later behavior."""
    if isinstance(command, Navigate):
        return replace(state, current_screen=command.screen)
    if isinstance(command, ClearSession):
        return replace(state, session=None)
    if isinstance(command, ReplaceRecentSessions):
        return replace(state, recent_sessions=list(command.sessions))
    if isinstance(command, ReplaceGlobalSettings):
        return replace(state, settings=command.settings)
    if isinstance(command, UpdateGlobalSetting):
        validate_setting_update(command.field, command.value)
        return replace(state, settings=replace(state.settings, **{command.field: command.value}))
    if isinstance(command, LiveTelemetryUpdated):
        session = state.session
        if (
            session is None
            or session.mode is not SessionMode.LIVE
            or not command.request_id
            or command.request_id != session.live_request_id
        ):
            return state
        values = (command.input_level, command.confidence)
        optional_values = (command.current_f0, command.physical_latency_ms, command.pipeline_latency_ms)
        if not all(math.isfinite(value) for value in values) or not all(
            value is None or math.isfinite(value) for value in optional_values
        ):
            raise StateError("Live telemetry values must be finite")
        if not 0.0 <= command.confidence <= 1.0:
            raise StateError("Live confidence must be between zero and one")
        return replace(
            state,
            session=replace(
                session,
                key=command.effective_key or session.key,
                detected_key=command.detected_key or session.detected_key,
                audio_status=replace(
                    session.audio_status,
                    mic_connected=command.mic_connected,
                    input_level=max(0.0, command.input_level),
                    current_f0=command.current_f0,
                    tracking_confidence=command.confidence,
                    tracking_state=command.tracking_state,
                    in_tune=command.in_tune,
                    error_message=command.error_message,
                ),
                latency=replace(
                    session.latency,
                    current_ms=command.physical_latency_ms,
                    pipeline_ms=command.pipeline_latency_ms,
                    provenance=command.latency_provenance,
                ),
            ),
        )
    if isinstance(command, StartLiveSession):
        if not command.request_id:
            raise StateError("live session requires a request id")
        session = state.session
        if session is not None and session.mode is SessionMode.UPLOAD:
            raise StateError("cannot start live session while upload session is active")
        voices = VoiceState(active={voice: voice in command.selected_voices for voice in ("3rd", "5th", "octave")})
        new_session = SessionState(
            mode=SessionMode.LIVE,
            live_request_id=command.request_id,
            style_preset=command.style_preset,
            voices=voices,
            key=command.requested_key or KeyState(auto_detect=True),
            playback=PlaybackState(is_playing=False, is_recording=True),
        )
        return replace(state, current_screen=Screen.LIVE_SESSION, session=new_session)
    if isinstance(command, StopLiveSession):
        session = state.session
        if session is None or session.mode is not SessionMode.LIVE:
            raise StateError("stopping a live session requires an active live session")
        if command.request_id and session.live_request_id and command.request_id != session.live_request_id:
            return state
        buffers = command.recorded_buffers
        available_voices = tuple(
            voice
            for voice in ("3rd", "5th", "octave")
            if buffers is not None and isinstance(getattr(buffers, "a_per_voice", None), dict) and buffers.a_per_voice.get(voice) is not None
        )
        selected_voices = tuple(
            voice
            for voice in available_voices
            if session.voices.active.get(voice, False)
        )
        return replace(
            state,
            current_screen=Screen.REVIEW_EXPORT,
            session=replace(
                session,
                recorded_buffers=buffers if buffers is not None else RecordedBuffers(),
                playback=replace(
                    session.playback,
                    available_voices=available_voices,
                    is_playing=False,
                    is_recording=False,
                ),
                export=replace(
                    session.export,
                    selected_voices=selected_voices,
                    status="idle",
                ),
            ),
        )
    if isinstance(command, SelectUploadKey):
        if not command.key.root or not command.key.scale_type:
            raise StateError("manual upload key requires a root and scale")
        return replace(state, last_manual_key=command.key)
    if isinstance(command, StartExport):
        if not command.request_id:
            raise StateError("export requires a request id")
        session = state.session
        if session is None:
            raise StateError("export requires an active session")
        if session.export.in_progress:
            raise StateError("an export is already running")
        available = session.recorded_buffers.a_per_voice
        if any(voice not in available or available[voice] is None for voice in session.export.selected_voices):
            raise StateError("export selection contains an unavailable voice")
        return replace(
            state,
            session=replace(
                session,
                export=ExportState(
                    in_progress=True,
                    request_id=command.request_id,
                    status="running",
                    selected_voices=session.export.selected_voices,
                ),
            ),
        )
    if isinstance(command, SetExportVoices):
        session = state.session
        if session is None:
            raise StateError("export voice selection requires an active session")
        if session.mode is SessionMode.LIVE and session.playback.is_recording:
            raise StateError("export voice selection cannot change during live recording")
        if session.export.in_progress:
            raise StateError("export voice selection cannot change during export")
        voices = tuple(dict.fromkeys(command.voices))
        available = session.recorded_buffers.a_per_voice
        if any(voice not in {"3rd", "5th", "octave"} for voice in voices):
            raise StateError("export selection contains an unsupported voice")
        if any(voice not in available or available[voice] is None for voice in voices):
            raise StateError("export selection contains an unavailable voice")
        return replace(state, session=replace(session, export=replace(session.export, selected_voices=voices)))
    if isinstance(
        command,
        (UpdateExportProgress, RequestExportCancellation, CancelExport, CompleteExport, FailExport),
    ):
        session = state.session
        if session is None or command.request_id != session.export.request_id:
            return state
        export_state = session.export
        if isinstance(command, UpdateExportProgress):
            if export_state.status != "running" or export_state.cancellation_requested:
                return state
            if command.phase == "rendering":
                if (
                    command.completed_units is None
                    or command.total_units is None
                    or command.total_units <= 0
                ):
                    raise StateError("rendering progress requires a positive total")
                completed = max(0, min(command.completed_units, command.total_units))
                if export_state.phase == "rendering":
                    completed = max(export_state.completed_units, completed)
                completed = min(completed, command.total_units)
                progress_pct = 100.0 * completed / command.total_units
                return replace(
                    state,
                    session=replace(
                        session,
                        export=replace(
                            export_state,
                            phase="rendering",
                            completed_units=completed,
                            total_units=command.total_units,
                            progress_pct=progress_pct,
                        ),
                    ),
                )
            if command.phase == "writing":
                if command.completed_units is not None or command.total_units is not None:
                    raise StateError("writing progress must not report a percentage")
                return replace(
                    state,
                    session=replace(
                        session,
                        export=replace(
                            export_state,
                            phase="writing",
                            progress_pct=None,
                        ),
                    ),
                )
            raise StateError(f"unsupported export progress phase: {command.phase}")
        if isinstance(command, RequestExportCancellation):
            if export_state.status != "running":
                return state
            return replace(
                state,
                session=replace(
                    session,
                    export=replace(
                        export_state,
                        status="cancelling",
                        cancellation_requested=True,
                        progress_pct=None,
                    ),
                ),
            )
        if isinstance(command, CancelExport):
            if export_state.status not in {"running", "cancelling"}:
                return state
            return replace(
                state,
                session=replace(
                    session,
                    export=replace(
                        export_state,
                        in_progress=False,
                        status="cancelled",
                        phase=None,
                        progress_pct=None,
                        cancellation_requested=True,
                    ),
                ),
            )
        if isinstance(command, CompleteExport):
            if export_state.status not in {"running", "cancelling"}:
                return state
            return replace(
                state,
                session=replace(
                    session,
                    export=replace(
                        export_state,
                        in_progress=False,
                        status="completed",
                        phase=None,
                        completed_units=(
                            export_state.total_units
                            if export_state.total_units is not None
                            else export_state.completed_units
                        ),
                        progress_pct=100.0,
                        error=None,
                        last_export_path=command.output_path,
                        cancellation_requested=False,
                    ),
                ),
            )
        if isinstance(command, FailExport):
            if export_state.status not in {"running", "cancelling"}:
                return state
            return replace(
                state,
                session=replace(
                    session,
                    export=replace(
                        export_state,
                        in_progress=False,
                        status="failed",
                        phase=None,
                        progress_pct=None,
                        error=f"{command.error_code}: {command.error_message}",
                        cancellation_requested=False,
                    ),
                ),
            )
    if isinstance(command, StartUploadAnalysis):
        if not command.request_id or not command.source_file:
            raise StateError("upload analysis requires a request id and source file")
        if any(voice not in {"3rd", "5th", "octave"} for voice in command.selected_voices):
            raise StateError("upload analysis contains an unsupported voice")
        session = state.session
        if session is None:
            if command.continuation_request_id is not None:
                raise StateError("upload continuation token has no active session")
            session = SessionState(mode=SessionMode.UPLOAD)
        else:
            is_unstarted_upload = (
                session.mode is SessionMode.UPLOAD
                and session.source_file is None
                and not session.upload_analysis.request_id
                and session.recorded_buffers.original is None
            )
            if is_unstarted_upload:
                if command.continuation_request_id is not None:
                    raise StateError("upload continuation token has no active request")
            elif (
                session.mode is not SessionMode.UPLOAD
                or not command.continuation_request_id
                or command.continuation_request_id != session.upload_analysis.request_id
                or command.source_file != session.source_file
            ):
                raise StateError("starting upload analysis would replace an active session")
        voices = VoiceState(active={voice: voice in command.selected_voices for voice in ("3rd", "5th", "octave")})
        session = replace(
            session,
            source_file=command.source_file,
            style_preset=command.style_preset,
            pending_key=command.requested_key,
            voices=voices,
            playback=replace(
                session.playback,
                is_playing=False,
                position_seconds=0.0,
                duration_seconds=0.0,
                pending_seek_seconds=None,
                pending_seek_request_id=None,
                available_voices=(),
                ended=False,
                error_code=None,
                error_message=None,
            ),
            recorded_buffers=RecordedBuffers(
                original=None,
                a_per_voice={voice: None for voice in ("3rd", "5th", "octave")},
                sample_rate=None,
            ),
            upload_analysis=UploadAnalysisState(request_id=command.request_id, status="running"),
        )
        screen = (
            state.current_screen
            if command.keep_configure_open and state.current_screen is Screen.UPLOAD_CONFIGURE
            else Screen.UPLOAD_ANALYZING
        )
        return replace(state, current_screen=screen, session=session)

    if isinstance(command, (UpdateUploadAnalysisProgress, RequestUploadCancellation, CompleteUploadAnalysis, CancelUploadAnalysis, FailUploadAnalysis)):
        session = state.session
        if session is None or session.mode is not SessionMode.UPLOAD:
            return state
        upload_state = session.upload_analysis
        if command.request_id != upload_state.request_id:
            return state
        if isinstance(command, UpdateUploadAnalysisProgress):
            if upload_state.status != "running" or upload_state.cancellation_requested:
                return state
            completed = max(0, command.completed_units)
            if command.phase == upload_state.phase:
                completed = max(upload_state.completed_units, completed)
            total = command.total_units
            return replace(
                state,
                session=replace(
                    session,
                    upload_analysis=replace(
                        upload_state,
                        phase=command.phase,
                        completed_units=completed,
                        total_units=total,
                    ),
                ),
            )
        if isinstance(command, RequestUploadCancellation):
            if upload_state.status != "running":
                return state
            return replace(
                state,
                session=replace(
                    session,
                    upload_analysis=replace(upload_state, status="cancelling", cancellation_requested=True),
                ),
            )
        if isinstance(command, CancelUploadAnalysis):
            if not upload_state.cancellation_requested:
                return state
            return replace(
                state,
                session=replace(
                    session,
                    upload_analysis=replace(upload_state, status="cancelled", phase=None),
                ),
            )
        if isinstance(command, FailUploadAnalysis):
            if upload_state.status != "running" or upload_state.cancellation_requested:
                return state
            return replace(
                state,
                session=replace(
                    session,
                    upload_analysis=replace(
                        upload_state,
                        status="failed",
                        phase=None,
                        error_code=command.error_code,
                        error_message=command.error_message,
                    ),
                ),
            )
        if isinstance(command, CompleteUploadAnalysis):
            if upload_state.status != "running" or upload_state.cancellation_requested:
                return state
            session = replace(
                session,
                key=command.key,
                detected_key=command.detected_key or command.key,
                pending_key=None,
                recorded_buffers=command.recorded_buffers,
                playback=replace(
                    session.playback,
                    duration_seconds=command.duration_seconds,
                    position_seconds=0.0,
                    available_voices=tuple(
                        voice
                        for voice, enabled in session.voices.active.items()
                        if enabled and voice in command.recorded_buffers.a_per_voice
                    ),
                    ended=False,
                    error_code=None,
                    error_message=None,
                ),
                upload_analysis=replace(
                    upload_state,
                    status="completed",
                    phase=None,
                    cancellation_requested=False,
                    error_code=None,
                    error_message=None,
                ),
            )
            return replace(state, current_screen=Screen.UPLOAD_CONFIGURE, session=session)
    if isinstance(command, (PlaybackPlay, PlaybackPause, PlaybackSeek, PlaybackSetVoices)):
        session = state.session
        if session is None or session.mode is not SessionMode.UPLOAD:
            raise StateError("playback command requires an active upload session")
        if isinstance(command, PlaybackSeek):
            if not command.request_id or not math.isfinite(command.seek_seconds):
                raise StateError("seek requires a request id and finite position")
            return replace(
                state,
                session=replace(
                    session,
                    playback=replace(
                        session.playback,
                        pending_seek_seconds=command.seek_seconds,
                        pending_seek_request_id=command.request_id,
                    ),
                ),
            )
        if isinstance(command, PlaybackSetVoices):
            enabled = set(command.enabled_voices)
            available = set(session.playback.available_voices)
            if enabled.difference(available):
                raise StateError("playback requested a voice that was not rendered")
            active = {voice: voice in enabled for voice in ("3rd", "5th", "octave")}
            return replace(state, session=replace(session, voices=replace(session.voices, active=active)))
        return state
    if isinstance(command, PlaybackSeekApplied):
        session = state.session
        if session is None or session.mode is not SessionMode.UPLOAD:
            return state
        playback = session.playback
        if command.request_id != playback.pending_seek_request_id:
            return state
        return replace(
            state,
            session=replace(
                session,
                playback=replace(
                    playback,
                    position_seconds=command.position_seconds,
                    is_playing=command.is_playing,
                    ended=command.ended,
                    pending_seek_seconds=None,
                    pending_seek_request_id=None,
                ),
            ),
        )
    if isinstance(command, PlaybackEngineUpdate):
        session = state.session
        if session is None or session.mode is not SessionMode.UPLOAD:
            return state
        if session.playback.pending_seek_request_id is not None:
            return state
        return replace(
            state,
            session=replace(
                session,
                playback=replace(
                    session.playback,
                    position_seconds=command.position_seconds,
                    is_playing=command.is_playing,
                    ended=command.ended,
                ),
            ),
        )
    if isinstance(command, PlaybackFailed):
        session = state.session
        if session is None or session.mode is not SessionMode.UPLOAD:
            return state
        return replace(
            state,
            session=replace(
                session,
                playback=replace(
                    session.playback,
                    is_playing=False,
                    position_seconds=command.position_seconds,
                    error_code=command.error_code,
                    error_message=command.error_message,
                ),
            ),
        )
    if isinstance(command, PlaybackRecovered):
        session = state.session
        if session is None or session.mode is not SessionMode.UPLOAD:
            return state
        return replace(
            state,
            session=replace(
                session,
                playback=replace(
                    session.playback,
                    position_seconds=command.position_seconds,
                    is_playing=command.is_playing,
                    error_code=None,
                    error_message=None,
                ),
            ),
        )
    raise StateError(f"unsupported command: {type(command).__name__}")
