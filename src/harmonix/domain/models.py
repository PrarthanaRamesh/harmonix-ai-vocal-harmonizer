"""Canonical state model definitions without runtime audio behavior."""

from dataclasses import dataclass, field
from typing import Optional

from .enums import DefaultKeyBehavior, Screen, SessionMode, StylePreset


@dataclass(frozen=True)
class RecentSessionSummary:
    id: str
    display_name: str
    mode: SessionMode
    source_file_path: Optional[str]
    last_used_key_root: Optional[str]
    last_used_key_scale_type: Optional[str]
    voice_toggles: tuple[tuple[str, bool], ...]
    style_preset: Optional[str]
    timestamp: str


@dataclass
class KeyState:
    root: Optional[str] = None
    scale_type: Optional[str] = None
    auto_detect: bool = True
    detection_confidence: Optional[float] = None


@dataclass
class VoiceState:
    active: dict[str, bool] = field(
        default_factory=lambda: {"3rd": False, "5th": False, "octave": False}
    )
    per_voice_target_hz: dict[str, float] = field(default_factory=dict)


@dataclass
class PlaybackState:
    is_playing: bool = False
    is_recording: bool = False
    position_seconds: float = 0.0
    duration_seconds: float = 0.0
    pending_seek_seconds: Optional[float] = None
    pending_seek_request_id: Optional[str] = None
    available_voices: tuple[str, ...] = ()
    ended: bool = False
    error_code: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class RecordedBuffers:
    original: object = None
    a_per_voice: dict[str, object] = field(default_factory=dict)
    sample_rate: Optional[int] = None


@dataclass
class AudioStatus:
    mic_connected: bool = False
    input_level: float = 0.0
    tracking_confidence: float = 0.0
    current_f0: Optional[float] = None
    in_tune: bool = False
    tracking_state: str = "idle"
    error_message: Optional[str] = None


@dataclass
class LatencyState:
    current_ms: Optional[float] = None
    pipeline_ms: Optional[float] = None
    provenance: str = "unavailable"
    warning_threshold_ms: Optional[float] = None
    critical_threshold_ms: Optional[float] = None


@dataclass
class ExportState:
    in_progress: bool = False
    progress_pct: Optional[float] = None
    error: Optional[str] = None
    last_export_path: Optional[str] = None
    request_id: Optional[str] = None
    status: str = "idle"
    phase: Optional[str] = None
    completed_units: int = 0
    total_units: Optional[int] = None
    cancellation_requested: bool = False
    selected_voices: tuple[str, ...] = ()


@dataclass
class UploadAnalysisState:
    request_id: Optional[str] = None
    status: str = "idle"
    phase: Optional[str] = None
    completed_units: int = 0
    total_units: Optional[int] = None
    cancellation_requested: bool = False
    error_code: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class GlobalSettings:
    audio_input_device: Optional[str] = None
    audio_output_device: Optional[str] = None
    default_key_behavior: DefaultKeyBehavior = DefaultKeyBehavior.AUTO
    default_style_preset: StylePreset = StylePreset.POP
    latency_buffer_frame_size: int = 2048
    export_default_format: str = "wav"
    export_default_quality: Optional[str] = None
    reduced_motion: bool = False


@dataclass
class SessionState:
    mode: SessionMode
    source_file: Optional[str] = None
    key: KeyState = field(default_factory=KeyState)
    voices: VoiceState = field(default_factory=VoiceState)
    style_preset: StylePreset = StylePreset.POP
    mix_balance: float = 0.0
    playback: PlaybackState = field(default_factory=PlaybackState)
    audio_status: AudioStatus = field(default_factory=AudioStatus)
    latency: LatencyState = field(default_factory=LatencyState)
    export: ExportState = field(default_factory=ExportState)
    upload_analysis: UploadAnalysisState = field(default_factory=UploadAnalysisState)
    recorded_buffers: RecordedBuffers = field(default_factory=RecordedBuffers)
    detected_key: KeyState = field(default_factory=KeyState)
    pending_key: Optional[KeyState] = None
    live_request_id: Optional[str] = None


@dataclass
class AppState:
    current_screen: Screen = Screen.HOME
    recent_sessions: list[RecentSessionSummary] = field(default_factory=list)
    session: Optional[SessionState] = None
    settings: GlobalSettings = field(default_factory=GlobalSettings)
    last_manual_key: Optional[KeyState] = None
