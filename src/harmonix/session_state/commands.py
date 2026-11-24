"""Commands accepted by the foundational state store."""

from dataclasses import dataclass

from ..domain.enums import Screen, StylePreset
from ..domain.models import GlobalSettings, KeyState, RecordedBuffers, RecentSessionSummary


@dataclass(frozen=True)
class Navigate:
    screen: Screen


@dataclass(frozen=True)
class ClearSession:
    pass


@dataclass(frozen=True)
class EndSession:
    """Application-level intent; PygameApp persists before dispatching ClearSession."""


@dataclass(frozen=True)
class ReplaceRecentSessions:
    sessions: tuple[RecentSessionSummary, ...]


@dataclass(frozen=True)
class ReplaceGlobalSettings:
    settings: GlobalSettings


@dataclass(frozen=True)
class UpdateGlobalSetting:
    field: str
    value: object


@dataclass(frozen=True)
class LiveTelemetryUpdated:
    request_id: str
    input_level: float
    current_f0: float | None
    confidence: float
    tracking_state: str
    in_tune: bool
    detected_key: KeyState | None
    effective_key: KeyState | None
    physical_latency_ms: float | None
    pipeline_latency_ms: float | None
    latency_provenance: str
    mic_connected: bool = True
    error_message: str | None = None


@dataclass(frozen=True)
class StartLiveSession:
    request_id: str
    selected_voices: tuple[str, ...] = ("3rd", "5th")
    style_preset: StylePreset = StylePreset.POP
    requested_key: KeyState | None = None


@dataclass(frozen=True)
class StopLiveSession:
    recorded_buffers: RecordedBuffers
    request_id: str | None = None


@dataclass(frozen=True)
class StartUploadAnalysis:
    request_id: str
    source_file: str
    selected_voices: tuple[str, ...]
    style_preset: StylePreset = StylePreset.POP
    requested_key: KeyState | None = None
    keep_configure_open: bool = False
    continuation_request_id: str | None = None


@dataclass(frozen=True)
class UpdateUploadAnalysisProgress:
    request_id: str
    phase: str
    completed_units: int
    total_units: int | None


@dataclass(frozen=True)
class RequestUploadCancellation:
    request_id: str


@dataclass(frozen=True)
class CompleteUploadAnalysis:
    request_id: str
    recorded_buffers: RecordedBuffers
    key: KeyState
    duration_seconds: float
    detected_key: KeyState | None = None


@dataclass(frozen=True)
class SelectUploadKey:
    key: KeyState


@dataclass(frozen=True)
class CancelUploadAnalysis:
    request_id: str


@dataclass(frozen=True)
class FailUploadAnalysis:
    request_id: str
    error_code: str
    error_message: str


@dataclass(frozen=True)
class StartExport:
    request_id: str


@dataclass(frozen=True)
class SetExportVoices:
    voices: tuple[str, ...]


@dataclass(frozen=True)
class UpdateExportProgress:
    request_id: str
    phase: str
    completed_units: int | None
    total_units: int | None


@dataclass(frozen=True)
class RequestExportCancellation:
    request_id: str


@dataclass(frozen=True)
class CancelExport:
    request_id: str


@dataclass(frozen=True)
class CompleteExport:
    request_id: str
    output_path: str


@dataclass(frozen=True)
class FailExport:
    request_id: str
    error_code: str
    error_message: str


@dataclass(frozen=True)
class PlaybackPlay:
    pass


@dataclass(frozen=True)
class PlaybackPause:
    pass


@dataclass(frozen=True)
class PlaybackSeek:
    request_id: str
    seek_seconds: float


@dataclass(frozen=True)
class PlaybackSetVoices:
    enabled_voices: tuple[str, ...]


@dataclass(frozen=True)
class PlaybackEngineUpdate:
    position_seconds: float
    is_playing: bool
    ended: bool = False


@dataclass(frozen=True)
class PlaybackSeekApplied:
    request_id: str
    position_seconds: float
    is_playing: bool
    ended: bool = False


@dataclass(frozen=True)
class PlaybackFailed:
    error_code: str
    error_message: str
    position_seconds: float


@dataclass(frozen=True)
class PlaybackRecovered:
    position_seconds: float
    is_playing: bool
