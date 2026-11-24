"""Latest-value bridge from Live processing to the Pygame/state thread."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

from .key_scale_resolver import KeyScale


@dataclass(frozen=True)
class LiveTelemetryFrame:
    request_id: str
    input_level: float
    current_f0: float | None
    confidence: float
    tracking_state: str
    in_tune: bool
    detected_key: KeyScale | None
    effective_key: KeyScale | None
    physical_latency_ms: float | None
    pipeline_latency_ms: float | None
    latency_provenance: str
    mic_connected: bool = True
    error_message: str | None = None


class VisualizerBridge:
    """Coalesce worker telemetry so the UI never waits for the audio pipeline."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._sequence = 0
        self._latest: LiveTelemetryFrame | None = None

    def publish(self, frame: LiveTelemetryFrame) -> int:
        """Publish one worker-produced snapshot; intended for non-callback threads."""
        with self._lock:
            self._sequence += 1
            self._latest = frame
            return self._sequence

    def read_after(self, sequence: int) -> tuple[int, LiveTelemetryFrame] | None:
        """Return the latest newer snapshot without waiting for audio processing."""
        with self._lock:
            if self._latest is None or self._sequence <= sequence:
                return None
            return self._sequence, self._latest
