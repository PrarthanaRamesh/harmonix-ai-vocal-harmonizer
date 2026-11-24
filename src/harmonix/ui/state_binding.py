"""Compact read-only projections and UI intent helpers for Pygame views."""

from dataclasses import dataclass

import numpy as np

from ..session_state.app_state import AppUIProjection, SessionUIProjection
from ..domain.models import RecentSessionSummary


@dataclass(frozen=True)
class AppViewModel:
    """Immutable data a screen may render without owning application state."""

    screen_name: str
    recent_sessions: tuple[RecentSessionSummary, ...]
    has_active_session: bool
    reduced_motion: bool
    default_key_behavior: str
    default_style_preset: str
    latency_buffer_frame_size: int
    session: SessionUIProjection | None
    waveform: tuple[tuple[float, float], ...]
    export_selected_voices: tuple[str, ...] = ()
    export_in_progress: bool = False
    export_status: str = "idle"
    export_phase: str | None = None
    export_progress_pct: float | None = None
    export_error: str | None = None


def build_view_model(
    state: AppUIProjection,
    waveform: tuple[tuple[float, float], ...] = (),
) -> AppViewModel:
    """Project canonical state into display data without duplicating ownership."""
    return AppViewModel(
        screen_name=state.current_screen,
        recent_sessions=state.recent_sessions,
        has_active_session=state.session is not None,
        reduced_motion=state.reduced_motion,
        default_key_behavior=state.default_key_behavior,
        default_style_preset=state.default_style_preset,
        latency_buffer_frame_size=state.latency_buffer_frame_size,
        session=state.session,
        waveform=waveform,
        export_selected_voices=state.session.export_selected_voices if state.session else (),
        export_in_progress=state.session.export_in_progress if state.session else False,
        export_status=state.session.export_status if state.session else "idle",
        export_phase=state.session.export_phase if state.session else None,
        export_progress_pct=state.session.export_progress_pct if state.session else None,
        export_error=state.session.export_error if state.session else None,
    )


def downsample_waveform(audio: np.ndarray, point_count: int = 320) -> tuple[tuple[float, float], ...]:
    """Reduce one immutable mono buffer to min/max display bins without copying it."""
    if point_count <= 0:
        raise ValueError("point_count must be positive")
    if audio.ndim != 1 or audio.size == 0:
        return ()
    count = min(point_count, int(audio.size))
    waveform: list[tuple[float, float]] = []
    for index in range(count):
        start = index * audio.size // count
        end = max(start + 1, (index + 1) * audio.size // count)
        segment = audio[start:end]
        waveform.append((float(np.min(segment)), float(np.max(segment))))
    return tuple(waveform)


def upload_progress_label(completed: int, total: int | None) -> str:
    """Avoid presenting unknown or underreported work as a false percentage."""
    if total is None or total <= 0 or completed > total:
        return f"{completed:,} units"
    return f"{completed:,} / {total:,} units"
