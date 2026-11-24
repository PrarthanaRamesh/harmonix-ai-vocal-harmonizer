"""Background WAV export worker with canonical state reporting."""

from __future__ import annotations

from concurrent.futures import CancelledError
import logging
from pathlib import Path
from threading import Lock, Thread, current_thread
from typing import Sequence
from uuid import uuid4

from ..domain.errors import ExportValidationError
from ..domain.models import RecordedBuffers
from ..session_state.app_state import StateStore
from ..session_state.commands import (
    CancelExport,
    CompleteExport,
    FailExport,
    RequestExportCancellation,
    StartExport,
    UpdateExportProgress,
)
from .engine import ExportCancellation, ExportEngine

logger = logging.getLogger(__name__)


class ExportWorker:
    """Run the WAV engine on a background thread and publish lifecycle events."""

    def __init__(self, store: StateStore, engine: ExportEngine | None = None) -> None:
        self.store = store
        self.engine = engine or ExportEngine()
        self._thread: Thread | None = None
        self._request_id: str | None = None
        self._cancel_event: ExportCancellation | None = None
        self._lock = Lock()

    def start(self, destination: str | Path, *, enabled_voices: Sequence[str]) -> str:
        """Start exporting the current session using an explicit voice selection."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("A WAV export is already running.")
            source = self.store.export_source()
            if source is None:
                raise ExportValidationError("export requires a session with recorded audio")
            request_id = uuid4().hex
            cancel_event = ExportCancellation()
            voice_selection = tuple(enabled_voices)
            self.store.publish_export_event(StartExport(request_id))
            self._request_id = request_id
            self._cancel_event = cancel_event
            self._thread = Thread(
                target=self._run,
                args=(request_id, source.recorded_buffers, destination, voice_selection, cancel_event),
                name="harmonix-export",
                daemon=True,
            )
            try:
                self._thread.start()
            except Exception as exc:
                self._publish_terminal(
                    FailExport(request_id, type(exc).__name__, str(exc)),
                    request_id,
                )
                raise
            return request_id

    def request_cancel(self) -> None:
        with self._lock:
            request_id = self._request_id
            cancel_event = self._cancel_event
            thread = self._thread
            if request_id is None or cancel_event is None or thread is None or not thread.is_alive():
                return
            if not cancel_event.cancel():
                return
            self.store.publish_export_event(RequestExportCancellation(request_id))

    def join(self, timeout: float | None = None) -> bool:
        """Wait up to timeout and report whether the worker actually terminated."""
        with self._lock:
            thread = self._thread
        if thread is None:
            return True
        if thread.ident is None:
            return True
        if thread is current_thread():
            return False
        thread.join(timeout)
        return not thread.is_alive()

    def shutdown(self, timeout: float | None = None) -> bool:
        """Request cooperative cancellation and report whether shutdown completed."""
        self.request_cancel()
        return self.join(timeout)

    @property
    def is_alive(self) -> bool:
        with self._lock:
            thread = self._thread
        return thread is not None and thread.is_alive()

    def _run(
        self,
        request_id: str,
        recorded_buffers: RecordedBuffers,
        destination: str | Path,
        enabled_voices: tuple[str, ...],
        cancel_event: ExportCancellation,
    ) -> None:
        def publish_progress(phase: str, completed: int | None, total: int | None) -> None:
            if phase != "completed":
                self.store.publish_export_event(
                    UpdateExportProgress(request_id, phase, completed, total)
                )

        try:
            output_path = self.engine.export(
                recorded_buffers,
                destination,
                enabled_voices=enabled_voices,
                progress_callback=publish_progress,
                cancel_event=cancel_event,
            )
        except CancelledError:
            self._publish_terminal(CancelExport(request_id), request_id)
        except Exception as exc:
            self._publish_terminal(
                FailExport(request_id, type(exc).__name__, str(exc)),
                request_id,
            )
        else:
            self._publish_terminal(CompleteExport(request_id, output_path), request_id)

    def _publish_terminal(
        self,
        command: CancelExport | CompleteExport | FailExport,
        request_id: str,
    ) -> bool:
        """Publish a terminal event, retrying once without inventing state on failure."""
        for attempt in (1, 2):
            try:
                self.store.publish_export_event(command)
            except Exception:
                logger.exception(
                    "Export terminal state publication attempt %d/2 failed for request %s",
                    attempt,
                    request_id,
                )
            else:
                return True
        logger.error(
            "Export terminal state for request %s could not be confirmed; "
            "canonical export state may remain active",
            request_id,
        )
        return False
