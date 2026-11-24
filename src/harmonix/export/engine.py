"""Export engine for WAV rendering and safe file replacement."""

from __future__ import annotations

import logging
import os
import tempfile
from concurrent.futures import CancelledError
from pathlib import Path
from threading import Event, Lock
from typing import Callable, Sequence

import numpy as np
from scipy.io import wavfile

from ..config.models import DEFAULT_EXPORT_FORMAT
from ..domain.errors import ExportValidationError, ExportWriteError
from ..domain.models import RecordedBuffers

SUPPORTED_VOICES = ("3rd", "5th", "octave")
EXPORT_RENDER_BLOCK_SIZE = 2048
ProgressCallback = Callable[[str, int | None, int | None], None]
logger = logging.getLogger(__name__)


class ExportCancellation:
    """Serialize cancellation with the final destination commit.

    The token owns the commit lock. Worker cancellation acquires its worker
    lock before this token lock; the engine acquires only the token lock.
    The token lock covers the final cancellation check, ``os.replace``, and
    commit marker only. Callbacks and state publication must happen outside it.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._cancelled = False
        self._committed = False

    def is_set(self) -> bool:
        with self._lock:
            return self._cancelled

    def cancel(self) -> bool:
        """Request cancellation, returning false if the destination already committed."""
        with self._lock:
            if self._committed:
                return False
            self._cancelled = True
            return True

    def replace(self, source: Path, destination: Path) -> None:
        """Serialize the final cancellation check and replacement commit point."""
        with self._lock:
            if self._cancelled:
                raise CancelledError("WAV export was cancelled before destination commit")
            os.replace(source, destination)
            self._committed = True


class ExportEngine:
    """Render a final mono WAV mix from the source and enabled harmony tracks."""

    def export(
        self,
        recorded_buffers: RecordedBuffers,
        destination: str | os.PathLike[str],
        *,
        enabled_voices: Sequence[str] | None = None,
        progress_callback: ProgressCallback | None = None,
        cancel_event: Event | ExportCancellation | None = None,
    ) -> str:
        """Export the original plus selected harmony buffers as a WAV.

        When ``enabled_voices`` is omitted, all available supported harmony
        buffers are included. A session-aware caller must explicitly pass the
        voices active at session end to satisfy FR-9.1.
        """
        try:
            destination_path = Path(destination)
        except (TypeError, ValueError) as exc:
            raise ExportValidationError("export destination must be a valid file path") from exc
        if destination_path.suffix.lower() != ".wav":
            raise ExportValidationError("Harmonix Item 11 exports WAV only")
        if destination_path.name == "":
            raise ExportValidationError("export destination must include a file name")
        if not destination_path.parent.exists() or not destination_path.parent.is_dir():
            raise ExportValidationError("export destination parent directory does not exist")
        if destination_path.exists() and not destination_path.is_file():
            raise ExportValidationError("export destination must not be a directory")

        original = self._require_original(recorded_buffers)
        sample_rate = self._require_sample_rate(recorded_buffers)
        selected_voices = self._select_voices(recorded_buffers, enabled_voices)

        voice_arrays = {
            voice_name: self._require_voice(recorded_buffers, voice_name, original.shape)
            for voice_name in selected_voices
        }
        mixed = np.empty_like(original)
        total_samples = int(original.size)
        for start in range(0, total_samples, EXPORT_RENDER_BLOCK_SIZE):
            self._raise_if_cancelled(cancel_event)
            end = min(start + EXPORT_RENDER_BLOCK_SIZE, total_samples)
            block = original[start:end].copy()
            for voice_array in voice_arrays.values():
                block += voice_array[start:end]
            mixed[start:end] = block
            if not self._report_progress(progress_callback, "rendering", end, total_samples):
                progress_callback = None

        self._raise_if_cancelled(cancel_event)
        if not self._report_progress(progress_callback, "writing", None, None):
            progress_callback = None
        self._write_wav_safely(destination_path, sample_rate, mixed, cancel_event=cancel_event)
        self._report_progress(progress_callback, "completed", total_samples, total_samples)

        return str(destination_path)

    @staticmethod
    def _require_original(recorded_buffers: RecordedBuffers) -> np.ndarray:
        if not isinstance(recorded_buffers.original, np.ndarray):
            raise ExportValidationError("export requires a recorded original buffer")
        try:
            original = np.asarray(recorded_buffers.original, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ExportValidationError("export original buffer must contain numeric samples") from exc
        if original.ndim != 1:
            raise ExportValidationError("export requires a mono original buffer")
        if original.size == 0 or not np.isfinite(original).all():
            raise ExportValidationError("export original buffer is empty or contains non-finite samples")
        return original

    @staticmethod
    def _require_sample_rate(recorded_buffers: RecordedBuffers) -> int:
        sample_rate = recorded_buffers.sample_rate
        if (
            isinstance(sample_rate, (bool, np.bool_))
            or not isinstance(sample_rate, (int, np.integer))
            or sample_rate <= 0
        ):
            raise ExportValidationError("export requires a positive integer sample rate")
        return int(sample_rate)

    @staticmethod
    def _select_voices(
        recorded_buffers: RecordedBuffers,
        enabled_voices: Sequence[str] | None,
    ) -> tuple[str, ...]:
        if enabled_voices is None:
            return tuple(
                voice
                for voice in SUPPORTED_VOICES
                if voice in recorded_buffers.a_per_voice
                and recorded_buffers.a_per_voice[voice] is not None
            )
        if isinstance(enabled_voices, str):
            raise ExportValidationError("enabled voices must be a sequence of voice names")
        try:
            voice_iterator = iter(enabled_voices)
        except Exception as exc:
            raise ExportValidationError("enabled voices must be an iterable of voice names") from exc
        selected = []
        seen: set[str] = set()
        while True:
            try:
                voice_name = next(voice_iterator)
            except StopIteration:
                break
            except Exception as exc:
                raise ExportValidationError("enabled voice selection could not be read") from exc
            if not isinstance(voice_name, str) or voice_name not in SUPPORTED_VOICES:
                raise ExportValidationError(f"unsupported export voice: {voice_name!r}")
            if voice_name in seen:
                continue
            if voice_name not in recorded_buffers.a_per_voice or recorded_buffers.a_per_voice[voice_name] is None:
                raise ExportValidationError(f"export voice buffer is unavailable: {voice_name}")
            seen.add(voice_name)
            selected.append(voice_name)
        return tuple(selected)

    @staticmethod
    def _require_voice(
        recorded_buffers: RecordedBuffers,
        voice_name: str,
        original_shape: tuple[int, ...],
    ) -> np.ndarray:
        voice_buffer = recorded_buffers.a_per_voice[voice_name]
        if not isinstance(voice_buffer, np.ndarray):
            raise ExportValidationError(f"export voice buffer is invalid for {voice_name}")
        try:
            voice_array = np.asarray(voice_buffer, dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ExportValidationError(f"export voice buffer must contain numeric samples for {voice_name}") from exc
        if voice_array.shape != original_shape:
            raise ExportValidationError(f"voice buffer mismatch for {voice_name}")
        if not np.isfinite(voice_array).all():
            raise ExportValidationError(f"export voice buffer contains non-finite samples for {voice_name}")
        return voice_array

    @staticmethod
    def _report_progress(
        callback: ProgressCallback | None,
        phase: str,
        completed: int | None,
        total: int | None,
    ) -> bool:
        if callback is not None:
            try:
                callback(phase, completed, total)
            except Exception:
                logger.warning(
                    "Export progress callback failed during %s; disabling further notifications",
                    phase,
                    exc_info=True,
                )
                return False
        return True

    @staticmethod
    def _raise_if_cancelled(cancel_event: Event | ExportCancellation | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError("WAV export was cancelled before destination commit")

    @staticmethod
    def _write_wav_safely(
        destination: Path,
        sample_rate: int,
        audio: np.ndarray,
        *,
        cancel_event: Event | ExportCancellation | None = None,
    ) -> None:
        if not destination.parent.exists() or not destination.parent.is_dir():
            raise ExportWriteError(f"unable to write export to {destination}")

        temp_file: Path | None = None
        try:
            ExportEngine._raise_if_cancelled(cancel_event)
            with tempfile.NamedTemporaryFile(
                prefix="harmonix-export-",
                suffix=".tmp",
                dir=str(destination.parent),
                delete=False,
            ) as handle:
                temp_file = Path(handle.name)
            wavfile.write(str(temp_file), sample_rate, audio)
            if isinstance(cancel_event, ExportCancellation):
                cancel_event.replace(temp_file, destination)
            else:
                ExportEngine._raise_if_cancelled(cancel_event)
                os.replace(temp_file, destination)
        except CancelledError as exc:
            cleanup_error = ExportEngine._cleanup_temp_file(temp_file)
            if cleanup_error is not None:
                exc.add_note(f"temporary file cleanup failed: {cleanup_error}")
            raise
        except PermissionError as exc:
            cleanup_error = ExportEngine._cleanup_temp_file(temp_file)
            message = f"permission denied while writing export to {destination}"
            if cleanup_error is not None:
                message = f"{message}; temporary file cleanup failed: {cleanup_error}"
            raise ExportWriteError(message) from exc
        except Exception as exc:  # pragma: no cover - runtime path assertions are tested via ExportWriteError.
            cleanup_error = ExportEngine._cleanup_temp_file(temp_file)
            message = f"unable to write export to {destination}"
            if cleanup_error is not None:
                message = f"{message}; temporary file cleanup failed: {cleanup_error}"
            raise ExportWriteError(message) from exc

    @staticmethod
    def _cleanup_temp_file(temp_file: Path | None) -> OSError | None:
        if temp_file is None:
            return None
        try:
            temp_file.unlink(missing_ok=True)
        except OSError as exc:
            return exc
        return None


DEFAULT_EXPORT_FORMAT = DEFAULT_EXPORT_FORMAT
