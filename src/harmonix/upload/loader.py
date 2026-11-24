"""Decode supported uploads and normalize them to the canonical audio format."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable

import librosa
import numpy as np
import soundfile as sf

from ..config.models import MAX_UPLOAD_DURATION_SECONDS, SUPPORTED_UPLOAD_FORMATS
from ..config.validation import validate_upload_metadata
from ..domain.errors import ConfigurationError
from ..input.adapter import CANONICAL_SAMPLE_RATE, DEFAULT_FRAME_SIZE, InputAdapter
from .errors import (
    CorruptUploadError,
    InvalidUploadDurationError,
    MP3DecoderUnavailableError,
    UploadCancelledError,
    UploadDurationExceededError,
    UnsupportedUploadFormatError,
)

ProgressCallback = Callable[[str, int, int], None]
DECODE_CHUNK_FRAMES = 65536


@dataclass(frozen=True)
class LoadedAudio:
    audio: np.ndarray
    source_sample_rate: int
    source_sample_count: int
    sample_rate: int
    duration_seconds: float


@dataclass(frozen=True)
class CanonicalFrame:
    sample_start: int
    audio: np.ndarray


def validate_decoded_duration(sample_count: int, sample_rate: int) -> float:
    if sample_count <= 0 or sample_rate <= 0:
        raise InvalidUploadDurationError("Decoded audio must contain samples at a positive sample rate.")
    duration = sample_count / sample_rate
    if duration > MAX_UPLOAD_DURATION_SECONDS:
        raise UploadDurationExceededError("Decoded upload duration exceeds 600 seconds.")
    return duration


def prepare_canonical_frames(audio: np.ndarray, frame_size: int = DEFAULT_FRAME_SIZE) -> tuple[CanonicalFrame, ...]:
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError("Canonical frame input must be mono.")
    if frame_size <= 0:
        raise ValueError("frame_size must be positive.")
    if array.size < frame_size:
        return ()
    starts = range(0, array.size - frame_size + 1, frame_size)
    return tuple(CanonicalFrame(start, array[start : start + frame_size]) for start in starts)


class UploadLoader:
    def __init__(self, *, decode_chunk_frames: int = DECODE_CHUNK_FRAMES) -> None:
        if decode_chunk_frames <= 0:
            raise ValueError("decode_chunk_frames must be positive.")
        self.decode_chunk_frames = decode_chunk_frames

    def load(
        self,
        path: str | Path,
        *,
        cancel_event: Event | None = None,
        progress: ProgressCallback | None = None,
    ) -> LoadedAudio:
        source = Path(path)
        extension = source.suffix.lower().lstrip(".")
        if extension not in SUPPORTED_UPLOAD_FORMATS:
            raise UnsupportedUploadFormatError(f"Unsupported upload format: {extension or '(none)'}.")
        try:
            validate_upload_metadata(extension, 0.0)
        except ConfigurationError as exc:
            raise UnsupportedUploadFormatError(str(exc)) from exc

        chunks: list[np.ndarray] = []
        decoded_frames = 0
        source_rate = 0
        declared_frames = 0
        try:
            with sf.SoundFile(source, mode="r") as audio_file:
                source_rate = int(audio_file.samplerate)
                declared_frames = int(audio_file.frames)
                if source_rate <= 0:
                    raise InvalidUploadDurationError("Decoder reported an invalid sample rate.")
                while True:
                    self._raise_if_cancelled(cancel_event)
                    chunk = audio_file.read(
                        frames=self.decode_chunk_frames,
                        dtype="float32",
                        always_2d=True,
                    )
                    if chunk.size == 0:
                        break
                    chunks.append(chunk)
                    decoded_frames += int(chunk.shape[0])
                    if decoded_frames / source_rate > MAX_UPLOAD_DURATION_SECONDS:
                        raise UploadDurationExceededError("Decoded upload duration exceeds 600 seconds.")
                    if progress is not None:
                        progress("decode", decoded_frames, declared_frames)
        except (UploadCancelledError, UploadDurationExceededError, InvalidUploadDurationError):
            raise
        except sf.LibsndfileError as exc:
            if extension == "mp3" and not self._mp3_decoder_available():
                raise MP3DecoderUnavailableError(
                    "The installed soundfile/libsndfile build does not advertise MP3 decoding."
                ) from exc
            raise CorruptUploadError(f"Could not decode upload: {source.name}.") from exc
        except (OSError, RuntimeError, ValueError) as exc:
            raise CorruptUploadError(f"Could not read upload: {source.name}.") from exc

        self._raise_if_cancelled(cancel_event)
        duration = validate_decoded_duration(decoded_frames, source_rate)
        if not chunks:
            raise InvalidUploadDurationError("Decoded upload contains no audio samples.")
        decoded = np.concatenate(chunks, axis=0)
        mono = InputAdapter.normalize_to_mono_float32(decoded)
        try:
            canonical = (
                mono
                if source_rate == CANONICAL_SAMPLE_RATE
                else librosa.resample(
                    mono,
                    orig_sr=source_rate,
                    target_sr=CANONICAL_SAMPLE_RATE,
                    fix=True,
                )
            )
        except Exception as exc:
            raise CorruptUploadError("Could not resample decoded upload to 44.1 kHz.") from exc
        canonical = np.asarray(canonical, dtype=np.float32)
        if canonical.ndim != 1 or canonical.size == 0 or not np.isfinite(canonical).all():
            raise CorruptUploadError("Canonical upload audio is empty or non-finite.")
        return LoadedAudio(
            audio=canonical,
            source_sample_rate=source_rate,
            source_sample_count=decoded_frames,
            sample_rate=CANONICAL_SAMPLE_RATE,
            duration_seconds=duration,
        )

    @staticmethod
    def _raise_if_cancelled(cancel_event: Event | None) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise UploadCancelledError("Upload analysis was cancelled.")

    @staticmethod
    def _mp3_decoder_available() -> bool:
        try:
            return any(name.upper() in {"MP3", "MPEG"} for name in sf.available_formats())
        except Exception:
            return False