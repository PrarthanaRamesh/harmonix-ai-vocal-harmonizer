"""Milestone 3 audio input/device adapter.

This module intentionally stays independent of pitch detection and other downstream
pipeline stages. It handles device enumeration, validation, state transitions, and
bounded callback-to-worker handoff for live audio capture.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any

import numpy as np
import samplerate
import sounddevice as sd

from ..config.validation import validate_frame_size
from ..domain.errors import (
    CaptureStartError,
    ConfigurationError,
    DeviceUnavailableError,
    NoInputDeviceError,
    PermissionDeniedError,
    QueueOverflowError,
    StaleSelectedDeviceError,
)

CANONICAL_SAMPLE_RATE = 44100
DEFAULT_FRAME_SIZE = 2048
FRAME_SIZE_MIN = 256
FRAME_SIZE_MAX = 4096


@dataclass(frozen=True)
class DeviceInfo:
    index: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    is_default: bool = False
    host_api_name: str = ""


class InputAdapter:
    """Handle input-device discovery, validation, capture state, and canonical frame assembly."""

    def __init__(
        self,
        device_index: int | None = None,
        queue_size: int = 16,
        *,
        input_selector: str | None = None,
        output_selector: str | None = None,
        frame_size: int = DEFAULT_FRAME_SIZE,
    ):
        validate_frame_size(frame_size)
        self.device_index = device_index
        self.input_selector = input_selector
        self.output_selector = output_selector
        self.frame_size = frame_size
        self.queue_size = max(1, int(queue_size))
        self._queue: list[np.ndarray] = []
        self._queue_lock = Lock()
        self._capturing = False
        self._resampler = None
        self._output_resampler = None
        self._native_sample_rate: int | None = None
        self._pending_native: np.ndarray | None = None
        self._queue_overflowed = False
        self.fallback_occurred = False
        self._active_stream: Any = None

    @property
    def queue_overflowed(self) -> bool:
        return self._queue_overflowed

    @property
    def native_sample_rate(self) -> int | None:
        return self._native_sample_rate

    @staticmethod
    def list_devices() -> list[DeviceInfo]:
        """List available input devices using the approved sounddevice stack."""
        return InputAdapter._enumerate_devices("input")

    @staticmethod
    def list_output_devices() -> list[DeviceInfo]:
        """List output-capable devices without changing the input-list contract."""
        return InputAdapter._enumerate_devices("output")

    @staticmethod
    def _enumerate_devices(direction: str) -> list[DeviceInfo]:
        if direction not in {"input", "output"}:
            raise ValueError("direction must be input or output")
        host_apis = sd.query_hostapis()
        default_index = InputAdapter._default_index(direction)
        devices: list[DeviceInfo] = []
        for index, device in enumerate(sd.query_devices()):
            try:
                input_channels = int(device.get("max_input_channels", 0) or 0)
                output_channels = int(device.get("max_output_channels", 0) or 0)
            except Exception:
                input_channels = output_channels = 0
            if (input_channels if direction == "input" else output_channels) <= 0:
                continue
            host_api_index = device.get("hostapi")
            try:
                host_api_name = str(host_apis[int(host_api_index)].get("name", ""))
            except (TypeError, ValueError, IndexError, AttributeError):
                host_api_name = ""
            info = DeviceInfo(
                index=index,
                name=str(device.get("name", f"device_{index}")),
                max_input_channels=input_channels,
                max_output_channels=output_channels,
                default_samplerate=float(device.get("default_samplerate", 44100.0) or 44100.0),
                is_default=index == default_index,
                host_api_name=host_api_name,
            )
            devices.append(info)
        return devices

    def has_input_device(self) -> bool:
        try:
            self.require_selected_device()
            return True
        except (NoInputDeviceError, StaleSelectedDeviceError, DeviceUnavailableError, ConfigurationError):
            return False

    def require_selected_device(self) -> DeviceInfo:
        if self.device_index is not None:
            for device in self.list_devices():
                if device.index == self.device_index:
                    return device
            if self.device_index < 0:
                raise DeviceUnavailableError(f"Input device index {self.device_index} is unavailable.")
            raise StaleSelectedDeviceError(f"Selected device index {self.device_index} is stale or unavailable.")
        if self.input_selector is not None:
            return self._resolve_selector(self.input_selector, "input")
        default_index = self._default_index("input")
        if default_index is None:
            raise NoInputDeviceError("No microphone/input system default is available.")
        try:
            return self._device_by_index(default_index, "input")
        except StaleSelectedDeviceError as exc:
            raise NoInputDeviceError("The system-default input device is unavailable.") from exc

    def start_capture(self, device_index: int | None = None) -> "DeviceInfo | None":
        """Start input capture and return the DeviceInfo actually used.

        Returns the DeviceInfo of the device that was selected (which may differ from
        the saved device when EC-1.5 fallback to the system default occurs).  Returns
        None only when the system default is selected without any saved preference.
        Callers may inspect the returned DeviceInfo to update persisted settings when a
        fallback occurred.
        """
        target = device_index if device_index is not None else self.device_index
        self.fallback_occurred = False
        try:
            device = self.require_selected_device() if target is None else self._device_by_index(target, "input")
            self._native_sample_rate = int(device.default_samplerate)
            if self._native_sample_rate <= 0:
                raise DeviceUnavailableError(f"Input device {device.index} reports an invalid sample rate.")
            self._resampler = samplerate.Resampler('sinc_best', 1)
            self._capturing = True
            self._queue_overflowed = False
            return device
        except (StaleSelectedDeviceError, DeviceUnavailableError) as exc:
            if isinstance(exc, DeviceUnavailableError) and self.input_selector is None:
                self.stop_capture()
                raise
            # EC-1.5: saved device is stale/missing — attempt system-default fallback.
            default_index = self._default_index("input")
            if default_index is None:
                self.stop_capture()
                raise NoInputDeviceError(
                    "Saved input device is stale and no system-default input device is available."
                )
            try:
                fallback = self._device_by_index(default_index, "input")
            except StaleSelectedDeviceError as exc:
                self.stop_capture()
                raise NoInputDeviceError(
                    "Saved input device is stale and the system-default input device is also unavailable."
                ) from exc
            try:
                self._native_sample_rate = int(fallback.default_samplerate)
                if self._native_sample_rate <= 0:
                    raise DeviceUnavailableError(
                        f"System-default input device {fallback.index} reports an invalid sample rate."
                    )
                self._resampler = samplerate.Resampler('sinc_best', 1)
                self._capturing = True
                self._queue_overflowed = False
                self.fallback_occurred = True
                return fallback
            except (DeviceUnavailableError, ConfigurationError):
                self.stop_capture()
                raise
        except PermissionError as exc:
            # EC-1.2: OS denied microphone access — surface as a distinct error type.
            self.stop_capture()
            raise PermissionDeniedError(
                f"Microphone access was denied by the operating system: {exc}"
            ) from exc
        except (NoInputDeviceError, DeviceUnavailableError, ConfigurationError):
            self.stop_capture()
            raise
        except Exception as exc:  # pragma: no cover
            self.stop_capture()
            raise CaptureStartError(f"Could not start capture for selected input: {exc}") from exc

    def create_duplex_stream(self, callback: Any, device_index: int | None = None) -> Any:
        """Create one validated duplex stream; None selectors mean explicit system defaults."""
        try:
            input_device = (
                self._device_by_index(device_index, "input")
                if device_index is not None
                else self.require_selected_device()
            )
            output_device = self._resolve_selector(self.output_selector, "output") if self.output_selector is not None else self._default_device("output")
            sample_rate = int(input_device.default_samplerate)
            if sample_rate <= 0:
                raise DeviceUnavailableError(f"Input device {input_device.index} reports an invalid sample rate.")
            sd.check_input_settings(device=input_device.index, channels=1, dtype="float32", samplerate=sample_rate)
            sd.check_output_settings(device=output_device.index if output_device else None, channels=1, dtype="float32", samplerate=sample_rate)
            return sd.Stream(
                # Resolve system defaults once, then pin this stream to those validated devices.
                device=(input_device.index, output_device.index),
                samplerate=sample_rate,
                channels=(1, 1),
                dtype="float32",
                blocksize=self.frame_size,
                callback=callback,
            )
        except (NoInputDeviceError, StaleSelectedDeviceError, DeviceUnavailableError, ConfigurationError):
            self.stop_capture()
            raise
        except PermissionError as exc:
            # EC-1.2: OS denied access to the audio device — surface as a distinct error type.
            self.stop_capture()
            raise PermissionDeniedError(
                f"Audio device access was denied by the operating system: {exc}"
            ) from exc
        except Exception as exc:
            self.stop_capture()
            raise CaptureStartError(f"Could not create duplex stream for selected devices: {exc}") from exc

    @staticmethod
    def _default_index(direction: str) -> int | None:
        defaults = sd.default.device
        if not isinstance(defaults, (tuple, list)) or len(defaults) < 2:
            return None
        index = defaults[0 if direction == "input" else 1]
        return int(index) if isinstance(index, (int, np.integer)) and int(index) >= 0 else None

    @staticmethod
    def _device_by_index(index: int, direction: str) -> DeviceInfo:
        devices = InputAdapter.list_devices() if direction == "input" else InputAdapter.list_output_devices()
        for device in devices:
            if device.index == index:
                return device
        raise StaleSelectedDeviceError(f"Selected {direction} device index {index} is stale or unavailable.")

    @classmethod
    def _default_device(cls, direction: str) -> DeviceInfo | None:
        index = cls._default_index(direction)
        if index is None:
            raise DeviceUnavailableError(f"No {direction} system default is available.")
        return cls._device_by_index(index, direction)

    @classmethod
    def _resolve_selector(cls, selector: str, direction: str) -> DeviceInfo:
        from ..config.validation import resolve_audio_device_selector

        devices = cls.list_devices() if direction == "input" else cls.list_output_devices()
        resolved = resolve_audio_device_selector(selector, devices)
        if resolved is None:
            raise DeviceUnavailableError(f"No {direction} system default is available.")
        return resolved

    def stop_capture(self) -> None:
        self._capturing = False
        self.fallback_occurred = False
        self._queue = []
        self._pending_native = None
        self._native_sample_rate = None
        self._resampler = None
        self._output_resampler = None
        self._active_stream = None

    def is_capturing(self) -> bool:
        return self._capturing

    def _detect_native_sample_rate(self, device_index: int) -> int:
        device = self.require_selected_device()
        native_rate = int(device.default_samplerate)
        if native_rate <= 0:
            raise DeviceUnavailableError(f"Device {device_index} reports an invalid sample rate.")
        return native_rate

    @staticmethod
    def normalize_to_mono_float32(data: np.ndarray) -> np.ndarray:
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim == 1:
            return arr.astype(np.float32, copy=False)
        if arr.ndim >= 2:
            return arr.mean(axis=1, dtype=np.float32).astype(np.float32, copy=False)
        return arr.astype(np.float32, copy=False)

    def enqueue_native_chunk(self, chunk: np.ndarray) -> None:
        if self.queue_size <= 0:
            raise QueueOverflowError("Queue size must be positive or data cannot be queued.")
        with self._queue_lock:
            if len(self._queue) >= self.queue_size:
                self._queue_overflowed = True
                raise QueueOverflowError("Input queue is full; callback-to-worker handoff dropped a chunk.")
            self._queue.append(np.asarray(chunk, dtype=np.float32))

    def _pull_next_native_chunk(self) -> np.ndarray | None:
        with self._queue_lock:
            if not self._queue:
                return None
            return self._queue.pop(0)

    def process_native_chunk(self, chunk: np.ndarray, sample_rate: int) -> np.ndarray:
        mono = self.normalize_to_mono_float32(chunk)
        if sample_rate == CANONICAL_SAMPLE_RATE:
            return mono.astype(np.float32, copy=False)
        if self._resampler is None:
            self._resampler = samplerate.Resampler('sinc_best', 1)
        out = self._resampler.process(mono, float(CANONICAL_SAMPLE_RATE) / float(sample_rate), end_of_input=False)
        return np.asarray(out, dtype=np.float32)

    def process_canonical_output(self, chunk: np.ndarray, native_sample_rate: int) -> np.ndarray:
        """Resample canonical 44,100 Hz output audio to the stream's native sample rate."""
        arr = np.asarray(chunk, dtype=np.float32)
        if native_sample_rate == CANONICAL_SAMPLE_RATE:
            return arr.astype(np.float32, copy=False)
        if self._output_resampler is None:
            self._output_resampler = samplerate.Resampler('sinc_best', 1)
        out = self._output_resampler.process(
            arr, float(native_sample_rate) / float(CANONICAL_SAMPLE_RATE), end_of_input=False
        )
        return np.asarray(out, dtype=np.float32)

    def emit_canonical_frames(self, data: np.ndarray, sample_rate: int) -> list[np.ndarray]:
        """Convert callback-sized chunks into complete fixed 2048-sample processing frames."""
        validate_frame_size(DEFAULT_FRAME_SIZE)
        mono = self.normalize_to_mono_float32(data)
        if sample_rate != CANONICAL_SAMPLE_RATE:
            if self._resampler is None:
                self._resampler = samplerate.Resampler('sinc_best', 1)
            mono = np.asarray(
                self._resampler.process(mono, float(CANONICAL_SAMPLE_RATE) / float(sample_rate), end_of_input=False),
                dtype=np.float32,
            )
        if self._pending_native is not None and self._pending_native.size:
            mono = np.concatenate((self._pending_native, mono))
        remainder = mono.size % DEFAULT_FRAME_SIZE
        complete_size = mono.size - remainder
        self._pending_native = mono[complete_size:].copy() if remainder else None
        if complete_size == 0:
            return []
        return [mono[i : i + DEFAULT_FRAME_SIZE] for i in range(0, complete_size, DEFAULT_FRAME_SIZE)]
