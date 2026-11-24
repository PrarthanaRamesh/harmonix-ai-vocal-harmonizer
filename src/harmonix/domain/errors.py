"""Typed errors for foundational validation, state operations, and M3 input-device handling."""


class HarmonixError(Exception):
    """Base class for expected Harmonix errors."""


class ConfigurationError(HarmonixError, ValueError):
    """Raised when a foundational configuration value is invalid."""


class StateError(HarmonixError, ValueError):
    """Raised when a state command cannot be applied."""


class DeviceError(HarmonixError, ValueError):
    """Raised when the selected audio input device is unusable or unavailable."""


class NoInputDeviceError(DeviceError):
    """Raised when no usable microphone/input device is available."""


class StaleSelectedDeviceError(DeviceError):
    """Raised when the saved or selected device is stale or missing."""


class DeviceUnavailableError(DeviceError):
    """Raised when a device cannot be used at runtime."""


class PermissionDeniedError(DeviceError):
    """Raised when the OS denies access to a microphone or capture stream."""


class CaptureStartError(DeviceError):
    """Raised when the input stream cannot be started or configured."""


class QueueOverflowError(HarmonixError, RuntimeError):
    """Raised when the callback-to-worker queue exceeds its configured bounds."""


class ExportError(HarmonixError, RuntimeError):
    """Base class for failed export operations."""


class ExportValidationError(ExportError, ValueError):
    """Raised when the export input or destination is invalid."""


class ExportWriteError(ExportError):
    """Raised when the output file cannot be written safely."""


class PersistenceError(HarmonixError, RuntimeError):
    """Base class for persistence and metadata store failures."""


class PersistenceReadError(PersistenceError):
    """Raised when JSON metadata cannot be loaded safely."""


class PersistenceWriteError(PersistenceError):
    """Raised when state metadata cannot be written safely."""


class PersistenceLockError(PersistenceError):
    """Raised when another Harmonix process already owns the app-data store lock."""


class PlaybackError(HarmonixError, RuntimeError):
    """Base class for upload playback failures."""


class PlaybackSourceError(PlaybackError, ValueError):
    """Raised when upload buffers cannot be used for synchronized playback."""


class PlaybackOutputError(PlaybackError):
    """Raised when the playback output stream fails or cannot be recovered."""