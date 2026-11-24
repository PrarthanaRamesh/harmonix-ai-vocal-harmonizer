"""Validation for configuration values established in Milestone 1 and M3."""

from collections.abc import Iterable, Mapping

from .models import CORE_VOICES, MAX_UPLOAD_DURATION_SECONDS, SUPPORTED_UPLOAD_FORMATS
from ..domain.errors import ConfigurationError, DeviceUnavailableError
from ..domain.models import GlobalSettings


def validate_frame_size(frame_size: int) -> None:
    """Validate the M3 canonical frame-size contract."""
    if not isinstance(frame_size, int) or isinstance(frame_size, bool):
        raise ConfigurationError("frame_size must be an integer")
    if frame_size < 256 or frame_size > 4096:
        raise ConfigurationError("frame_size must be between 256 and 4096 samples")


def validate_settings(settings: GlobalSettings) -> None:
    """Validate only settings whose rules are resolved by the master spec."""
    if settings.export_default_format.lower() != "wav":
        raise ConfigurationError("Milestone 1 supports the required WAV export format only")
    validate_frame_size(settings.latency_buffer_frame_size)


def validate_setting_update(field: str, value: object) -> None:
    """Validate one user attempted setting before persistence or state mutation."""
    from ..domain.enums import DefaultKeyBehavior, StylePreset

    if field == "default_key_behavior":
        if not isinstance(value, DefaultKeyBehavior):
            raise ConfigurationError("default_key_behavior must be auto or last_used")
    elif field == "default_style_preset":
        if not isinstance(value, StylePreset):
            raise ConfigurationError("default_style_preset must be Pop, Choir, or Barbershop")
    elif field == "latency_buffer_frame_size":
        validate_frame_size(value)  # type: ignore[arg-type]
    elif field == "reduced_motion":
        if not isinstance(value, bool):
            raise ConfigurationError("reduced_motion must be a Boolean")
    elif field in {"audio_input_device", "audio_output_device"}:
        if value is not None:
            _parse_audio_device_selector(value)
    else:
        raise ConfigurationError(f"setting cannot be changed in this phase: {field}")


def format_audio_device_selector(name: str, host_api_name: str) -> str:
    """Format the best-available sounddevice selector without persisting an index."""
    if not isinstance(name, str) or not isinstance(host_api_name, str):
        raise ConfigurationError("audio device name and host API must be strings")
    if not name.strip() or not host_api_name.strip() or name != name.strip() or host_api_name != host_api_name.strip():
        raise ConfigurationError("audio device name and host API must be non-empty and trimmed")
    return f"{name}, {host_api_name}"


def resolve_audio_device_selector(selector: str | None, devices: Iterable[object]) -> object | None:
    """Resolve an exact name/host-API pair; None explicitly selects system default."""
    if selector is None:
        return None
    name, host_api_name = _parse_audio_device_selector(selector)
    matches: list[object] = []
    for device in devices:
        if isinstance(device, Mapping):
            device_name = device.get("name")
            host_api = device.get("host_api_name", device.get("host_api"))
        else:
            device_name = getattr(device, "name", None)
            host_api = getattr(device, "host_api_name", getattr(device, "host_api", None))
        if device_name == name and host_api == host_api_name:
            matches.append(device)
    if not matches:
        raise DeviceUnavailableError(
            f"Saved audio device '{selector}' is unavailable; retry or explicitly select System default."
        )
    if len(matches) != 1:
        raise ConfigurationError(
            f"Saved audio device '{selector}' is ambiguous; select the intended device explicitly."
        )
    return matches[0]


def _parse_audio_device_selector(value: object) -> tuple[str, str]:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ConfigurationError("audio device selector must be a trimmed 'name, host API' string")
    name, separator, host_api = value.rpartition(", ")
    if not separator or not name or not host_api or name != name.strip() or host_api != host_api.strip():
        raise ConfigurationError("audio device selector must contain a non-empty name and host API")
    return name, host_api


def validate_upload_metadata(file_extension: str, duration_seconds: float) -> None:
    """Validate resolved upload constraints without decoding or processing audio."""
    extension = file_extension.lower().lstrip(".")
    if extension not in SUPPORTED_UPLOAD_FORMATS:
        raise ConfigurationError(f"unsupported upload format: {extension}")
    if duration_seconds < 0 or duration_seconds > MAX_UPLOAD_DURATION_SECONDS:
        raise ConfigurationError("upload duration must be between 0 and 600 seconds")


def validate_voice_name(voice_name: str) -> None:
    """Validate a core voice identifier without generating harmony."""
    if voice_name not in CORE_VOICES:
        raise ConfigurationError(f"unsupported core voice: {voice_name}")
