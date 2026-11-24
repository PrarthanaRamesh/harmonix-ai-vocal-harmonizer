"""JSON persistence for app settings and recent-session metadata."""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Mapping, Sequence

try:
    import msvcrt
except ImportError:  # pragma: no cover - Windows-only lock implementation fallback.
    msvcrt = None

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback.
    fcntl = None

from ..config.models import (
    APPDATA_DIR_NAME,
    MAX_RECENT_SESSIONS,
    LOCK_FILE_NAME,
    RECENT_SESSIONS_FILE_NAME,
    SETTINGS_FILE_NAME,
)
from ..config.validation import validate_frame_size
from ..domain.errors import PersistenceLockError, PersistenceReadError, PersistenceWriteError
from ..domain.enums import SessionMode
from ..domain.models import GlobalSettings, RecentSessionSummary


def _recent_session_timestamp_key(value: str | None) -> tuple[int, datetime]:
    """Order valid timezone-aware ISO timestamps by instant; unknown times sort last."""
    if value is None:
        return (0, datetime.min.replace(tzinfo=timezone.utc))
    try:
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return (0, datetime.min.replace(tzinfo=timezone.utc))
        return (1, parsed.astimezone(timezone.utc))
    except (OverflowError, ValueError):
        # The existing metadata contract accepts timestamp strings without
        # validating their format. Preserve them, but don't let them break
        # loading or outrank timestamps whose instant is known.
        return (0, datetime.min.replace(tzinfo=timezone.utc))


@dataclass(frozen=True)
class RecentSessionMetadata:
    source_file: str | None = None
    key_root: str | None = None
    key_scale_type: str | None = None
    voices: tuple[str, ...] = ()
    style_preset: str | None = None
    last_used_at: str | None = None
    id: str | None = None
    display_name: str | None = None
    mode: str = SessionMode.UPLOAD.value
    voice_toggles: tuple[tuple[str, bool], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.id or self._stable_legacy_id(),
            "display_name": self.display_name or self._default_display_name(),
            "mode": self.mode,
            "source_file": self.source_file,
            "key_root": self.key_root,
            "key_scale_type": self.key_scale_type,
            "voices": list(self.voices),
            "voice_toggles": dict(self.voice_toggles),
            "style_preset": self.style_preset,
            "last_used_at": self.last_used_at or datetime.now(timezone.utc).isoformat(),
        }
        return payload

    def to_summary(self) -> RecentSessionSummary:
        toggles = dict(self.voice_toggles)
        for voice in self.voices:
            toggles.setdefault(voice, True)
        return RecentSessionSummary(
            id=self.id or self._stable_legacy_id(),
            display_name=self.display_name or self._default_display_name(),
            mode=SessionMode(self.mode),
            source_file_path=self.source_file,
            last_used_key_root=self.key_root,
            last_used_key_scale_type=self.key_scale_type,
            voice_toggles=tuple(sorted(toggles.items())),
            style_preset=self.style_preset,
            timestamp=self.last_used_at or "",
        )

    def _default_display_name(self) -> str:
        if self.display_name:
            return self.display_name
        if self.source_file:
            return Path(self.source_file).name
        timestamp = self.last_used_at or "unknown date"
        return f"Live session — {timestamp}"

    def _stable_legacy_id(self) -> str:
        canonical = json.dumps(
            {
                "source_file": self.source_file,
                "key_root": self.key_root,
                "key_scale_type": self.key_scale_type,
                "voices": list(self.voices),
                "style_preset": self.style_preset,
                "last_used_at": self.last_used_at,
                "mode": self.mode,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return str(uuid.uuid5(uuid.NAMESPACE_URL, canonical))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "RecentSessionMetadata | None":
        source_file = payload.get("source_file")
        if source_file is not None and (not isinstance(source_file, str) or not source_file.strip()):
            return None
        mode = payload.get("mode", SessionMode.UPLOAD.value)
        if not isinstance(mode, str) or mode not in {item.value for item in SessionMode}:
            return None
        if mode == SessionMode.UPLOAD.value and not source_file:
            return None
        key_root = payload.get("key_root")
        key_scale_type = payload.get("key_scale_type")
        style_preset = payload.get("style_preset")
        last_used_at = payload.get("last_used_at")
        record_id = payload.get("id")
        display_name = payload.get("display_name")
        if any(value is not None and not isinstance(value, str) for value in (key_root, key_scale_type, style_preset, last_used_at, record_id, display_name)):
            return None
        if record_id is not None and not record_id.strip() or display_name is not None and not display_name.strip():
            return None
        voices_value = payload.get("voices", ())
        if isinstance(voices_value, str):
            voices = (voices_value,)
        elif isinstance(voices_value, (list, tuple)):
            if any(not isinstance(voice, str) for voice in voices_value):
                return None
            voices = tuple(voices_value)
        else:
            return None
        toggles_value = payload.get("voice_toggles", {})
        if not isinstance(toggles_value, Mapping) or any(
            not isinstance(voice, str) or not isinstance(enabled, bool)
            for voice, enabled in toggles_value.items()
        ):
            return None
        return cls(
            source_file=source_file.strip() if source_file else None,
            key_root=key_root,
            key_scale_type=key_scale_type,
            voices=voices,
            style_preset=style_preset,
            last_used_at=last_used_at,
            id=record_id,
            display_name=display_name,
            mode=mode,
            voice_toggles=tuple(sorted(toggles_value.items())),
        )


class JSONPersistenceStore:
    """Persist app settings and recent-session metadata in the Windows local app-data directory."""

    def __init__(self, base_dir: str | os.PathLike[str] | None = None) -> None:
        if base_dir is None:
            local_appdata = os.environ.get("LOCALAPPDATA")
            if local_appdata:
                base_path = Path(local_appdata) / APPDATA_DIR_NAME
            else:
                base_path = Path.home() / "AppData" / "Local" / APPDATA_DIR_NAME
        else:
            resolved = Path(base_dir)
            if resolved.name == APPDATA_DIR_NAME:
                base_path = resolved
            else:
                base_path = resolved / APPDATA_DIR_NAME
        self.base_dir = base_path
        self.settings_path = self.base_dir / SETTINGS_FILE_NAME
        self.recent_sessions_path = self.base_dir / RECENT_SESSIONS_FILE_NAME
        self.lock_path = self.base_dir / LOCK_FILE_NAME
        self._lock = RLock()
        self._lifetime_lock_handle = None
        self._corrupt_paths: set[Path] = set()
        self._unreadable_paths: set[Path] = set()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    @property
    def settings_warning(self) -> str | None:
        if self.settings_path in self._unreadable_paths:
            return "Settings could not be read. Defaults are in use; close the app and check the settings file or its permissions."
        return None

    def load_settings(self, defaults: GlobalSettings | None = None) -> GlobalSettings:
        settings = defaults or GlobalSettings()
        try:
            payload = self._load_json(self.settings_path)
        except FileNotFoundError:
            return settings
        except (json.JSONDecodeError, UnicodeError):
            self._unreadable_paths.discard(self.settings_path)
            self._corrupt_paths.add(self.settings_path)
            return settings
        except OSError:
            self._unreadable_paths.add(self.settings_path)
            return settings
        except (TypeError, ValueError):
            self._corrupt_paths.add(self.settings_path)
            return settings

        self._unreadable_paths.discard(self.settings_path)
        try:
            if not isinstance(payload, dict):
                self._corrupt_paths.add(self.settings_path)
                return settings
            if not self._is_valid_settings_payload(payload):
                self._corrupt_paths.add(self.settings_path)
                return settings
            for field, value in payload.items():
                if field == "audio_input_device" and value is not None:
                    settings.audio_input_device = str(value)
                elif field == "audio_output_device" and value is not None:
                    settings.audio_output_device = str(value)
                elif field == "default_key_behavior" and isinstance(value, str):
                    settings.default_key_behavior = settings.default_key_behavior.__class__(value)
                elif field == "default_style_preset" and isinstance(value, str):
                    settings.default_style_preset = settings.default_style_preset.__class__(value)
                elif field == "latency_buffer_frame_size" and isinstance(value, int):
                    settings.latency_buffer_frame_size = value
                elif field == "export_default_format" and isinstance(value, str):
                    settings.export_default_format = value
                elif field == "export_default_quality" and value is not None:
                    settings.export_default_quality = str(value)
                elif field == "reduced_motion" and isinstance(value, bool):
                    settings.reduced_motion = value
            return settings
        except json.JSONDecodeError:
            self._corrupt_paths.add(self.settings_path)
            return settings
        except (TypeError, ValueError):
            self._corrupt_paths.add(self.settings_path)
            return settings

    def save_settings(self, settings: GlobalSettings) -> None:
        payload = {
            "audio_input_device": settings.audio_input_device,
            "audio_output_device": settings.audio_output_device,
            "default_key_behavior": settings.default_key_behavior.value,
            "default_style_preset": settings.default_style_preset.value,
            "latency_buffer_frame_size": settings.latency_buffer_frame_size,
            "export_default_format": settings.export_default_format,
            "export_default_quality": settings.export_default_quality,
            "reduced_motion": settings.reduced_motion,
        }
        self._write_json(self.settings_path, payload)

    def load_recent_sessions(self) -> list[RecentSessionMetadata]:
        try:
            payload = self._load_json(self.recent_sessions_path)
        except FileNotFoundError:
            return []
        except (json.JSONDecodeError, UnicodeError):
            return []
        except OSError:
            self._unreadable_paths.add(self.recent_sessions_path)
            return []

        if not isinstance(payload, list):
            self._corrupt_paths.add(self.recent_sessions_path)
            return []

        sessions: list[RecentSessionMetadata] = []
        for item in payload:
            if not isinstance(item, Mapping):
                self._corrupt_paths.add(self.recent_sessions_path)
                return []
            metadata = RecentSessionMetadata.from_mapping(item)
            if metadata is None:
                self._corrupt_paths.add(self.recent_sessions_path)
                return []
            if metadata.mode == SessionMode.UPLOAD.value and (
                not metadata.source_file or not Path(metadata.source_file).exists()
            ):
                continue
            sessions.append(metadata)
        sessions.sort(key=lambda session: _recent_session_timestamp_key(session.last_used_at), reverse=True)
        return sessions[:MAX_RECENT_SESSIONS]

    def save_recent_sessions(self, sessions: Sequence[str | Mapping[str, Any] | RecentSessionMetadata]) -> list[RecentSessionMetadata]:
        cleaned: list[RecentSessionMetadata] = []
        for session in sessions:
            if isinstance(session, str):
                source_file = session.strip()
                if not source_file or not Path(source_file).exists():
                    continue
                cleaned.append(RecentSessionMetadata(source_file=source_file, last_used_at=datetime.now(timezone.utc).isoformat()))
            elif isinstance(session, RecentSessionMetadata):
                if session.mode == SessionMode.LIVE.value or (
                    session.source_file and Path(session.source_file).exists()
                ):
                    cleaned.append(session)
            elif isinstance(session, Mapping):
                metadata = RecentSessionMetadata.from_mapping(session)
                if metadata is not None and (
                    metadata.mode == SessionMode.LIVE.value
                    or metadata.source_file and Path(metadata.source_file).exists()
                ):
                    cleaned.append(metadata)

        cleaned.sort(key=lambda session: _recent_session_timestamp_key(session.last_used_at), reverse=True)
        trimmed = cleaned[:MAX_RECENT_SESSIONS]
        self._write_json(self.recent_sessions_path, [session.to_dict() for session in trimmed])
        return trimmed

    def is_single_instance_active(self) -> bool:
        try:
            with self._locked_file():
                return True
        except PersistenceLockError:
            return False

    def acquire_lifetime_lock(self) -> None:
        """Acquire and retain the cross-process lock until explicit shutdown."""
        with self._lock:
            if self._lifetime_lock_handle is not None:
                return
            try:
                handle = self.lock_path.open("a+b")
            except OSError as exc:
                raise PersistenceLockError("unable to open the application lock file") from exc
            try:
                self._acquire_lock(handle)
            except Exception:
                handle.close()
                raise
            self._lifetime_lock_handle = handle

    def release_lifetime_lock(self) -> None:
        with self._lock:
            handle = self._lifetime_lock_handle
            if handle is None:
                return
            self._lifetime_lock_handle = None
            self._release_lock(handle)
            handle.close()

    def _load_json(self, path: Path) -> Any:
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (json.JSONDecodeError, UnicodeError):
            self._corrupt_paths.add(path)
            raise

    @staticmethod
    def _is_valid_settings_payload(payload: Mapping[str, Any]) -> bool:
        """Check known persisted fields against the documented settings data types."""
        for field, value in payload.items():
            if field in {"audio_input_device", "audio_output_device"}:
                if value is not None and not isinstance(value, str):
                    return False
            elif field == "default_key_behavior":
                if not isinstance(value, str):
                    return False
                try:
                    GlobalSettings().default_key_behavior.__class__(value)
                except ValueError:
                    return False
            elif field == "default_style_preset":
                if value is not None:
                    if not isinstance(value, str):
                        return False
                    try:
                        GlobalSettings().default_style_preset.__class__(value)
                    except ValueError:
                        return False
            elif field == "latency_buffer_frame_size":
                if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
                    return False
                if value is not None:
                    try:
                        validate_frame_size(value)
                    except ValueError:
                        return False
            elif field == "export_default_format":
                if value not in {"wav", "mp3"}:
                    return False
            elif field == "export_default_quality":
                if value is not None and value not in {"high", "medium", "low"}:
                    return False
            elif field == "reduced_motion" and not isinstance(value, bool):
                return False
        return True

    @contextmanager
    def _locked_file(self):
        try:
            file_handle = self.lock_path.open("a+b")
        except OSError as exc:
            raise PersistenceLockError("unable to open the application lock file") from exc
        acquired = False
        try:
            self._acquire_lock(file_handle)
            acquired = True
            yield
        finally:
            if acquired:
                self._release_lock(file_handle)
            file_handle.close()

    def _acquire_lock(self, handle: Any) -> None:
        if os.name == "nt" and msvcrt is not None:
            try:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                return
            except OSError as exc:
                raise PersistenceLockError("execution lock already held") from exc
        if fcntl is not None:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except OSError as exc:
                raise PersistenceLockError("execution lock already held") from exc
        raise PersistenceLockError("execution lock unavailable on this platform")

    def _release_lock(self, handle: Any) -> None:
        try:
            if os.name == "nt" and msvcrt is not None:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            elif fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass

    def _write_json(self, path: Path, payload: Any) -> None:
        try:
            with self._lock:
                if self._lifetime_lock_handle is None:
                    lock_context = self._locked_file()
                else:
                    lock_context = _null_context()
                with lock_context:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path == self.settings_path:
                        self._check_settings_readable_before_save(path)
                    elif path == self.recent_sessions_path:
                        self._check_recent_sessions_readable_before_save(path)
                    if path in self._corrupt_paths and path.exists():
                        self._preserve_corrupt_file(path)
                    self._atomic_write(path, payload)
                    self._corrupt_paths.discard(path)
        except PersistenceWriteError:
            raise
        except PersistenceLockError:
            raise
        except (OSError, TypeError, ValueError) as exc:
            raise PersistenceWriteError(f"unable to persist JSON to {path}") from exc

    def _check_settings_readable_before_save(self, path: Path) -> None:
        try:
            with path.open("rb") as handle:
                raw = handle.read()
        except FileNotFoundError:
            if path in self._unreadable_paths:
                raise PersistenceReadError("previously unreadable settings file is now missing; refusing to replace it")
            return
        except OSError as exc:
            self._unreadable_paths.add(path)
            raise PersistenceReadError("settings file became unreadable; refusing to replace it") from exc

        self._unreadable_paths.discard(path)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeError):
            self._corrupt_paths.add(path)
            return
        if not isinstance(payload, dict) or not self._is_valid_settings_payload(payload):
            self._corrupt_paths.add(path)

    def _check_recent_sessions_readable_before_save(self, path: Path) -> None:
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            if path in self._unreadable_paths:
                raise PersistenceReadError("previously unreadable recent-session file is now missing; refusing to replace it")
            return
        except OSError as exc:
            self._unreadable_paths.add(path)
            raise PersistenceReadError("recent-session file became unreadable; refusing to replace it") from exc

        self._unreadable_paths.discard(path)
        try:
            existing = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeError):
            self._corrupt_paths.add(path)
            return
        if not isinstance(existing, list) or any(
            not isinstance(item, Mapping) or RecentSessionMetadata.from_mapping(item) is None
            for item in existing
        ):
            self._corrupt_paths.add(path)

    def _preserve_corrupt_file(self, path: Path) -> Path:
        try:
            original = path.read_bytes()
            for _ in range(10):
                backup = path.with_name(f"{path.name}.corrupt-{uuid.uuid4().hex}.bak")
                try:
                    with backup.open("xb") as handle:
                        handle.write(original)
                        handle.flush()
                        os.fsync(handle.fileno())
                    return backup
                except FileExistsError:
                    continue
                except OSError:
                    try:
                        backup.unlink(missing_ok=True)
                    except OSError:
                        pass
                    raise
            raise FileExistsError("unable to choose a unique corrupt metadata backup name")
        except OSError as exc:
            raise PersistenceWriteError(f"unable to preserve corrupt metadata at {path}") from exc

    def _atomic_write(self, path: Path, payload: Any) -> None:
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", suffix=".tmp", delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
        except Exception as exc:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise PersistenceWriteError(f"unable to persist JSON to {path}") from exc


class _null_context:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


DEFAULT_APPDATA_DIR = (
    Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / APPDATA_DIR_NAME
)
