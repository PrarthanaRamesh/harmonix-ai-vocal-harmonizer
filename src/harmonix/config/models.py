"""Specification-backed constants that do not resolve later OPEN decisions."""

SUPPORTED_UPLOAD_FORMATS = ("wav", "mp3", "flac")
MAX_UPLOAD_DURATION_SECONDS = 10 * 60
CORE_VOICES = ("3rd", "5th", "octave")
MAX_RECENT_SESSIONS = 5
DEFAULT_EXPORT_FORMAT = "wav"
APPDATA_DIR_NAME = "HarmonixV2"
SETTINGS_FILE_NAME = "settings.json"
RECENT_SESSIONS_FILE_NAME = "recent_sessions.json"
LOCK_FILE_NAME = ".harmonix.lock"