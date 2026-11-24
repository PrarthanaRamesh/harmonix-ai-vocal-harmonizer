"""Stable domain enumerations used by the application state."""

from enum import Enum


class Screen(str, Enum):
    HOME = "home"
    LIVE_CONFIGURE = "live_configure"
    LIVE_SESSION = "live_session"
    UPLOAD_SELECT = "upload_select"
    UPLOAD_ANALYZING = "upload_analyzing"
    UPLOAD_CONFIGURE = "upload_configure"
    UPLOAD_PLAYBACK = "upload_playback"
    REVIEW_EXPORT = "review_export"
    SETTINGS = "settings"


class SessionMode(str, Enum):
    LIVE = "live"
    UPLOAD = "upload"


class DefaultKeyBehavior(str, Enum):
    AUTO = "auto"
    LAST_USED = "last_used"


class StylePreset(str, Enum):
    POP = "pop"
    CHOIR = "choir"
    BARBERSHOP = "barbershop"