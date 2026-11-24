"""Distinct failures for upload loading and analysis."""


class UploadError(Exception):
    """Base error for an upload that cannot produce a complete result."""


class UnsupportedUploadFormatError(UploadError):
    """The selected file extension is not supported."""


class MP3DecoderUnavailableError(UploadError):
    """The installed locked environment cannot decode MP3 data."""


class InvalidUploadDurationError(UploadError):
    """The decoded audio has no valid sample duration or sample rate."""


class UploadDurationExceededError(UploadError):
    """The decoded audio exceeds the ten-minute upload limit."""


class CorruptUploadError(UploadError):
    """The file cannot be read or decoded as audio."""


class SilentOrUnusableAudioError(UploadError):
    """No usable voiced pitch observations were produced for the upload."""


class AmbiguousUploadKeyError(UploadError):
    """M5 marked the complete-track key evidence ambiguous."""


class InsufficientUploadKeyEvidenceError(UploadError):
    """M5 reported insufficient complete-track key evidence."""


class UploadCancelledError(UploadError):
    """The upload worker observed and completed a cancellation request."""


class UploadPitchShiftError(UploadError):
    """M7 could not render one required voiced target region."""


class UploadAnalysisError(UploadError):
    """An unexpected upload analysis stage failed."""