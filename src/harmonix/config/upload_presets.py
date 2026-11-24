"""Approved starting voice toggles for new Upload sessions."""

from ..domain.enums import StylePreset

UPLOAD_PRESET_VOICES = {
    StylePreset.POP: frozenset({"3rd", "5th"}),
    StylePreset.CHOIR: frozenset({"3rd", "5th", "octave"}),
}

# The octave voice remains the existing fixed octave. Choir has no independent
# octave spacing or style-specific interval rules in this phase.
