import unittest
from math import isfinite, pi, sin, sqrt

from harmonix.harmony_generator import HarmonyGenerator, HarmonyStatus
from harmonix.key_scale_resolver import KeyScale
from harmonix.pitch_detector import TrackingState


class HarmonyGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = HarmonyGenerator()
        self.c_major = KeyScale("C", "major")
        self.a_minor = KeyScale("A", "minor")

    @staticmethod
    def midi_frequency(midi: int) -> float:
        return 440.0 * 2.0 ** ((midi - 69) / 12.0)

    def test_c_major_diatonic_thirds_cover_all_scale_degrees(self) -> None:
        expected = [("C4", "E4"), ("D4", "F4"), ("E4", "G4"), ("F4", "A4"), ("G4", "B4"), ("A4", "C5"), ("B4", "D5")]

        for source, target in expected:
            source_midi = {"C4": 60, "D4": 62, "E4": 64, "F4": 65, "G4": 67, "A4": 69, "B4": 71}[source]
            result = self.generator.generate(self.midi_frequency(source_midi), self.c_major, "3rd")
            self.assertTrue(result.valid)
            self.assertEqual(result.target_pitch, target)
            self.assertEqual(result.status, HarmonyStatus.VALID)

    def test_c_major_fifths_cover_all_scale_degrees(self) -> None:
        expected = [("C4", "G4"), ("D4", "A4"), ("E4", "B4"), ("F4", "C5"), ("G4", "D5"), ("A4", "E5"), ("B4", "F5")]

        for source, target in expected:
            source_midi = {"C4": 60, "D4": 62, "E4": 64, "F4": 65, "G4": 67, "A4": 69, "B4": 71}[source]
            result = self.generator.generate(self.midi_frequency(source_midi), self.c_major, "5th")
            self.assertEqual(result.target_pitch, target)

    def test_c_major_octave_behavior(self) -> None:
        result = self.generator.generate(self.midi_frequency(60), self.c_major, "octave")

        self.assertEqual(result.source_pitch, "C4")
        self.assertEqual(result.target_pitch, "C5")
        self.assertEqual(result.target_scale_degree, 1)

    def test_non_c_major_key_changes_diatonic_target(self) -> None:
        result = self.generator.generate(self.midi_frequency(60), KeyScale("F", "major"), "3rd")

        self.assertEqual(result.target_pitch, "E4")

    def test_minor_scale_behavior(self) -> None:
        result = self.generator.generate(self.midi_frequency(69), self.a_minor, "3rd")

        self.assertEqual(result.source_pitch, "A4")
        self.assertEqual(result.target_pitch, "C5")

    def test_multiple_source_octaves_are_preserved(self) -> None:
        lower = self.generator.generate(self.midi_frequency(48), self.c_major, "3rd")
        upper = self.generator.generate(self.midi_frequency(72), self.c_major, "3rd")

        self.assertEqual(lower.target_pitch, "E3")
        self.assertEqual(upper.target_pitch, "E5")

    def test_scale_degree_wraparound_crosses_octave(self) -> None:
        result = self.generator.generate(self.midi_frequency(71), self.c_major, "3rd")

        self.assertEqual(result.source_scale_degree, 7)
        self.assertEqual(result.target_pitch, "D5")
        self.assertEqual(result.target_scale_degree, 2)

    def test_missing_and_unvoiced_input_are_invalid(self) -> None:
        missing = self.generator.generate(None, self.c_major, "3rd")
        unvoiced = self.generator.generate(self.midi_frequency(60), self.c_major, "3rd", voiced=False)
        tracking_lost = self.generator.generate(
            self.midi_frequency(60),
            self.c_major,
            "3rd",
            tracking_state=TrackingState.TRACKING_LOST,
        )

        self.assertEqual(missing.status, HarmonyStatus.MISSING_FUNDAMENTAL)
        self.assertEqual(unvoiced.status, HarmonyStatus.UNVOICED)
        self.assertEqual(tracking_lost.status, HarmonyStatus.TRACKING_LOST)
        self.assertFalse(missing.valid)
        self.assertIsNone(missing.target_frequency_hz)

    def test_missing_ambiguous_and_invalid_key_inputs_are_invalid(self) -> None:
        missing = self.generator.generate(self.midi_frequency(60), None, "3rd")
        ambiguous = self.generator.generate(self.midi_frequency(60), self.c_major, "3rd", key_ambiguous=True)
        invalid_scale = self.generator.generate(self.midi_frequency(60), KeyScale("C", "dorian"), "3rd")
        invalid_voice = self.generator.generate(self.midi_frequency(60), self.c_major, "4th")

        self.assertEqual(missing.status, HarmonyStatus.MISSING_KEY)
        self.assertEqual(ambiguous.status, HarmonyStatus.AMBIGUOUS_KEY)
        self.assertEqual(invalid_scale.status, HarmonyStatus.INVALID_SCALE)
        self.assertEqual(invalid_voice.status, HarmonyStatus.INVALID_VOICE)
        self.assertTrue(all(not result.valid for result in (missing, ambiguous, invalid_scale, invalid_voice)))

    def test_extreme_valid_targets_are_clamped(self) -> None:
        low = self.generator.generate(20.0, self.c_major, "3rd")
        high = self.generator.generate(900.0, self.c_major, "octave")

        self.assertTrue(low.valid)
        self.assertTrue(high.valid)
        self.assertTrue(low.was_clamped)
        self.assertTrue(high.was_clamped)
        self.assertEqual(low.target_frequency_hz, 65.0)
        self.assertEqual(high.target_frequency_hz, 1000.0)

    def test_nonfinite_input_is_invalid_without_nonfinite_output(self) -> None:
        for frequency in (float("nan"), float("inf"), float("-inf")):
            result = self.generator.generate(frequency, self.c_major, "3rd")
            self.assertEqual(result.status, HarmonyStatus.NONFINITE_FREQUENCY)
            self.assertFalse(result.valid)
            self.assertIsNone(result.target_frequency_hz)

    def test_target_generation_is_deterministic(self) -> None:
        first = self.generator.generate(440.0, self.c_major, "3rd")
        second = self.generator.generate(440.0, self.c_major, "3rd")

        self.assertEqual(first, second)
        self.assertTrue(isfinite(first.target_frequency_hz))

    def test_harmony_is_not_one_fixed_chromatic_offset(self) -> None:
        c_third = self.generator.generate(self.midi_frequency(60), self.c_major, "3rd")
        d_third = self.generator.generate(self.midi_frequency(62), self.c_major, "3rd")

        c_semitones = round(12.0 * (c_third.target_frequency_hz / c_third.source_frequency_hz).bit_length(), 6) if False else None
        self.assertEqual(c_third.target_pitch, "E4")
        self.assertEqual(d_third.target_pitch, "F4")
        self.assertNotEqual(c_third.target_pitch, d_third.target_pitch)
        self.assertNotAlmostEqual(
            c_third.target_frequency_hz / c_third.source_frequency_hz,
            d_third.target_frequency_hz / d_third.source_frequency_hz,
            places=4,
        )

    def test_changing_effective_key_changes_target(self) -> None:
        c_major_target = self.generator.generate(self.midi_frequency(67), KeyScale("C", "major"), "3rd")
        f_major_target = self.generator.generate(self.midi_frequency(67), KeyScale("F", "major"), "3rd")

        self.assertEqual(c_major_target.target_pitch, "B4")
        self.assertEqual(f_major_target.target_pitch, "B-4")
        self.assertNotEqual(c_major_target.target_frequency_hz, f_major_target.target_frequency_hz)

    def test_source_cents_deviation_is_preserved(self) -> None:
        source = self.midi_frequency(60) * 2.0 ** (25.0 / 1200.0)
        result = self.generator.generate(source, self.c_major, "3rd")
        expected = self.midi_frequency(64) * 2.0 ** (25.0 / 1200.0)

        self.assertAlmostEqual(result.target_frequency_hz, expected, places=6)


if __name__ == "__main__":
    unittest.main()
