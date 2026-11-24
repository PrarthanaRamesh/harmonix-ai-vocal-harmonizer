import unittest
from types import SimpleNamespace

import numpy as np

from harmonix.latency import LatencyMeasurement, LatencyProvenance
from m8_hardware_validation import (
    ConditionHarmonyGenerator,
    OnsetDetector,
    ShiftMode,
    build_condition_matrix,
    classify_measurement,
    pair_onsets_with_measurements,
    shift_ratio,
    validate_trial_count,
)


class FakeHarmonyGenerator:
    def generate(self, frequency, key, voice, **kwargs):
        return SimpleNamespace(
            valid=True,
            target_frequency_hz=frequency * 1.5,
            was_clamped=False,
            source_frequency_hz=frequency,
        )


class HardwareValidationTests(unittest.TestCase):
    def test_condition_matrix_covers_zero_one_and_three_voice_cases(self) -> None:
        conditions = build_condition_matrix(5.0)

        self.assertEqual(len(conditions), 9)
        self.assertEqual(conditions[0].voice_count, 0)
        self.assertIsNone(conditions[0].shift_mode)
        one_voice = [item for item in conditions if item.voice_count == 1]
        three_voices = [item for item in conditions if item.voice_count == 3]
        self.assertEqual({item.shift_mode for item in one_voice}, set(ShiftMode))
        self.assertEqual({item.shift_mode for item in three_voices}, set(ShiftMode))
        self.assertEqual(one_voice[0].voices, ("3rd",))
        self.assertEqual(three_voices[0].voices, ("3rd", "5th", "octave"))

    def test_shift_ratios_are_explicit(self) -> None:
        self.assertEqual(shift_ratio(ShiftMode.ZERO, 5.0), 1.0)
        self.assertEqual(shift_ratio(ShiftMode.OCTAVE, 5.0), 2.0)
        self.assertAlmostEqual(shift_ratio(ShiftMode.DOWNWARD, 12.0), 0.5)
        with self.assertRaises(ValueError):
            shift_ratio(ShiftMode.DOWNWARD, 25.0)

    def test_zero_shift_overrides_target_to_source_frequency(self) -> None:
        generator = ConditionHarmonyGenerator(FakeHarmonyGenerator(), ShiftMode.ZERO, 5.0)

        target = generator.generate(440.0, object(), "3rd")

        self.assertEqual(target.target_frequency_hz, 440.0)
        self.assertTrue(target.valid)
        self.assertFalse(target.was_clamped)

    def test_diatonic_shift_preserves_real_generator_target(self) -> None:
        generator = ConditionHarmonyGenerator(FakeHarmonyGenerator(), ShiftMode.DIATONIC, 5.0)

        target = generator.generate(440.0, object(), "3rd")

        self.assertEqual(target.target_frequency_hz, 660.0)

    def test_onset_detector_requires_separated_above_threshold_bursts(self) -> None:
        detector = OnsetDetector(sample_rate=1000, threshold=0.1, minimum_gap_ms=20, window_ms=10)
        signal = np.concatenate(
            (
                np.zeros(20, dtype=np.float32),
                np.ones(20, dtype=np.float32) * 0.5,
                np.zeros(20, dtype=np.float32),
                np.ones(20, dtype=np.float32) * 0.5,
            )
        )

        onsets = detector.feed(signal)

        self.assertEqual(onsets, (20, 60))

    def test_invalid_capture_timestamp_never_becomes_physical_latency(self) -> None:
        measurement = LatencyMeasurement(
            trial=1,
            capture_timestamp=None,
            output_timestamp=10.1,
            pipeline_timestamp=10.0,
            provenance=LatencyProvenance.PIPELINE_CALLBACK_TO_DAC,
            sample_start=0,
            sample_count=4096,
            active_voice_count=1,
            mode="live",
            input_device="mic",
            output_device="speaker",
            sample_rate=44100,
            block_size=2048,
            processing_segment_size=8192,
        )

        result = classify_measurement(measurement)

        self.assertIsNone(result["physical_capture_to_dac_latency_ms"])
        self.assertEqual(result["physical_capture_to_dac_status"], "unavailable")
        self.assertAlmostEqual(result["pipeline_latency_ms"], 100.0)

    def test_verified_capture_timestamp_can_produce_physical_latency(self) -> None:
        measurement = LatencyMeasurement(
            trial=1,
            capture_timestamp=10.0,
            output_timestamp=10.1,
            pipeline_timestamp=10.05,
            provenance=LatencyProvenance.VERIFIED_CAPTURE_TO_DAC,
            sample_start=0,
            sample_count=4096,
            active_voice_count=1,
            mode="live",
            input_device="mic",
            output_device="speaker",
            sample_rate=44100,
            block_size=2048,
            processing_segment_size=8192,
        )

        result = classify_measurement(measurement)

        self.assertAlmostEqual(result["physical_capture_to_dac_latency_ms"], 100.0)
        self.assertAlmostEqual(result["pipeline_latency_ms"], 50.0)

    def test_onset_pairing_marks_missing_output_record(self) -> None:
        records = pair_onsets_with_measurements("voices_1__zero", [100], [], 20)

        self.assertEqual(records[0].status, "measurement_missing")
        self.assertIsNone(records[0].measurement)

    def test_trial_count_must_meet_master_minimum(self) -> None:
        validate_trial_count(20)
        with self.assertRaises(ValueError):
            validate_trial_count(19)


if __name__ == "__main__":
    unittest.main()