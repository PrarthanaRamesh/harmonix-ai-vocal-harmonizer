import unittest
from unittest.mock import patch

import librosa
import numpy as np

from harmonix.pitch_detector import (
    CANONICAL_SAMPLE_RATE,
    CONFIDENCE_THRESHOLD,
    FMAX_HZ,
    FMIN_HZ,
    FRAME_SIZE,
    PitchDetector,
    TrackingState,
)


class PitchDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = PitchDetector()
        self.time = np.arange(FRAME_SIZE, dtype=np.float32) / CANONICAL_SAMPLE_RATE

    def test_440_hz_sine_is_detected(self) -> None:
        frame = np.sin(2.0 * np.pi * 440.0 * self.time).astype(np.float32)

        result = self.detector.detect(frame)

        self.assertIsNotNone(result.frequency_hz)
        self.assertAlmostEqual(result.frequency_hz, 440.0, delta=8.0)
        self.assertGreaterEqual(result.confidence, CONFIDENCE_THRESHOLD)
        self.assertTrue(result.voiced)
        self.assertEqual(result.tracking_state, TrackingState.TRACKING)

    def test_silence_is_unvoiced_without_fake_pitch(self) -> None:
        result = self.detector.detect(np.zeros(FRAME_SIZE, dtype=np.float32))

        self.assertIsNone(result.frequency_hz)
        self.assertFalse(result.voiced)
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(result.tracking_state, TrackingState.TRACKING_LOST)

    def test_white_noise_is_not_confidently_pitched(self) -> None:
        noise = np.random.default_rng(20260921).standard_normal(FRAME_SIZE).astype(np.float32)

        result = self.detector.detect(noise)

        self.assertTrue(
            result.frequency_hz is None
            or not result.voiced
            or result.confidence < CONFIDENCE_THRESHOLD
        )

    def test_changing_pitch_produces_changing_results(self) -> None:
        first_frame = np.sin(2.0 * np.pi * 220.0 * self.time).astype(np.float32)
        second_frame = np.sin(2.0 * np.pi * 440.0 * self.time).astype(np.float32)

        first = self.detector.detect(first_frame)
        second = self.detector.detect(second_frame)

        self.assertIsNotNone(first.frequency_hz)
        self.assertIsNotNone(second.frequency_hz)
        self.assertAlmostEqual(first.frequency_hz, 220.0, delta=8.0)
        self.assertAlmostEqual(second.frequency_hz, 440.0, delta=8.0)
        self.assertGreater(abs(second.frequency_hz - first.frequency_hz), 100.0)
        self.assertGreaterEqual(first.confidence, CONFIDENCE_THRESHOLD)
        self.assertGreaterEqual(second.confidence, CONFIDENCE_THRESHOLD)

    def test_nonfinite_input_is_unvoiced_without_nonfinite_output(self) -> None:
        frame = np.zeros(FRAME_SIZE, dtype=np.float32)
        frame[100] = np.nan
        frame[200] = np.inf

        result = self.detector.detect(frame)

        self.assertIsNone(result.frequency_hz)
        self.assertTrue(np.isfinite(result.confidence))
        self.assertFalse(result.voiced)
        self.assertEqual(result.tracking_state, TrackingState.TRACKING_LOST)

    def test_out_of_range_pyin_frequency_is_rejected(self) -> None:
        f0 = np.array([50.0, 50.0], dtype=np.float64)
        voiced = np.array([True, True])
        confidence = np.array([0.99, 0.99], dtype=np.float64)

        with patch.object(librosa, "pyin", return_value=(f0, voiced, confidence)):
            result = self.detector.detect(np.zeros(FRAME_SIZE, dtype=np.float32))

        self.assertIsNone(result.frequency_hz)
        self.assertFalse(result.voiced)
        self.assertEqual(result.tracking_state, TrackingState.TRACKING_LOST)

    def test_low_confidence_pyin_result_is_tracking_lost(self) -> None:
        f0 = np.array([440.0, 440.0], dtype=np.float64)
        voiced = np.array([True, True])
        confidence = np.array([0.49, 0.49], dtype=np.float64)

        with patch.object(librosa, "pyin", return_value=(f0, voiced, confidence)):
            result = self.detector.detect(np.zeros(FRAME_SIZE, dtype=np.float32))

        self.assertIsNone(result.frequency_hz)
        self.assertFalse(result.voiced)
        self.assertLess(result.confidence, CONFIDENCE_THRESHOLD)
        self.assertEqual(result.tracking_state, TrackingState.TRACKING_LOST)

    def test_detector_uses_canonical_input_contract(self) -> None:
        self.assertEqual(self.detector.sample_rate, CANONICAL_SAMPLE_RATE)
        self.assertEqual(self.detector.frame_size, FRAME_SIZE)
        self.assertEqual(self.detector.fmin_hz, FMIN_HZ)
        self.assertEqual(self.detector.fmax_hz, FMAX_HZ)

        with self.assertRaises(ValueError):
            self.detector.detect(np.zeros(FRAME_SIZE - 1, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
