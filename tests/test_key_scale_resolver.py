import unittest
from math import pow

from harmonix.key_scale_resolver import (
    AnalysisMode,
    KeyScale,
    KeyScaleResolver,
)
from harmonix.pitch_detector import PitchDetectionResult, TrackingState


class KeyScaleResolverTests(unittest.TestCase):
    @staticmethod
    def pitch_class_hz(pitch_class: int) -> float:
        return 440.0 * pow(2.0, (pitch_class - 9) / 12.0)

    @classmethod
    def observation(cls, pitch_class: int, confidence: float = 0.9) -> PitchDetectionResult:
        return PitchDetectionResult(
            frequency_hz=cls.pitch_class_hz(pitch_class),
            confidence=confidence,
            voiced=True,
            tracking_state=TrackingState.TRACKING,
        )

    @classmethod
    def c_major_material(cls) -> list[PitchDetectionResult]:
        pitch_classes = [0] * 8 + [2, 4, 5, 7, 9, 11] + [0] * 4 + [7, 4] * 3
        return [cls.observation(pitch_class) for pitch_class in pitch_classes]

    @classmethod
    def a_minor_material(cls) -> list[PitchDetectionResult]:
        pitch_classes = [9] * 8 + [11, 0, 2, 4, 5, 7] + [9] * 4 + [4, 0] * 3
        return [cls.observation(pitch_class) for pitch_class in pitch_classes]

    def establish_live_c_major(self, resolver: KeyScaleResolver) -> None:
        for observation in self.c_major_material():
            resolver.observe(observation)

    def test_known_c_major_material_resolves_c_major(self) -> None:
        result = KeyScaleResolver().analyze_upload(self.c_major_material())

        self.assertEqual(result.detected_key, KeyScale("C", "major"))
        self.assertEqual(result.effective_key, KeyScale("C", "major"))
        self.assertFalse(result.ambiguous)
        self.assertFalse(result.insufficient_evidence)

    def test_known_a_minor_material_resolves_a_minor(self) -> None:
        result = KeyScaleResolver().analyze_upload(self.a_minor_material())

        self.assertEqual(result.detected_key, KeyScale("A", "minor"))
        self.assertEqual(result.effective_key, KeyScale("A", "minor"))
        self.assertFalse(result.ambiguous)

    def test_live_stable_key_is_stabilized(self) -> None:
        resolver = KeyScaleResolver()
        self.establish_live_c_major(resolver)

        result = resolver.result()

        self.assertEqual(result.detected_key, KeyScale("C", "major"))
        self.assertEqual(result.stabilized_key, KeyScale("C", "major"))
        self.assertEqual(result.effective_key, KeyScale("C", "major"))
        self.assertEqual(result.analysis_mode, AnalysisMode.LIVE)

    def test_short_conflicting_observation_does_not_switch_key(self) -> None:
        resolver = KeyScaleResolver()
        self.establish_live_c_major(resolver)
        before = resolver.result()

        resolver.observe(self.observation(9))
        after = resolver.result()

        self.assertEqual(before.stabilized_key, KeyScale("C", "major"))
        self.assertEqual(after.stabilized_key, KeyScale("C", "major"))
        self.assertEqual(after.effective_key, KeyScale("C", "major"))

    def test_stabilization_requires_repeated_candidate_evidence(self) -> None:
        resolver = KeyScaleResolver()
        for observation in self.c_major_material()[:8]:
            result = resolver.observe(observation)
            self.assertIsNone(result.stabilized_key)

        self.establish_live_c_major(resolver)
        self.assertEqual(resolver.result().stabilized_key, KeyScale("C", "major"))

    def test_hysteresis_preserves_stabilized_key_during_weak_conflict(self) -> None:
        resolver = KeyScaleResolver()
        self.establish_live_c_major(resolver)

        for pitch_class in [9, 4, 9, 4]:
            resolver.observe_pitch(self.pitch_class_hz(pitch_class), 0.2)

        result = resolver.result()

        self.assertEqual(result.stabilized_key, KeyScale("C", "major"))
        self.assertEqual(result.effective_key, KeyScale("C", "major"))

    def test_confidence_dips_are_excluded_from_key_evidence(self) -> None:
        resolver = KeyScaleResolver()
        for observation in self.c_major_material():
            resolver.observe(observation)
        before = resolver.result()

        resolver.observe(
            PitchDetectionResult(
                frequency_hz=self.pitch_class_hz(9),
                confidence=0.0,
                voiced=False,
                tracking_state=TrackingState.TRACKING_LOST,
            )
        )

        self.assertEqual(resolver.result().stabilized_key, before.stabilized_key)

    def test_ambiguous_relative_major_minor_material_is_explicit(self) -> None:
        material = [self.observation(pitch_class) for pitch_class in list(range(12)) * 2]

        result = KeyScaleResolver().analyze_upload(material)

        self.assertIsNone(result.effective_key)
        self.assertTrue(result.ambiguous)
        self.assertFalse(result.insufficient_evidence)

    def test_silence_produces_insufficient_evidence(self) -> None:
        silence = [
            PitchDetectionResult(None, 0.0, False, TrackingState.TRACKING_LOST)
            for _ in range(16)
        ]

        result = KeyScaleResolver().analyze_upload(silence)

        self.assertIsNone(result.detected_key)
        self.assertIsNone(result.effective_key)
        self.assertFalse(result.ambiguous)
        self.assertTrue(result.insufficient_evidence)

    def test_no_usable_pitch_observations_do_not_invent_key(self) -> None:
        result = KeyScaleResolver().analyze_upload(
            [
                PitchDetectionResult(None, 0.0, False, TrackingState.TRACKING_LOST),
                PitchDetectionResult(float("nan"), 0.9, True, TrackingState.TRACKING),
            ]
        )

        self.assertIsNone(result.effective_key)
        self.assertTrue(result.insufficient_evidence)

    def test_manual_key_overrides_automatic_effective_key(self) -> None:
        resolver = KeyScaleResolver()
        self.establish_live_c_major(resolver)
        resolver.set_manual_key("D", "minor")
        result = resolver.apply_processing_boundary()

        self.assertEqual(result.detected_key, KeyScale("C", "major"))
        self.assertEqual(result.stabilized_key, KeyScale("C", "major"))
        self.assertEqual(result.manual_key, KeyScale("D", "minor"))
        self.assertEqual(result.effective_key, KeyScale("D", "minor"))
        self.assertEqual(result.analysis_mode, AnalysisMode.MANUAL)

    def test_manual_key_change_applies_at_processing_boundary(self) -> None:
        resolver = KeyScaleResolver()
        self.establish_live_c_major(resolver)
        resolver.set_manual_key("D", "minor")
        resolver.apply_processing_boundary()
        resolver.set_manual_key("E", "minor")

        before_boundary = resolver.result()
        after_boundary = resolver.apply_processing_boundary()

        self.assertEqual(before_boundary.effective_key, KeyScale("D", "minor"))
        self.assertEqual(after_boundary.effective_key, KeyScale("E", "minor"))

    def test_rapid_conflicting_observations_do_not_oscillate(self) -> None:
        resolver = KeyScaleResolver()
        self.establish_live_c_major(resolver)
        for pitch_class in [9, 0, 9, 0, 9, 0, 9, 0]:
            resolver.observe(self.observation(pitch_class))

        self.assertEqual(resolver.result().stabilized_key, KeyScale("C", "major"))

    def test_upload_uses_complete_track_evidence(self) -> None:
        first_half = self.c_major_material()[:16]
        second_half = self.a_minor_material()[:16]
        result = KeyScaleResolver().analyze_upload(first_half + second_half)

        self.assertIsNotNone(result.confidence)
        self.assertEqual(result.analysis_mode, AnalysisMode.UPLOAD)
        self.assertNotEqual(result.analysis_mode, AnalysisMode.LIVE)

    def test_low_confidence_observations_are_weighted_less(self) -> None:
        strong_c = self.c_major_material()
        weak_a = [self.observation(9, 0.01) for _ in range(12)]

        result = KeyScaleResolver().analyze_upload(strong_c + weak_a)

        self.assertEqual(result.detected_key, KeyScale("C", "major"))

    def test_manual_key_clears_ambiguity_when_live_observations_are_ambiguous(self) -> None:
        resolver = KeyScaleResolver()
        # Single observation leaves automatic detection without stabilized key -> ambiguous in automatic mode
        live_result = resolver.observe(self.observation(9))
        self.assertIsNone(live_result.detected_key)
        self.assertTrue(live_result.ambiguous)
        self.assertFalse(live_result.insufficient_evidence)

        # Activating manual key must make manual key authoritative and clear ambiguity
        resolver.set_manual_key("A", "major")
        manual_result = resolver.apply_processing_boundary()

        self.assertEqual(manual_result.effective_key, KeyScale("A", "major"))
        self.assertEqual(manual_result.manual_key, KeyScale("A", "major"))
        self.assertFalse(manual_result.ambiguous)
        self.assertFalse(manual_result.insufficient_evidence)
        self.assertEqual(manual_result.analysis_mode, AnalysisMode.MANUAL)

    def test_manual_key_in_upload_clears_ambiguity_for_ambiguous_track(self) -> None:
        ambiguous_material = [self.observation(pitch_class) for pitch_class in list(range(12)) * 2]
        resolver = KeyScaleResolver()
        resolver.set_manual_key("D", "minor")

        result = resolver.analyze_upload(ambiguous_material)

        self.assertEqual(result.effective_key, KeyScale("D", "minor"))
        self.assertFalse(result.ambiguous)
        self.assertFalse(result.insufficient_evidence)
        self.assertEqual(result.analysis_mode, AnalysisMode.MANUAL)


if __name__ == "__main__":
    unittest.main()
