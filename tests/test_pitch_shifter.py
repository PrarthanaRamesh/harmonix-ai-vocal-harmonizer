import unittest

import numpy as np

from harmonix.pitch_detector import TrackingState
from harmonix.pitch_shifter import (
    F0_CEIL_HZ,
    F0_FLOOR_HZ,
    HOP_SIZE,
    LIVE_BLOCK_SIZE,
    PitchShiftStatus,
    PitchShifter,
    WORLD_SEGMENT_SIZE,
)


class FakeWorld:
    def __init__(self, *, f0=None, synthesis_delta=0, synthesis_shape=None, analysis_error=False, synthesis_error=False):
        self.f0 = np.asarray(f0 if f0 is not None else [440.0, 440.0, 0.0, 440.0], dtype=np.float64)
        self.synthesis_delta = synthesis_delta
        self.synthesis_shape = synthesis_shape
        self.analysis_error = analysis_error
        self.synthesis_error = synthesis_error
        self.last_target_f0 = None
        self.dio_calls = 0
        self.synthesize_calls = 0

    def dio(self, audio, fs, **kwargs):
        self.dio_calls += 1
        if self.analysis_error:
            raise RuntimeError("analysis failure")
        return self.f0.copy(), np.arange(self.f0.size, dtype=np.float64) * 0.005

    def stonemask(self, audio, f0, temporal_positions, fs):
        return np.asarray(f0, dtype=np.float64)

    def cheaptrick(self, audio, f0, temporal_positions, fs, **kwargs):
        return np.ones((len(f0), 8), dtype=np.float64)

    def d4c(self, audio, f0, temporal_positions, fs, **kwargs):
        return np.ones((len(f0), 8), dtype=np.float64)

    def synthesize(self, f0, spectrogram, aperiodicity, fs, **kwargs):
        self.synthesize_calls += 1
        if self.synthesis_error:
            raise RuntimeError("synthesis failure")
        self.last_target_f0 = np.asarray(f0, dtype=np.float64).copy()
        if self.synthesis_shape is not None:
            return np.zeros(self.synthesis_shape, dtype=np.float64)
        return np.zeros(WORLD_SEGMENT_SIZE + self.synthesis_delta, dtype=np.float64)


class PitchShifterTests(unittest.TestCase):
    @staticmethod
    def sine(frequency: float, samples: int = WORLD_SEGMENT_SIZE) -> np.ndarray:
        time = np.arange(samples, dtype=np.float64) / 44100.0
        return np.sin(2.0 * np.pi * frequency * time).astype(np.float32)

    @staticmethod
    def fake_shifter(**kwargs) -> tuple[PitchShifter, FakeWorld]:
        world = FakeWorld(**kwargs)
        return PitchShifter(world_api=world), world

    # ------------------------------------------------------------------
    # Full-buffer (shift_full_buffer) — unchanged behaviour
    # ------------------------------------------------------------------

    def test_known_sine_uses_real_world_path(self) -> None:
        result = PitchShifter().shift_full_buffer(self.sine(440.0), 440.0, 660.0)

        self.assertTrue(result.valid)
        self.assertEqual(result.status, PitchShiftStatus.VALID)
        self.assertEqual(result.output_sample_count, WORLD_SEGMENT_SIZE)
        self.assertEqual(result.audio.dtype, np.float32)
        self.assertTrue(np.isfinite(result.audio).all())

    def test_upward_shift_ratio_440_to_660(self) -> None:
        shifter, world = self.fake_shifter(f0=[440.0, 440.0])
        result = shifter.shift_full_buffer(np.ones(WORLD_SEGMENT_SIZE, dtype=np.float32), 440.0, 660.0)

        np.testing.assert_allclose(world.last_target_f0, [660.0, 660.0])
        self.assertAlmostEqual(result.shift_ratio, 1.5)

    def test_octave_shift_440_to_880(self) -> None:
        shifter, world = self.fake_shifter(f0=[440.0, 0.0, 440.0])
        result = shifter.shift_full_buffer(np.ones(WORLD_SEGMENT_SIZE, dtype=np.float32), 440.0, 880.0)

        np.testing.assert_allclose(world.last_target_f0, [880.0, 0.0, 880.0])
        self.assertAlmostEqual(result.shift_ratio, 2.0)

    def test_downward_shift_440_to_220(self) -> None:
        shifter, world = self.fake_shifter(f0=[440.0, 440.0])
        result = shifter.shift_full_buffer(np.ones(WORLD_SEGMENT_SIZE, dtype=np.float32), 440.0, 220.0)

        np.testing.assert_allclose(world.last_target_f0, [220.0, 220.0])
        self.assertAlmostEqual(result.shift_ratio, 0.5)

    def test_zero_shift_uses_world_and_preserves_length(self) -> None:
        shifter, world = self.fake_shifter(f0=[440.0, 440.0])
        result = shifter.shift_full_buffer(np.ones(WORLD_SEGMENT_SIZE, dtype=np.float32), 440.0, 440.0)

        self.assertEqual(result.status, PitchShiftStatus.ZERO_SHIFT)
        self.assertEqual(world.dio_calls, 1)
        self.assertEqual(world.synthesize_calls, 1)
        self.assertTrue(result.duration_preserved)

    def test_output_contract_is_mono_float32_and_same_length(self) -> None:
        shifter, _ = self.fake_shifter(f0=[440.0])
        result = shifter.shift_full_buffer(np.ones(WORLD_SEGMENT_SIZE, dtype=np.float32), 440.0, 660.0)

        self.assertEqual(result.audio.ndim, 1)
        self.assertEqual(result.audio.dtype, np.float32)
        self.assertEqual(result.input_sample_count, WORLD_SEGMENT_SIZE)
        self.assertEqual(result.output_sample_count, WORLD_SEGMENT_SIZE)
        self.assertTrue(result.duration_preserved)

    def test_result_audio_is_read_only(self) -> None:
        shifter, _ = self.fake_shifter(f0=[440.0])
        result = shifter.shift_full_buffer(np.ones(WORLD_SEGMENT_SIZE, dtype=np.float32), 440.0, 660.0)

        with self.assertRaises(ValueError):
            result.audio[0] = 1.0

    def test_non_1d_world_synthesis_output_is_rejected(self) -> None:
        shifter, _ = self.fake_shifter(f0=[440.0], synthesis_shape=(WORLD_SEGMENT_SIZE, 2))
        result = shifter.shift_full_buffer(np.ones(WORLD_SEGMENT_SIZE, dtype=np.float32), 440.0, 660.0)

        self.assertEqual(result.status, PitchShiftStatus.WORLD_SYNTHESIS_ERROR)
        self.assertFalse(result.valid)
        self.assertIsNone(result.audio)

    def test_silence_skips_world_and_returns_zero_audio(self) -> None:
        shifter, world = self.fake_shifter()
        result = shifter.shift_full_buffer(np.zeros(WORLD_SEGMENT_SIZE, dtype=np.float32), 440.0, 660.0)

        self.assertEqual(result.status, PitchShiftStatus.SILENCE)
        self.assertEqual(world.dio_calls, 0)
        self.assertTrue(np.array_equal(result.audio, np.zeros(WORLD_SEGMENT_SIZE, dtype=np.float32)))

    def test_unvoiced_and_missing_pitch(self) -> None:
        shifter, world = self.fake_shifter()
        unvoiced = shifter.shift_full_buffer(np.ones(WORLD_SEGMENT_SIZE, dtype=np.float32), 440.0, 660.0, source_voiced=False)
        missing = shifter.shift_full_buffer(np.ones(WORLD_SEGMENT_SIZE, dtype=np.float32), None, 660.0)

        self.assertEqual(unvoiced.status, PitchShiftStatus.UNVOICED)
        self.assertEqual(missing.status, PitchShiftStatus.MISSING_SOURCE_PITCH)
        self.assertEqual(world.dio_calls, 0)

    def test_tracking_lost_is_unvoiced(self) -> None:
        shifter, _ = self.fake_shifter()
        result = shifter.shift_full_buffer(
            np.ones(WORLD_SEGMENT_SIZE, dtype=np.float32),
            440.0,
            660.0,
            tracking_state=TrackingState.TRACKING_LOST,
        )

        self.assertEqual(result.status, PitchShiftStatus.UNVOICED)

    def test_nonfinite_audio_and_pitch_are_rejected(self) -> None:
        shifter, _ = self.fake_shifter()
        audio = np.ones(WORLD_SEGMENT_SIZE, dtype=np.float32)
        audio[10] = np.nan

        self.assertEqual(shifter.shift_full_buffer(audio, 440.0, 660.0).status, PitchShiftStatus.NONFINITE_INPUT)
        self.assertEqual(shifter.shift_full_buffer(np.ones(WORLD_SEGMENT_SIZE, dtype=np.float32), np.nan, 660.0).status, PitchShiftStatus.NONFINITE_PITCH)
        self.assertEqual(shifter.shift_full_buffer(np.ones(WORLD_SEGMENT_SIZE, dtype=np.float32), 440.0, np.inf).status, PitchShiftStatus.NONFINITE_PITCH)

    def test_target_range_and_ratio_limits(self) -> None:
        shifter, _ = self.fake_shifter()
        audio = np.ones(WORLD_SEGMENT_SIZE, dtype=np.float32)

        self.assertEqual(shifter.shift_full_buffer(audio, 440.0, 64.99).status, PitchShiftStatus.TARGET_OUT_OF_RANGE)
        self.assertEqual(shifter.shift_full_buffer(audio, 440.0, 1000.01).status, PitchShiftStatus.TARGET_OUT_OF_RANGE)
        self.assertEqual(shifter.shift_full_buffer(audio, 440.0, 65.0).status, PitchShiftStatus.SHIFT_RATIO_UNSUPPORTED)
        self.assertEqual(shifter.shift_full_buffer(audio, 440.0, 1000.0).status, PitchShiftStatus.VALID)

    def test_was_clamped_is_m6_metadata_not_recomputed(self) -> None:
        shifter, _ = self.fake_shifter(f0=[440.0])
        result = shifter.shift_full_buffer(np.ones(WORLD_SEGMENT_SIZE, dtype=np.float32), 440.0, 660.0, target_was_clamped=True)

        self.assertTrue(result.was_clamped)
        self.assertEqual(result.target_frequency_hz, 660.0)

    def test_world_analysis_and_synthesis_errors_are_typed(self) -> None:
        analysis_shifter, _ = self.fake_shifter(analysis_error=True)
        synthesis_shifter, _ = self.fake_shifter(synthesis_error=True)
        audio = np.ones(WORLD_SEGMENT_SIZE, dtype=np.float32)

        self.assertEqual(analysis_shifter.shift_full_buffer(audio, 440.0, 660.0).status, PitchShiftStatus.WORLD_ANALYSIS_ERROR)
        self.assertEqual(synthesis_shifter.shift_full_buffer(audio, 440.0, 660.0).status, PitchShiftStatus.WORLD_SYNTHESIS_ERROR)

    def test_one_sample_synthesis_mismatch_is_corrected(self) -> None:
        shifter, _ = self.fake_shifter(f0=[440.0], synthesis_delta=1)
        result = shifter.shift_full_buffer(np.ones(WORLD_SEGMENT_SIZE, dtype=np.float32), 440.0, 660.0)

        self.assertTrue(result.valid)
        self.assertEqual(result.output_sample_count, WORLD_SEGMENT_SIZE)

    def test_more_than_one_sample_mismatch_fails(self) -> None:
        shifter, _ = self.fake_shifter(f0=[440.0], synthesis_delta=-2)
        result = shifter.shift_full_buffer(np.ones(WORLD_SEGMENT_SIZE, dtype=np.float32), 440.0, 660.0)

        self.assertEqual(result.status, PitchShiftStatus.SAMPLE_COUNT_MISMATCH)
        self.assertFalse(result.valid)
        self.assertIsNone(result.audio)
        self.assertEqual(result.output_sample_count, 0)

    def test_mixed_voiced_unvoiced_world_f0_is_valid(self) -> None:
        shifter, world = self.fake_shifter(f0=[440.0, 0.0, np.nan, np.inf, 440.0])
        result = shifter.shift_full_buffer(np.ones(WORLD_SEGMENT_SIZE, dtype=np.float32), 440.0, 660.0)

        self.assertTrue(result.valid)
        np.testing.assert_allclose(world.last_target_f0, [660.0, 0.0, 0.0, 0.0, 660.0])

    def test_all_unvoiced_world_f0_is_unvoiced(self) -> None:
        shifter, _ = self.fake_shifter(f0=[0.0, np.nan, 0.0])
        result = shifter.shift_full_buffer(np.ones(WORLD_SEGMENT_SIZE, dtype=np.float32), 440.0, 660.0)

        self.assertEqual(result.status, PitchShiftStatus.UNVOICED)
        self.assertFalse(result.valid)

    def test_repeated_processing_is_deterministic(self) -> None:
        first_shifter, _ = self.fake_shifter(f0=[440.0, 0.0, 440.0])
        second_shifter, _ = self.fake_shifter(f0=[440.0, 0.0, 440.0])
        audio = np.ones(WORLD_SEGMENT_SIZE, dtype=np.float32)
        first = first_shifter.shift_full_buffer(audio, 440.0, 660.0)
        second = second_shifter.shift_full_buffer(audio, 440.0, 660.0)

        np.testing.assert_allclose(first.audio, second.audio, rtol=0.0, atol=1e-7)
        self.assertEqual(first.sample_rate, second.sample_rate)
        self.assertEqual(first.input_sample_count, second.input_sample_count)
        self.assertEqual(first.output_sample_count, second.output_sample_count)
        self.assertEqual(first.source_frequency_hz, second.source_frequency_hz)
        self.assertEqual(first.target_frequency_hz, second.target_frequency_hz)
        self.assertEqual(first.shift_ratio, second.shift_ratio)
        self.assertEqual(first.status, second.status)
        self.assertEqual(first.valid, second.valid)
        self.assertEqual(first.duration_preserved, second.duration_preserved)
        self.assertEqual(first.was_clamped, second.was_clamped)

    # ------------------------------------------------------------------
    # Live streaming — emission schedule and crossfade (unchanged behaviour)
    # ------------------------------------------------------------------

    def test_live_accumulation_first_output_block_four(self) -> None:
        shifter, world = self.fake_shifter(f0=[440.0])
        block = np.ones(LIVE_BLOCK_SIZE, dtype=np.float32)

        self.assertIsNone(shifter.process_block(block, 440.0, 660.0))
        self.assertIsNone(shifter.process_block(block, 440.0, 660.0))
        self.assertIsNone(shifter.process_block(block, 440.0, 660.0))
        first = shifter.process_block(block, 440.0, 660.0)

        self.assertIsNotNone(first)
        self.assertEqual(first.output_sample_count, HOP_SIZE)
        self.assertEqual(first.audio.size, HOP_SIZE)
        self.assertEqual(shifter.output_sample_clock, HOP_SIZE)
        self.assertEqual(world.dio_calls, 1)

    def test_live_block_five_none_and_block_six_second_output(self) -> None:
        shifter, world = self.fake_shifter(f0=[440.0])
        block = np.ones(LIVE_BLOCK_SIZE, dtype=np.float32)
        for _ in range(4):
            shifter.process_block(block, 440.0, 660.0)
        self.assertIsNone(shifter.process_block(block, 440.0, 660.0))
        second = shifter.process_block(block, 440.0, 660.0)

        self.assertIsNotNone(second)
        self.assertEqual(second.audio.size, HOP_SIZE)
        self.assertEqual(shifter.output_sample_clock, HOP_SIZE * 2)
        self.assertEqual(world.dio_calls, 2)

    def test_live_crossfade_is_finite_and_hop_aligned(self) -> None:
        shifter, _ = self.fake_shifter(f0=[440.0])
        block = np.ones(LIVE_BLOCK_SIZE, dtype=np.float32)
        for _ in range(6):
            result = shifter.process_block(block, 440.0, 660.0)
        self.assertTrue(np.isfinite(result.audio).all())
        self.assertEqual(result.audio.shape, (HOP_SIZE,))

    def test_crossfade_uses_equal_power_overlap_weights(self) -> None:
        previous = np.full(HOP_SIZE, 2.0, dtype=np.float32)
        current = np.full(HOP_SIZE, 4.0, dtype=np.float32)

        output = PitchShifter._crossfade(previous, current)
        phase = np.linspace(0.0, np.pi / 2.0, HOP_SIZE, dtype=np.float32)
        fade_in = np.sin(phase, dtype=np.float32)
        fade_out = np.cos(phase, dtype=np.float32)
        fade_out[-1] = 0.0
        fade_in[0] = 0.0

        # Equal power verification: fade_in^2 + fade_out^2 == 1.0 everywhere
        power_sum = fade_in**2 + fade_out**2
        np.testing.assert_allclose(power_sum, 1.0, rtol=0.0, atol=1e-6)

        # Output shape and finiteness
        self.assertEqual(output.shape, (HOP_SIZE,))
        self.assertTrue(np.isfinite(output).all())

        # Exact boundary values: starts at 100% previous, ends at 100% current
        self.assertAlmostEqual(float(output[0]), 2.0, places=5)
        self.assertAlmostEqual(float(output[-1]), 4.0, places=5)

        # Midpoint power / weighting check
        mid = HOP_SIZE // 2
        expected_mid = float(2.0 * np.cos(np.pi / 4.0) + 4.0 * np.sin(np.pi / 4.0))
        self.assertAlmostEqual(float(output[mid]), expected_mid, delta=0.01)

        expected = previous * fade_out + current * fade_in
        np.testing.assert_allclose(output, expected, rtol=0.0, atol=1e-7)

    # ------------------------------------------------------------------
    # NEW: Block-aligned metadata queue — synchronisation and reset
    # ------------------------------------------------------------------

    def test_meta_queue_length_matches_audio_buffer_before_first_emit(self) -> None:
        """_meta_queue length == _input_buffer.size // LIVE_BLOCK_SIZE at all times."""
        shifter, _ = self.fake_shifter(f0=[440.0])
        block = np.ones(LIVE_BLOCK_SIZE, dtype=np.float32)
        for i in range(1, 4):
            shifter.process_block(block, 440.0, 660.0)
            buf_blocks = shifter._input_buffer.size // LIVE_BLOCK_SIZE
            meta_len = len(shifter._meta_queue)
            self.assertEqual(buf_blocks, i, f"call {i}: audio buffer has {buf_blocks} blocks")
            self.assertEqual(meta_len, i, f"call {i}: meta queue has {meta_len} entries")

    def test_meta_queue_slides_in_sync_with_audio_buffer_on_emit(self) -> None:
        """After an emit the queue must contain exactly 2 entries (the retained overlap blocks)."""
        shifter, _ = self.fake_shifter(f0=[440.0])
        block = np.ones(LIVE_BLOCK_SIZE, dtype=np.float32)
        for _ in range(4):
            shifter.process_block(block, 440.0, 660.0)
        # After 4 blocks: one emit occurred, buffer slid by HOP_SIZE (2 blocks).
        self.assertEqual(len(shifter._meta_queue), 2)
        self.assertEqual(shifter._input_buffer.size, LIVE_BLOCK_SIZE * 2)

    def test_reset_clears_meta_queue_and_audio_buffer(self) -> None:
        """reset() must clear _meta_queue independently, not just _input_buffer."""
        shifter, _ = self.fake_shifter(f0=[440.0])
        block = np.ones(LIVE_BLOCK_SIZE, dtype=np.float32)
        for _ in range(3):
            shifter.process_block(block, 440.0, 660.0)
        self.assertEqual(len(shifter._meta_queue), 3)

        shifter.reset()
        self.assertEqual(len(shifter._meta_queue), 0)
        self.assertEqual(shifter._input_buffer.size, 0)
        self.assertEqual(shifter.output_sample_clock, 0)

    def test_reset_does_not_carry_old_metadata_into_next_session(self) -> None:
        """After reset(), feeding 4 new blocks with a different target must use the new target."""
        shifter, world = self.fake_shifter(f0=[440.0])
        block = np.ones(LIVE_BLOCK_SIZE, dtype=np.float32)
        # First session: target 660 Hz
        for _ in range(3):
            shifter.process_block(block, 440.0, 660.0)
        shifter.reset()
        # Second session: target 550 Hz — verify the emitted result uses 550, not 660
        for _ in range(3):
            shifter.process_block(block, 440.0, 550.0)
        first = shifter.process_block(block, 440.0, 550.0)
        self.assertIsNotNone(first)
        self.assertEqual(first.target_frequency_hz, 550.0)
        self.assertEqual(shifter.output_sample_clock, HOP_SIZE)

    # ------------------------------------------------------------------
    # NEW: Per-block target-pitch alignment across consecutive WORLD windows
    # ------------------------------------------------------------------

    def test_emitted_result_target_reflects_emitted_blocks_not_latest_block(self) -> None:
        """Emitted result metadata must come from the blocks being emitted (B0&B1),
        not from the most recently received block (B3).

        Emission schedule:
          - 8 calls → Hop1(call4, emits B0&B1), Hop2(call6, emits B2&B3), Hop3(call8, emits B4&B5)
          - B0–B3: target 523.25 Hz; B4–B7: target 660.0 Hz.
          - Hop 1 → target_frequency_hz must be 523.25 (from B0&B1)
          - Hop 2 → target_frequency_hz must be 523.25 (from B2&B3)
          - Hop 3 → target_frequency_hz must be 660.0  (from B4&B5)
        """
        shifter, _ = self.fake_shifter(f0=[440.0])
        block = np.ones(LIVE_BLOCK_SIZE, dtype=np.float32)

        hops = []
        for i in range(8):  # 8 calls → exactly 3 emits
            tgt = 523.25 if i < 4 else 660.0
            res = shifter.process_block(block, 440.0, tgt)
            if res is not None:
                hops.append(res)

        self.assertEqual(len(hops), 3)
        self.assertAlmostEqual(hops[0].target_frequency_hz, 523.25, places=2,
                               msg="Hop 1 (emits B0&B1): expected target 523.25")
        self.assertAlmostEqual(hops[1].target_frequency_hz, 523.25, places=2,
                               msg="Hop 2 (emits B2&B3): expected target 523.25")
        self.assertAlmostEqual(hops[2].target_frequency_hz, 660.0, places=2,
                               msg="Hop 3 (emits B4&B5): expected target 660.0")

    def test_per_frame_ratio_applied_per_block_with_fake_world(self) -> None:
        """Verify that FakeWorld receives a target_f0 array built from per-block ratios.

        With 4 blocks all having source=440 and different targets, the frames
        belonging to different blocks should receive different ratios.
        Block 0 & 1 → ratio 523.25/440 ≈ 1.1892
        Block 2 & 3 → ratio 660.0/440  ≈ 1.5000
        FakeWorld's f0 = [440] repeated; stonemask echoes that, so target_f0[k]
        should be 440 * ratio_for_block(k).
        """
        # Use enough FakeWorld frames to span all 4 blocks.
        # With 5ms frames at 44100 Hz: 38 frames for 8192 samples.
        # Build a synthetic f0 array of 38 voiced frames at 440 Hz.
        n_frames = 38
        fake_f0 = np.full(n_frames, 440.0, dtype=np.float64)
        shifter, world = self.fake_shifter(f0=fake_f0)
        block = np.ones(LIVE_BLOCK_SIZE, dtype=np.float32)

        # 4 blocks: B0&B1 → 523.25 Hz, B2&B3 → 660.0 Hz
        for i in range(4):
            tgt = 523.25 if i < 2 else 660.0
            shifter.process_block(block, 440.0, tgt)

        self.assertIsNotNone(world.last_target_f0)
        tf0 = world.last_target_f0

        # timeaxis[k] = k * 0.005s; frame k belongs to block floor(k*0.005*44100/2048)
        timeaxis = np.arange(n_frames) * 0.005
        block_ids = np.clip(
            np.floor(timeaxis * 44100 / LIVE_BLOCK_SIZE).astype(int), 0, 3
        )
        ratio_b01 = 523.25 / 440.0
        ratio_b23 = 660.0 / 440.0

        for k in range(n_frames):
            blk = block_ids[k]
            expected_ratio = ratio_b01 if blk < 2 else ratio_b23
            expected_target = 440.0 * expected_ratio
            self.assertAlmostEqual(
                tf0[k], expected_target, delta=0.01,
                msg=f"Frame {k} (block {blk}): expected {expected_target:.2f} Hz, got {tf0[k]:.2f} Hz"
            )

    # ------------------------------------------------------------------
    # NEW: Musical interval accuracy (major third, minor third, dim-5)
    # ------------------------------------------------------------------

    def test_major_third_target_ratio_correct_in_emitted_metadata(self) -> None:
        """Major third above A4 (440 Hz) in C major: C5 (523.25 Hz) → E5 (659.26 Hz).
        Source=C5=523.25, target=E5=659.26 Hz. Ratio ≈ 1.2599 (major third).
        """
        shifter, _ = self.fake_shifter(f0=[523.25])
        block = np.ones(LIVE_BLOCK_SIZE, dtype=np.float32)
        target_e5 = 659.26
        for _ in range(3):
            shifter.process_block(block, 523.25, target_e5)
        hop1 = shifter.process_block(block, 523.25, target_e5)
        self.assertIsNotNone(hop1)
        self.assertAlmostEqual(hop1.target_frequency_hz, target_e5, places=1)
        self.assertAlmostEqual(hop1.shift_ratio, target_e5 / 523.25, places=4)

    def test_minor_third_target_ratio_correct_in_emitted_metadata(self) -> None:
        """Minor third: A4 (440 Hz) → C5 (523.25 Hz). Ratio ≈ 1.1892."""
        shifter, _ = self.fake_shifter(f0=[440.0])
        block = np.ones(LIVE_BLOCK_SIZE, dtype=np.float32)
        target_c5 = 523.25
        for _ in range(3):
            shifter.process_block(block, 440.0, target_c5)
        hop1 = shifter.process_block(block, 440.0, target_c5)
        self.assertIsNotNone(hop1)
        self.assertAlmostEqual(hop1.target_frequency_hz, target_c5, places=1)
        self.assertAlmostEqual(hop1.shift_ratio, target_c5 / 440.0, places=4)

    def test_diminished_fifth_target_ratio_correct_in_emitted_metadata(self) -> None:
        """Diminished fifth (tritone): B4 (493.88 Hz) → F5 (698.46 Hz). Ratio ≈ 1.4142."""
        shifter, _ = self.fake_shifter(f0=[493.88])
        block = np.ones(LIVE_BLOCK_SIZE, dtype=np.float32)
        target_f5 = 698.46
        for _ in range(3):
            shifter.process_block(block, 493.88, target_f5)
        hop1 = shifter.process_block(block, 493.88, target_f5)
        self.assertIsNotNone(hop1)
        self.assertAlmostEqual(hop1.target_frequency_hz, target_f5, places=1)
        self.assertAlmostEqual(hop1.shift_ratio, target_f5 / 493.88, places=4)

    # ------------------------------------------------------------------
    # NEW: Voiced → unvoiced → voiced transitions
    # ------------------------------------------------------------------

    def test_all_unvoiced_segment_returns_invalid_result(self) -> None:
        """When all 4 blocks in a segment are unvoiced, process_block returns invalid.

        The emitted-block guard in shift_live() runs before the silence/audio checks,
        so unvoiced metadata always produces UNVOICED regardless of audio content.
        """
        shifter, _ = self.fake_shifter(f0=[440.0])
        silent = np.zeros(LIVE_BLOCK_SIZE, dtype=np.float32)
        for _ in range(3):
            shifter.process_block(silent, None, None, source_voiced=False)
        res = shifter.process_block(silent, None, None, source_voiced=False)
        self.assertIsNotNone(res)
        self.assertFalse(res.valid)
        self.assertIsNone(res.audio)
        self.assertEqual(res.status, PitchShiftStatus.UNVOICED)

    def test_partial_unvoiced_segment_emitted_blocks_voiced(self) -> None:
        """Segment = 2 voiced blocks (B0&B1) + 2 unvoiced (B2&B3).
        Emitted hop covers B0&B1 which are voiced → result should be valid
        because at least one emitted block is usable."""
        shifter, _ = self.fake_shifter(f0=[440.0])
        voiced = np.ones(LIVE_BLOCK_SIZE, dtype=np.float32)
        silent = np.zeros(LIVE_BLOCK_SIZE, dtype=np.float32)
        # B0, B1 voiced:
        shifter.process_block(voiced, 440.0, 550.0, source_voiced=True)
        shifter.process_block(voiced, 440.0, 550.0, source_voiced=True)
        # B2, B3 unvoiced:
        shifter.process_block(silent, None, None, source_voiced=False)
        res = shifter.process_block(silent, None, None, source_voiced=False)
        # Emitted hop covers B0&B1 (both voiced) → should succeed.
        self.assertIsNotNone(res)
        self.assertTrue(res.valid)

    def test_partial_unvoiced_segment_emitted_blocks_unvoiced(self) -> None:
        """Segment = 2 unvoiced blocks (B0&B1) + 2 voiced (B2&B3).
        Emitted hop covers B0&B1 which are unvoiced → result must be invalid."""
        shifter, _ = self.fake_shifter(f0=[440.0])
        voiced = np.ones(LIVE_BLOCK_SIZE, dtype=np.float32)
        silent = np.zeros(LIVE_BLOCK_SIZE, dtype=np.float32)
        # B0, B1 unvoiced:
        shifter.process_block(silent, None, None, source_voiced=False)
        shifter.process_block(silent, None, None, source_voiced=False)
        # B2, B3 voiced:
        shifter.process_block(voiced, 440.0, 550.0, source_voiced=True)
        res = shifter.process_block(voiced, 440.0, 550.0, source_voiced=True)
        # Emitted hop covers B0&B1 (both unvoiced) → must be invalid.
        self.assertIsNotNone(res)
        self.assertFalse(res.valid)
        self.assertIsNone(res.audio)

    def test_tracking_lost_in_emitted_blocks_produces_unvoiced_result(self) -> None:
        """TRACKING_LOST in emitted blocks must produce an invalid / unvoiced result."""
        shifter, _ = self.fake_shifter(f0=[440.0])
        block = np.ones(LIVE_BLOCK_SIZE, dtype=np.float32)
        # B0, B1 with TRACKING_LOST:
        shifter.process_block(block, 440.0, 550.0, tracking_state=TrackingState.TRACKING_LOST)
        shifter.process_block(block, 440.0, 550.0, tracking_state=TrackingState.TRACKING_LOST)
        # B2, B3 voiced:
        shifter.process_block(block, 440.0, 550.0, source_voiced=True)
        res = shifter.process_block(block, 440.0, 550.0, source_voiced=True)
        self.assertIsNotNone(res)
        self.assertFalse(res.valid)

    # ------------------------------------------------------------------
    # NEW: Real PyWorld streaming continuity (pre-existing, kept)
    # ------------------------------------------------------------------

    def test_real_world_streaming_continuity_across_consecutive_hops(self) -> None:
        """Verify that streaming continuous voiced audio through real PyWorld produces
        consecutive valid hops of HOP_SIZE with continuous RMS energy and bounded boundary steps."""
        sr = 44100
        t = np.arange(int(sr * 0.8)) / sr
        signal = (0.5 * np.sin(2.0 * np.pi * 440.0 * t) + 0.2 * np.sin(2.0 * np.pi * 880.0 * t)).astype(np.float32)

        shifter = PitchShifter()
        emitted_hops = []
        for start in range(0, len(signal) - LIVE_BLOCK_SIZE, LIVE_BLOCK_SIZE):
            block = signal[start : start + LIVE_BLOCK_SIZE]
            res = shifter.process_block(block, 440.0, 554.37, source_voiced=True)
            if res is not None and res.audio is not None:
                emitted_hops.append(res.audio)

        self.assertGreaterEqual(len(emitted_hops), 4)

        for hop in emitted_hops:
            self.assertEqual(hop.size, HOP_SIZE)
            self.assertEqual(hop.dtype, np.float32)
            self.assertTrue(np.isfinite(hop).all())

        # Check RMS continuity across steady-state hops
        full_steady = np.concatenate(emitted_hops[1:])
        seg_size = 512
        segment_rms = [np.sqrt(np.mean(full_steady[i : i + seg_size] ** 2)) for i in range(0, len(full_steady) - seg_size, seg_size)]
        mean_rms = float(np.mean(segment_rms))

        min_rms = float(np.min(segment_rms))
        max_rms = float(np.max(segment_rms))
        self.assertGreater(min_rms, 0.40 * mean_rms, f"Unexpected RMS dropout: min={min_rms:.3f}, mean={mean_rms:.3f}")
        self.assertLess(max_rms, 1.60 * mean_rms, f"Unexpected RMS surge: max={max_rms:.3f}, mean={mean_rms:.3f}")

        # Check boundary step discontinuity between consecutive hops
        for i in range(1, len(emitted_hops)):
            step = abs(emitted_hops[i][0] - emitted_hops[i - 1][-1])
            self.assertLess(step, 0.1, f"Excessive step discontinuity at hop {i} boundary: {step:.4f}")

    def test_real_world_voiced_unvoiced_transition(self) -> None:
        """Verify that transitioning between voiced and unvoiced audio handles state cleanly.

        With the aligned emitted-block gate:
        - Segment 1 (B0–B3, all voiced): emits valid result.
        - After that, B4 = unvoiced silence, B5 = unvoiced silence.
        - Segment 2 (B2&B3 voiced + B4&B5 unvoiced): emitted blocks B2&B3 are voiced
          → shift_live returns a valid result (emitted portion has signal).
        - Next segment must have B4&B5 as emitted blocks (both unvoiced) → invalid.
        """
        shifter = PitchShifter()
        voiced_block = np.sin(2.0 * np.pi * 440.0 * np.arange(LIVE_BLOCK_SIZE) / 44100.0).astype(np.float32)
        silent_block = np.zeros(LIVE_BLOCK_SIZE, dtype=np.float32)

        # Accumulate 4 voiced blocks → Hop 1 emits B0&B1, both voiced.
        for _ in range(3):
            self.assertIsNone(shifter.process_block(voiced_block, 440.0, 550.0, source_voiced=True))
        voiced_res = shifter.process_block(voiced_block, 440.0, 550.0, source_voiced=True)
        self.assertIsNotNone(voiced_res)
        self.assertTrue(voiced_res.valid)

        # B4 = unvoiced silence (no emit, buffer not yet full).
        no_emit = shifter.process_block(silent_block, None, None, source_voiced=False)
        self.assertIsNone(no_emit)

        # B5 = unvoiced silence → Segment 2 = B2(voiced) B3(voiced) B4(unvoiced) B5(unvoiced).
        # Emitted hop covers B2&B3 (both voiced) → valid result expected.
        mixed_res = shifter.process_block(silent_block, None, None, source_voiced=False)
        self.assertIsNotNone(mixed_res)
        self.assertTrue(mixed_res.valid,
                        msg="Segment 2 emits B2&B3 (voiced); result should be valid")

        # B6 = unvoiced silence.
        no_emit2 = shifter.process_block(silent_block, None, None, source_voiced=False)
        self.assertIsNone(no_emit2)

        # B7 = unvoiced silence → Segment 3 = B4(unvoiced) B5(unvoiced) B6(unvoiced) B7(unvoiced).
        # Emitted hop covers B4&B5 (both unvoiced) → invalid result expected.
        fully_unvoiced_res = shifter.process_block(silent_block, None, None, source_voiced=False)
        self.assertIsNotNone(fully_unvoiced_res)
        self.assertFalse(fully_unvoiced_res.valid,
                         msg="Segment 3 emits B4&B5 (unvoiced); result should be invalid")
        self.assertIsNone(fully_unvoiced_res.audio)

    # ------------------------------------------------------------------
    # NEW: Spectral pitch accuracy across a note transition (real PyWorld)
    # ------------------------------------------------------------------

    def test_real_world_emitted_target_correct_for_major_third_interval(self) -> None:
        """With a continuous A4 (440 Hz) sine and major-third target (C#5 = 554.37 Hz),
        real PyWorld output must report the correct target frequency on each hop."""
        sr = 44100
        t = np.arange(int(sr * 0.5)) / sr
        signal = (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)
        shifter = PitchShifter()
        target_hz = 554.37  # major third
        hops = []
        for start in range(0, len(signal) - LIVE_BLOCK_SIZE, LIVE_BLOCK_SIZE):
            block = signal[start : start + LIVE_BLOCK_SIZE]
            res = shifter.process_block(block, 440.0, target_hz, source_voiced=True)
            if res is not None and res.valid:
                hops.append(res)
        self.assertGreaterEqual(len(hops), 2)
        for hop in hops:
            self.assertAlmostEqual(hop.target_frequency_hz, target_hz, places=1,
                                   msg="Reported target_frequency_hz must equal the input target")

    def test_real_world_no_dual_pitch_in_crossfade_at_note_transition(self) -> None:
        """After the timing fix, the transition hop (Hop 3 in a 6A+6B block sequence)
        must not contain a competing spectral peak from the new note.

        Source A: 440 Hz, target 523.25 Hz (minor third / C major unison)
        Source B: 523.25 Hz, target 659.25 Hz (major third / E above C)
        6 blocks of A, 6 blocks of B.  Hop 3 emits B4&B5 (pure A audio).
        Its dominant FFT peak must be near 523.25 Hz; any competing peak
        must be < 30% relative magnitude.
        """
        sr = 44100
        # Build 12 blocks of audio: first 6 at 440 Hz, next 6 at 523.25 Hz.
        n_blocks = 12
        total_samples = n_blocks * LIVE_BLOCK_SIZE
        t_all = np.arange(total_samples) / sr
        freq_a, freq_b = 440.0, 523.25
        target_a, target_b = 523.25, 659.25
        signal = np.where(
            t_all < 6 * LIVE_BLOCK_SIZE / sr,
            0.5 * np.sin(2 * np.pi * freq_a * t_all),
            0.5 * np.sin(2 * np.pi * freq_b * t_all),
        ).astype(np.float32)

        shifter = PitchShifter()
        hops = []
        for b in range(n_blocks):
            start = b * LIVE_BLOCK_SIZE
            blk = signal[start : start + LIVE_BLOCK_SIZE]
            src = freq_a if b < 6 else freq_b
            tgt = target_a if b < 6 else target_b
            res = shifter.process_block(blk, src, tgt, source_voiced=True)
            if res is not None and res.valid:
                hops.append((b, res))

        # We need at least 3 valid hops to reach the transition-boundary hop.
        self.assertGreaterEqual(len(hops), 3, "Not enough valid hops to evaluate transition")

        # Hop index 2 (0-based): transition-boundary hop emitting B4&B5 (Note A audio).
        _, hop3 = hops[2]
        audio = hop3.audio.astype(np.float64)

        n_fft = 65536
        spectrum = np.abs(np.fft.rfft(audio * np.hanning(len(audio)), n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

        # Find dominant peak in [400, 800] Hz range.
        mask = (freqs >= 400) & (freqs <= 800)
        sub = spectrum.copy()
        sub[~mask] = 0.0
        p1_idx = int(np.argmax(sub))
        p1_freq = freqs[p1_idx]
        p1_mag = sub[p1_idx]

        # Mask P1 ± 50 Hz and find P2.
        sub2 = sub.copy()
        sub2[np.abs(freqs - p1_freq) < 50] = 0.0
        p2_idx = int(np.argmax(sub2))
        p2_freq = freqs[p2_idx]
        p2_mag = sub2[p2_idx]
        p2_rel = p2_mag / p1_mag if p1_mag > 0 else 0.0

        # P1 must be close to Note A target (523.25 Hz).
        cents_error = abs(1200 * np.log2(p1_freq / target_a)) if p1_freq > 0 else 9999
        self.assertLess(
            cents_error, 60,
            f"Hop 3 dominant peak {p1_freq:.1f} Hz is {cents_error:.0f} cents from target_a={target_a} Hz"
        )
        # P2 must not be a strong competing peak (dual-pitch contamination).
        self.assertLess(
            p2_rel, 0.30,
            f"Hop 3 has competing spectral peak at {p2_freq:.1f} Hz with relative amplitude {p2_rel:.2f}"
        )

    def test_invalid_block_size_is_rejected(self) -> None:
        shifter, _ = self.fake_shifter()
        with self.assertRaises(ValueError):
            shifter.process_block(np.zeros(LIVE_BLOCK_SIZE - 1, dtype=np.float32), 440.0, 660.0)


if __name__ == "__main__":
    unittest.main()
