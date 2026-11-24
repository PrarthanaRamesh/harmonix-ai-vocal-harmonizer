"""Loader tests; the MP3 integration fixture is Pygame-distribution-dependent."""

import tempfile
import unittest
from pathlib import Path
from threading import Event
from unittest.mock import patch

import numpy as np
import pygame
import soundfile as sf

from harmonix.input.adapter import CANONICAL_SAMPLE_RATE, DEFAULT_FRAME_SIZE
from harmonix.upload.errors import (
    CorruptUploadError,
    InvalidUploadDurationError,
    UploadCancelledError,
    UploadDurationExceededError,
    UnsupportedUploadFormatError,
)
from harmonix.upload.loader import UploadLoader, prepare_canonical_frames, validate_decoded_duration


class UploadLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.loader = UploadLoader(decode_chunk_frames=1024)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def write_audio(self, suffix: str, audio: np.ndarray, sample_rate: int) -> Path:
        path = self.root / f"fixture.{suffix}"
        sf.write(path, audio, sample_rate, format=suffix.upper())
        return path

    def tone(self, sample_rate: int = 22050, samples: int = 4096) -> np.ndarray:
        time = np.arange(samples, dtype=np.float32) / sample_rate
        return np.sin(2.0 * np.pi * 440.0 * time).astype(np.float32)

    def test_loads_wav_and_flac_as_canonical_mono_float32(self) -> None:
        source = self.tone()

        for extension in ("wav", "flac"):
            with self.subTest(extension=extension):
                result = self.loader.load(self.write_audio(extension, source, 22050))

                self.assertEqual(result.sample_rate, CANONICAL_SAMPLE_RATE)
                self.assertEqual(result.audio.dtype, np.float32)
                self.assertEqual(result.audio.ndim, 1)
                self.assertEqual(result.source_sample_count, source.size)
                self.assertAlmostEqual(result.duration_seconds, source.size / 22050)
                self.assertEqual(result.audio.size, round(source.size * CANONICAL_SAMPLE_RATE / 22050))

    def test_decodes_real_mp3_from_locked_pygame_distribution(self) -> None:
        mp3_path = Path(pygame.__file__).parent / "examples" / "data" / "house_lo.mp3"
        if not mp3_path.is_file():
            self.skipTest(
                "Environment-dependent integration test: Pygame's bundled house_lo.mp3 is unavailable; "
                "the asset is not copied or redistributed."
            )

        result = self.loader.load(mp3_path)

        self.assertEqual(result.sample_rate, CANONICAL_SAMPLE_RATE)
        self.assertEqual(result.audio.dtype, np.float32)
        self.assertGreater(result.source_sample_count, 0)
        self.assertAlmostEqual(result.duration_seconds, 80064 / 11025)

    def test_rejects_unsupported_extension(self) -> None:
        path = self.root / "fixture.ogg"
        path.write_bytes(b"not audio")

        with self.assertRaises(UnsupportedUploadFormatError):
            self.loader.load(path)

    def test_corrupt_supported_file_is_distinct_from_unsupported_format(self) -> None:
        path = self.root / "fixture.wav"
        path.write_bytes(b"not a wav")

        with self.assertRaises(CorruptUploadError):
            self.loader.load(path)

    def test_multichannel_downmix_is_deterministic(self) -> None:
        source = np.column_stack((self.tone(), np.zeros(4096, dtype=np.float32)))
        path = self.write_audio("wav", source, CANONICAL_SAMPLE_RATE)

        result = self.loader.load(path)

        expected = sf.read(path, dtype="float32", always_2d=True)[0].mean(axis=1, dtype=np.float32)
        np.testing.assert_allclose(result.audio, expected, rtol=0.0, atol=1e-6)

    def test_decoded_duration_limits_are_exact(self) -> None:
        self.assertEqual(validate_decoded_duration(600 * 44100, 44100), 600.0)
        with self.assertRaises(UploadDurationExceededError):
            validate_decoded_duration(600 * 44100 + 1, 44100)
        with self.assertRaises(InvalidUploadDurationError):
            validate_decoded_duration(0, 44100)

    def test_loader_stops_after_decoded_audio_exceeds_limit(self) -> None:
        path = self.write_audio("wav", self.tone(100, 1024), 100)
        with patch("harmonix.upload.loader.MAX_UPLOAD_DURATION_SECONDS", 1.0):
            with self.assertRaises(UploadDurationExceededError):
                self.loader.load(path)

    def test_cancel_is_observed_between_decode_chunks(self) -> None:
        path = self.write_audio("wav", self.tone(22050, 4096), 22050)
        cancel = Event()

        def cancel_after_progress(phase: str, completed: int, total: int) -> None:
            self.assertEqual(phase, "decode")
            self.assertGreater(completed, 0)
            self.assertGreaterEqual(total, completed)
            cancel.set()

        with self.assertRaises(UploadCancelledError):
            self.loader.load(path, cancel_event=cancel, progress=cancel_after_progress)

    def test_decode_progress_is_based_on_decoded_frames(self) -> None:
        path = self.write_audio("wav", self.tone(22050, 4096), 22050)
        progress: list[tuple[int, int]] = []

        self.loader.load(path, progress=lambda phase, done, total: progress.append((done, total)))

        self.assertGreaterEqual(len(progress), 4)
        self.assertEqual([done for done, _ in progress], sorted(done for done, _ in progress))
        self.assertTrue(all(done <= total for done, total in progress))

    def test_canonical_frames_are_nonoverlapping_and_leave_partial_tail_unframed(self) -> None:
        source = np.arange(DEFAULT_FRAME_SIZE + 300, dtype=np.float32)

        frames = prepare_canonical_frames(source)

        self.assertEqual([frame.sample_start for frame in frames], [0])
        for frame in frames:
            self.assertEqual(frame.audio.size, DEFAULT_FRAME_SIZE)
            np.testing.assert_array_equal(frame.audio, source[frame.sample_start : frame.sample_start + DEFAULT_FRAME_SIZE])
        np.testing.assert_array_equal(source[DEFAULT_FRAME_SIZE:], np.arange(DEFAULT_FRAME_SIZE, DEFAULT_FRAME_SIZE + 300, dtype=np.float32))