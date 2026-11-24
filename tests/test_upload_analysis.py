import unittest
from threading import Event, Lock, Thread
from types import SimpleNamespace

import numpy as np

from harmonix.domain.enums import SessionMode
from harmonix.domain.models import AppState, KeyState, RecordedBuffers, SessionState
from harmonix.harmony_generator import HarmonyGenerator
from harmonix.key_scale_resolver import AnalysisMode, KeyResolutionResult, KeyScale, KeyScaleResolver
from harmonix.pitch_detector import PitchDetectionResult, PitchDetector, TrackingState
from harmonix.pitch_shifter import PitchShiftStatus, PitchShifter, WORLD_SEGMENT_SIZE
from harmonix.session_state.app_state import StateStore
from harmonix.session_state.commands import (
    CompleteUploadAnalysis,
    StartUploadAnalysis,
    UpdateUploadAnalysisProgress,
)
from harmonix.upload.analysis import UploadAnalysisWorker, UploadAnalyzer, _ProgressCoalescer
from harmonix.upload.errors import (
    AmbiguousUploadKeyError,
    InsufficientUploadKeyEvidenceError,
    SilentOrUnusableAudioError,
    UploadCancelledError,
    UploadPitchShiftError,
)
from harmonix.upload.loader import LoadedAudio


class FakeLoader:
    def __init__(self, audio: np.ndarray) -> None:
        self.audio = audio

    def load(self, _path, *, cancel_event=None, progress=None):
        if cancel_event is not None and cancel_event.is_set():
            raise UploadCancelledError("cancelled before load")
        if progress is not None:
            progress("decode", self.audio.size, self.audio.size)
        return LoadedAudio(
            audio=self.audio.copy(),
            source_sample_rate=44100,
            source_sample_count=self.audio.size,
            sample_rate=44100,
            duration_seconds=self.audio.size / 44100,
        )


class SequenceDetector:
    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def detect(self, _frame):
        result = self.results[self.calls]
        self.calls += 1
        return result


class FixedResolver:
    def __init__(self, *, ambiguous=False, insufficient=False, effective_key=None):
        self.ambiguous = ambiguous
        self.insufficient = insufficient
        self.effective_key = effective_key or KeyScale("C", "major")
        self.observations = []
        self.manual_key = None

    def set_manual_key(self, root, scale_type):
        self.manual_key = KeyScale(root, scale_type)

    def apply_processing_boundary(self):
        return None

    def analyze_upload(self, observations):
        self.observations = list(observations)
        effective = self.manual_key or self.effective_key
        return KeyResolutionResult(
            detected_key=effective,
            stabilized_key=effective,
            manual_key=self.manual_key,
            effective_key=effective,
            confidence=0.9,
            ambiguous=self.ambiguous,
            insufficient_evidence=self.insufficient,
            analysis_mode=AnalysisMode.MANUAL if self.manual_key else AnalysisMode.UPLOAD,
        )


class MappingHarmony:
    def __init__(self, targets=None):
        self.targets = targets or {}
        self.calls = []

    def generate(self, source, _key, voice, **_kwargs):
        self.calls.append((source, voice))
        target = self.targets.get(source, source + 110.0)
        return SimpleNamespace(valid=True, target_frequency_hz=target)


class FakeShifter:
    def __init__(self, *, fail_status=None):
        self.calls = []
        self.fail_status = fail_status

    def shift_full_buffer(self, audio, source, target, **_kwargs):
        self.calls.append((audio.copy(), source, target))
        if self.fail_status is not None:
            return SimpleNamespace(
                valid=False,
                audio=None,
                sample_rate=44100,
                output_sample_count=0,
                duration_preserved=False,
                status=self.fail_status,
            )
        output = np.full(audio.size, target - source, dtype=np.float32)
        return SimpleNamespace(
            valid=True,
            audio=output,
            sample_rate=44100,
            output_sample_count=audio.size,
            duration_preserved=True,
            status=PitchShiftStatus.VALID,
        )


class UploadAnalysisTests(unittest.TestCase):
    frame_size = 2048

    def pitch(self, frequency=440.0, *, voiced=True):
        return PitchDetectionResult(
            frequency_hz=frequency if voiced else None,
            confidence=0.9 if voiced else 0.0,
            voiced=voiced,
            tracking_state=TrackingState.TRACKING if voiced else TrackingState.TRACKING_LOST,
        )

    def analyzer(self, pitches, *, resolver_factory=None, harmony=None, shifters=None, audio=None):
        audio = audio if audio is not None else np.ones(self.frame_size * len(pitches) + 73, dtype=np.float32)
        detector = SequenceDetector(pitches)
        shifters = shifters or {voice: FakeShifter() for voice in ("3rd", "5th", "octave")}
        analyzer = UploadAnalyzer(
            loader=FakeLoader(audio),
            pitch_detector=detector,
            key_resolver_factory=resolver_factory or FixedResolver,
            harmony_generator=harmony or MappingHarmony(),
            pitch_shifters=shifters,
        )
        return analyzer, detector, shifters

    def test_complete_track_analysis_produces_aligned_original_and_selected_voice(self):
        pitches = [self.pitch() for _ in range(4)]
        resolver = FixedResolver()
        analyzer, detector, shifters = self.analyzer(pitches, resolver_factory=lambda: resolver)
        progress = []

        result = analyzer.analyze("tone.wav", ["3rd"], progress=lambda *args: progress.append(args))

        self.assertEqual(detector.calls, 4)
        self.assertEqual(len(resolver.observations), 4)
        self.assertEqual(result.frame_count, 4)
        self.assertEqual(result.recorded_buffers.sample_rate, 44100)
        self.assertEqual(result.recorded_buffers.original.dtype, np.float32)
        self.assertEqual(result.recorded_buffers.original.size, 4 * self.frame_size + 73)
        self.assertEqual(set(result.recorded_buffers.a_per_voice), {"3rd", "5th", "octave"})
        np.testing.assert_array_equal(
            result.recorded_buffers.a_per_voice["3rd"][: 4 * self.frame_size],
            np.full(4 * self.frame_size, 110.0, dtype=np.float32),
        )
        np.testing.assert_array_equal(
            result.recorded_buffers.a_per_voice["3rd"][-73:], np.zeros(73, dtype=np.float32)
        )
        for inactive in ("5th", "octave"):
            np.testing.assert_array_equal(result.recorded_buffers.a_per_voice[inactive], np.zeros_like(result.recorded_buffers.original))
        self.assertTrue(any(phase == "key_analysis" for phase, _, _ in progress))
        self.assertEqual(len(shifters["3rd"].calls), 1)

    def test_complete_track_real_m5_and_existing_m6_target_are_used(self):
        pitch_classes = [0] * 8 + [2, 4, 5, 7, 9, 11] + [0] * 4 + [7, 4] * 3
        pitches = [self.pitch(440.0 * 2.0 ** ((pitch_class - 9) / 12.0)) for pitch_class in pitch_classes]
        audio = np.ones(self.frame_size * len(pitches), dtype=np.float32)
        shifters = {voice: FakeShifter() for voice in ("3rd", "5th", "octave")}
        analyzer = UploadAnalyzer(
            loader=FakeLoader(audio),
            pitch_detector=SequenceDetector(pitches),
            key_resolver_factory=FixedResolver,
            harmony_generator=HarmonyGenerator(),
            pitch_shifters=shifters,
        )

        result = analyzer.analyze("c-major.wav", ["3rd"])

        self.assertEqual(result.key_resolution.effective_key, KeyScale("C", "major"))
        first_c4 = 440.0 * 2.0 ** ((0 - 9) / 12.0)
        expected = HarmonyGenerator().generate(first_c4, KeyScale("C", "major"), "3rd").target_frequency_hz
        self.assertAlmostEqual(shifters["3rd"].calls[0][2], expected)

    def test_variable_targets_form_separate_regions_without_boundary_blending(self):
        pitches = [self.pitch(440.0), self.pitch(440.0), self.pitch(220.0), self.pitch(220.0)]
        shifters = {voice: FakeShifter() for voice in ("3rd", "5th", "octave")}
        analyzer, _, _ = self.analyzer(pitches, harmony=MappingHarmony({440.0: 660.0, 220.0: 330.0}), shifters=shifters)

        result = analyzer.analyze("regions.wav", ["3rd"])

        self.assertEqual(len(shifters["3rd"].calls), 2)
        self.assertEqual([(call[1], call[2]) for call in shifters["3rd"].calls], [(440.0, 660.0), (220.0, 330.0)])
        voice = result.recorded_buffers.a_per_voice["3rd"]
        np.testing.assert_array_equal(voice[: 2 * self.frame_size], np.full(2 * self.frame_size, 220.0, dtype=np.float32))
        np.testing.assert_array_equal(voice[2 * self.frame_size : 4 * self.frame_size], np.full(2 * self.frame_size, 110.0, dtype=np.float32))

    def test_unvoiced_frames_remain_zero_and_split_target_regions(self):
        pitches = [self.pitch(), self.pitch(voiced=False), self.pitch()]
        shifter = FakeShifter()
        analyzer, _, _ = self.analyzer(pitches, shifters={"3rd": shifter})

        result = analyzer.analyze("gap.wav", ["3rd"])

        self.assertEqual(len(shifter.calls), 2)
        voice = result.recorded_buffers.a_per_voice["3rd"]
        np.testing.assert_array_equal(voice[self.frame_size : 2 * self.frame_size], np.zeros(self.frame_size, dtype=np.float32))

    def test_zero_voices_preserves_original_and_zero_fills_every_voice(self):
        pitches = [self.pitch() for _ in range(2)]
        analyzer, _, shifters = self.analyzer(pitches)

        result = analyzer.analyze("original-only.wav", [])

        np.testing.assert_array_equal(result.recorded_buffers.original, np.ones(self.frame_size * 2 + 73, dtype=np.float32))
        self.assertTrue(all(not shifter.calls for shifter in shifters.values()))
        for voice in result.recorded_buffers.a_per_voice.values():
            np.testing.assert_array_equal(voice, np.zeros_like(result.recorded_buffers.original))

    def test_silent_unusable_audio_has_distinct_failure(self):
        pitches = [self.pitch(voiced=False) for _ in range(3)]
        analyzer, _, _ = self.analyzer(pitches)

        with self.assertRaises(SilentOrUnusableAudioError):
            analyzer.analyze("silence.wav", ["3rd"])

    def test_ambiguous_and_insufficient_key_evidence_are_not_guessed(self):
        for resolver, error in (
            (FixedResolver(ambiguous=True), AmbiguousUploadKeyError),
            (FixedResolver(insufficient=True), InsufficientUploadKeyEvidenceError),
        ):
            with self.subTest(error=error.__name__):
                analyzer, _, _ = self.analyzer([self.pitch()] * 2, resolver_factory=lambda: resolver)
                with self.assertRaises(error):
                    analyzer.analyze("uncertain.wav", ["3rd"])

    def test_m7_failure_propagates_as_upload_pitch_shift_failure(self):
        failed = FakeShifter(fail_status=PitchShiftStatus.WORLD_ANALYSIS_ERROR)
        analyzer, _, _ = self.analyzer([self.pitch()] * 2, shifters={"3rd": failed})

        with self.assertRaisesRegex(UploadPitchShiftError, "world_analysis_error"):
            analyzer.analyze("failure.wav", ["3rd"])

    def test_cancellation_during_pitch_analysis_stops_before_key_resolution(self):
        cancel = Event()

        class CancellingDetector(SequenceDetector):
            def detect(self, frame):
                result = super().detect(frame)
                cancel.set()
                return result

        audio = np.ones(self.frame_size * 3, dtype=np.float32)
        resolver = FixedResolver()
        analyzer = UploadAnalyzer(
            loader=FakeLoader(audio),
            pitch_detector=CancellingDetector([self.pitch()] * 3),
            key_resolver_factory=lambda: resolver,
            harmony_generator=MappingHarmony(),
            pitch_shifters={"3rd": FakeShifter()},
        )

        with self.assertRaises(UploadCancelledError):
            analyzer.analyze("cancel.wav", ["3rd"], cancel_event=cancel)
        self.assertEqual(resolver.observations, [])

    def test_progress_advances_from_actual_phase_work(self):
        analyzer, _, _ = self.analyzer([self.pitch()] * 3)
        progress = []

        analyzer.analyze("progress.wav", ["3rd"], progress=lambda *update: progress.append(update))

        by_phase = {}
        for phase, completed, total in progress:
            self.assertGreaterEqual(completed, 0)
            if total is not None:
                self.assertGreaterEqual(total, completed)
            by_phase.setdefault(phase, []).append(completed)
        self.assertTrue({"decode", "pitch_analysis", "key_analysis", "harmony_generation", "prerender"}.issubset(by_phase))
        self.assertTrue(all(values == sorted(values) for values in by_phase.values()))
        final_by_phase = {}
        for phase, completed, total in progress:
            final_by_phase[phase] = (completed, total)
        for completed, total in final_by_phase.values():
            self.assertIsNotNone(total)
            self.assertEqual(completed, total)

    def test_progress_dispatch_count_is_bounded_and_final_counters_are_exact(self):
        published = []
        coalescer = _ProgressCoalescer(lambda *snapshot: published.append(snapshot))

        for completed in range(1, 10001):
            coalescer.report("pitch_analysis", completed, 10000)
        coalescer.flush()

        self.assertLessEqual(len(published), 100)
        self.assertEqual(published[-1], ("pitch_analysis", 10000, 10000))

    def test_progress_flushes_partial_phase_snapshot_on_phase_change(self):
        published = []
        coalescer = _ProgressCoalescer(lambda *snapshot: published.append(snapshot))

        coalescer.report("pitch_analysis", 5, 500)
        coalescer.report("pitch_analysis", 6, 500)
        coalescer.report("key_analysis", 2, 8)

        self.assertEqual(published[0], ("pitch_analysis", 5, 500))
        self.assertEqual(published[1], ("pitch_analysis", 6, 500))
        self.assertEqual(published[2], ("key_analysis", 2, 8))
        self.assertEqual(coalescer.intermediate_dispatches, 1)

    def test_underreported_decode_total_changes_without_exceeding_fixed_budget(self):
        published = []
        coalescer = _ProgressCoalescer(lambda *snapshot: published.append(snapshot))
        provisional_updates = [(2, 1), (7, 3), (8, 3), (13, 8), (14, 8), (22, 17), (23, 17)]

        for completed, provisional_total in provisional_updates:
            coalescer.report("decode", completed, provisional_total)
        coalescer.finish_phase("decode", 1000, 1000)

        self.assertLessEqual(len(published), 100)
        self.assertEqual(
            published,
            [
                ("decode", 2, 1),
                ("decode", 8, 3),
                ("decode", 14, 8),
                ("decode", 23, 17),
                ("decode", 1000, 1000),
            ],
        )
        self.assertIn(("decode", 1000, 1000), published)
        self.assertEqual(published[-1], ("decode", 1000, 1000))
        self.assertGreater(published[0][1], published[0][2])

    def test_reducer_preserves_supplied_completed_and_underreported_total(self):
        store = StateStore(AppState(session=SessionState(mode=SessionMode.UPLOAD)))
        store.dispatch(StartUploadAnalysis("request", "file.wav", ("3rd",)))

        store.dispatch(UpdateUploadAnalysisProgress("request", "decode", 12, 5))

        upload_state = store.snapshot().session.upload_analysis
        self.assertEqual(upload_state.completed_units, 12)
        self.assertEqual(upload_state.total_units, 5)

    def test_upload_analyzer_is_locked_to_2048_sample_frames(self):
        with self.assertRaises(TypeError):
            UploadAnalyzer(frame_size=1024)

        class WrongFrameDetector(SequenceDetector):
            frame_size = 1024

        with self.assertRaisesRegex(ValueError, "2048-sample"):
            UploadAnalyzer(pitch_detector=WrongFrameDetector([self.pitch()]))

    def test_each_analysis_uses_a_fresh_m5_resolver(self):
        created = []
        audio = np.ones(self.frame_size * 4, dtype=np.float32)

        class RepeatingDetector:
            frame_size = 2048

            def detect(_self, _frame):
                return self.pitch()

        def make_resolver():
            resolver = FixedResolver()
            created.append(resolver)
            return resolver

        analyzer = UploadAnalyzer(
            loader=FakeLoader(audio),
            pitch_detector=RepeatingDetector(),
            key_resolver_factory=make_resolver,
            harmony_generator=MappingHarmony(),
            pitch_shifters={"3rd": FakeShifter()},
        )
        analyzer.analyze("first.wav", ["3rd"], manual_key=KeyScale("D", "minor"))
        analyzer.analyze("second.wav", ["3rd"])

        self.assertEqual(len(created), 2)
        self.assertIsNot(created[0], created[1])
        self.assertEqual(created[0].manual_key, KeyScale("D", "minor"))
        self.assertIsNone(created[1].manual_key)
        self.assertEqual(len(created[0].observations), 4)
        self.assertEqual(len(created[1].observations), 4)

    def test_unknown_total_progress_uses_logarithmic_dispatches(self):
        published = []
        coalescer = _ProgressCoalescer(lambda *snapshot: published.append(snapshot))

        for completed in range(1, 10001):
            coalescer.report("decode", completed, None)
        coalescer.flush()

        self.assertLessEqual(len(published), 16)
        self.assertEqual(published[-1], ("decode", 10000, None))

    def test_real_m4_m6_m7_upload_orchestration_runs_offline(self):
        sample_count = WORLD_SEGMENT_SIZE
        time = np.arange(sample_count, dtype=np.float64) / 44100.0
        audio = (0.5 * np.sin(2.0 * np.pi * 440.0 * time)).astype(np.float32)

        class CountingPitchDetector(PitchDetector):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def detect(self, frame):
                self.calls += 1
                return super().detect(frame)

        class CountingPitchShifter(PitchShifter):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def shift_full_buffer(self, *args, **kwargs):
                self.calls += 1
                return super().shift_full_buffer(*args, **kwargs)

        detector = CountingPitchDetector()
        shifter = CountingPitchShifter()
        analyzer = UploadAnalyzer(
            loader=FakeLoader(audio),
            pitch_detector=detector,
            key_resolver_factory=FixedResolver,
            harmony_generator=HarmonyGenerator(),
            pitch_shifters={"3rd": shifter},
        )

        result = analyzer.analyze("synthetic-c4.wav", ["3rd"])

        self.assertEqual(detector.calls, sample_count // 2048)
        self.assertGreater(shifter.calls, 0)
        self.assertEqual(result.recorded_buffers.original.size, sample_count)
        self.assertEqual(result.recorded_buffers.a_per_voice["3rd"].size, sample_count)
        self.assertEqual(result.recorded_buffers.sample_rate, 44100)
        self.assertTrue(np.isfinite(result.recorded_buffers.a_per_voice["3rd"]).all())

    def test_worker_success_commits_canonical_recorded_buffers(self):
        from harmonix.upload.analysis import UploadAnalysisResult

        key = KeyScale("C", "major")
        resolution = KeyResolutionResult(key, key, None, key, 0.9, False, False, AnalysisMode.UPLOAD)
        original = np.ones(32, dtype=np.float32)
        voices = {voice: np.zeros(32, dtype=np.float32) for voice in ("3rd", "5th", "octave")}
        buffers = __import__("harmonix.domain.models", fromlist=["RecordedBuffers"]).RecordedBuffers(
            original=original,
            a_per_voice=voices,
            sample_rate=44100,
        )

        class SuccessfulAnalyzer:
            def analyze(self, *_args, **_kwargs):
                return UploadAnalysisResult(buffers, resolution, 32 / 44100, 1)

        store = StateStore(AppState(session=SessionState(mode=SessionMode.UPLOAD)))
        worker = UploadAnalysisWorker(store, SuccessfulAnalyzer())
        worker.start("complete.wav", ["3rd"])
        worker.join(2.0)

        session = store.snapshot().session
        self.assertFalse(worker.is_alive)
        self.assertEqual(session.upload_analysis.status, "completed")
        np.testing.assert_array_equal(session.recorded_buffers.original, original)
        self.assertEqual(session.recorded_buffers.sample_rate, 44100)
        self.assertEqual(session.playback.duration_seconds, 32 / 44100)

    def test_worker_failure_commits_failure_without_partial_buffers(self):
        class FailingAnalyzer:
            def analyze(self, *_args, **_kwargs):
                raise UploadPitchShiftError("M7 region failed")

        store = StateStore(AppState(session=SessionState(mode=SessionMode.UPLOAD)))
        worker = UploadAnalysisWorker(store, FailingAnalyzer())
        worker.start("failed.wav", ["3rd"])
        worker.join(2.0)

        session = store.snapshot().session
        self.assertFalse(worker.is_alive)
        self.assertEqual(session.upload_analysis.status, "failed")
        self.assertEqual(session.upload_analysis.error_code, "UploadPitchShiftError")
        self.assertIsNone(session.recorded_buffers.original)

    def test_new_analysis_clears_old_playback_metadata_through_cancel_and_failure(self):
        initial_session = SessionState(mode=SessionMode.UPLOAD)
        initial_session.playback.position_seconds = 12.0
        initial_session.playback.duration_seconds = 42.0
        initial_session.playback.pending_seek_seconds = 8.0
        initial_session.playback.is_playing = True

        for terminal in ("cancelled", "failed"):
            with self.subTest(terminal=terminal):
                store = StateStore(AppState(session=initial_session))
                store.dispatch(StartUploadAnalysis("request", "new.wav", ("3rd",)))
                store.dispatch(UpdateUploadAnalysisProgress("request", "decode", 1, 1))
                state = store.snapshot().session
                self.assertEqual(state.playback.position_seconds, 0.0)
                self.assertEqual(state.playback.duration_seconds, 0.0)
                self.assertIsNone(state.playback.pending_seek_seconds)
                if terminal == "cancelled":
                    from harmonix.session_state.commands import CancelUploadAnalysis, RequestUploadCancellation

                    store.dispatch(RequestUploadCancellation("request"))
                    store.dispatch(CancelUploadAnalysis("request"))
                else:
                    from harmonix.session_state.commands import FailUploadAnalysis

                    store.dispatch(FailUploadAnalysis("request", "test_failure", "failed"))
                final_state = store.snapshot().session
                self.assertEqual(final_state.playback.position_seconds, 0.0)
                self.assertEqual(final_state.playback.duration_seconds, 0.0)
                self.assertIsNone(final_state.recorded_buffers.original)

    def test_worker_cancellation_is_canonical_and_discards_partial_result(self):
        started = Event()

        class WaitingAnalyzer:
            def analyze(self, _path, _voices, *, manual_key=None, cancel_event=None, progress=None):
                started.set()
                while not cancel_event.is_set():
                    Event().wait(0.005)
                raise UploadCancelledError("cancelled")

        initial = AppState(session=SessionState(mode=SessionMode.UPLOAD))
        store = StateStore(initial)
        worker = UploadAnalysisWorker(store, WaitingAnalyzer())
        request_id = worker.start("waiting.wav", ["3rd"])
        self.assertTrue(started.wait(2.0))

        worker.request_cancel()
        worker.join(2.0)

        self.assertFalse(worker.is_alive)
        session = store.snapshot().session
        self.assertEqual(session.upload_analysis.request_id, request_id)
        self.assertEqual(session.upload_analysis.status, "cancelled")
        self.assertTrue(session.upload_analysis.cancellation_requested)
        self.assertIsNone(session.recorded_buffers.original)

    def test_new_render_supersedes_running_request_for_same_upload(self):
        entered = [Event(), Event()]
        cancelled = [Event(), Event()]
        calls = {"count": 0}
        calls_lock = Lock()

        class WaitingAnalyzer:
            def analyze(self, path, _voices, *, manual_key=None, cancel_event=None, progress=None):
                with calls_lock:
                    call_index = calls["count"]
                    calls["count"] += 1
                entered[call_index].set()
                while not cancel_event.is_set():
                    Event().wait(0.005)
                cancelled[call_index].set()
                raise UploadCancelledError("superseded")

        store = StateStore()
        worker = UploadAnalysisWorker(store, WaitingAnalyzer())
        first_request = worker.start("same.wav", ["3rd"])
        self.assertTrue(entered[0].wait(2.0))
        worker.start(
            "same.wav",
            ["3rd"],
            continuation_request_id=first_request,
        )
        self.assertTrue(entered[1].wait(2.0))
        self.assertTrue(cancelled[0].wait(2.0))
        self.assertEqual(store.snapshot().session.upload_analysis.status, "running")
        self.assertTrue(worker.is_alive)
        worker.request_cancel()
        worker.join(2.0)
        self.assertFalse(worker.is_alive)
        self.assertTrue(cancelled[1].is_set())
        self.assertEqual(store.snapshot().session.upload_analysis.request_id, worker._request_id)

    def test_canonical_progress_is_monotone_within_phase_and_resets_between_phases(self):
        store = StateStore(AppState(session=SessionState(mode=SessionMode.UPLOAD)))
        store.dispatch(StartUploadAnalysis("request", "file.wav", ("3rd",)))
        store.dispatch(UpdateUploadAnalysisProgress("request", "decode", 100, 100))
        store.dispatch(UpdateUploadAnalysisProgress("request", "decode", 50, 100))
        self.assertEqual(store.snapshot().session.upload_analysis.completed_units, 100)

        store.dispatch(UpdateUploadAnalysisProgress("request", "pitch_analysis", 1, 3))
        self.assertEqual(store.snapshot().session.upload_analysis.completed_units, 1)
        self.assertEqual(store.snapshot().session.upload_analysis.phase, "pitch_analysis")

    def test_stale_key_render_cannot_replace_latest_request(self):
        store = StateStore()
        store.dispatch(StartUploadAnalysis("first", "file.wav", ("3rd",)))
        store.dispatch(StartUploadAnalysis("latest", "file.wav", ("3rd",), continuation_request_id="first"))
        stale = RecordedBuffers(original=np.ones(8, dtype=np.float32), a_per_voice={}, sample_rate=44100)
        store.dispatch(
            CompleteUploadAnalysis(
                "first", stale, KeyState(root="C", scale_type="major"), 1.0,
                KeyState(root="G", scale_type="major"),
            )
        )
        session = store.snapshot().session
        self.assertEqual(session.upload_analysis.request_id, "latest")
        self.assertIsNone(session.recorded_buffers.original)

    def test_key_and_matching_buffers_commit_together_and_start_disables_old_playback(self):
        store = StateStore(AppState(session=SessionState(mode=SessionMode.UPLOAD)))
        old = RecordedBuffers(
            original=np.zeros(4, dtype=np.float32),
            a_per_voice={"3rd": np.zeros(4, dtype=np.float32)},
            sample_rate=44100,
        )
        store.dispatch(StartUploadAnalysis("first", "file.wav", ("3rd",)))
        store.dispatch(
            CompleteUploadAnalysis("first", old, KeyState("D", "minor", False), 1.0, KeyState("C", "major"))
        )
        self.assertEqual(store.snapshot().session.key.root, "D")
        self.assertEqual(store.snapshot().session.detected_key.root, "C")
        self.assertIsNotNone(store.playback_source())

        store.dispatch(StartUploadAnalysis("second", "file.wav", ("3rd",), continuation_request_id="first"))
        self.assertIsNone(store.playback_source())
        self.assertIsNone(store.snapshot().session.recorded_buffers.original)


if __name__ == "__main__":
    unittest.main()
