import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from harmonix.domain.errors import PlaybackSourceError
from harmonix.domain.models import AppState, PlaybackState, RecordedBuffers, SessionState
from harmonix.domain.enums import SessionMode
from harmonix.playback_engine import PlaybackController, PlaybackEngine
from harmonix.session_state.app_state import StateStore
from harmonix.session_state.commands import (
    PlaybackPause,
    PlaybackPlay,
    PlaybackRecovered,
    PlaybackSeek,
    PlaybackSeekApplied,
    PlaybackSetVoices,
)


def buffers(sample_count=32, *, sample_rate=10):
    original = np.full(sample_count, 0.2, dtype=np.float32)
    voices = {
        "3rd": np.full(sample_count, 0.4, dtype=np.float32),
        "5th": np.full(sample_count, 0.6, dtype=np.float32),
        "octave": np.full(sample_count, 0.8, dtype=np.float32),
    }
    return RecordedBuffers(original, voices, sample_rate)


class FakeOutputStream:
    def __init__(self, callback, *, fail_start=False, **_kwargs):
        self.callback = callback
        self.fail_start = fail_start
        self.started = False
        self.stopped = False
        self.closed = False

    def start(self):
        if self.fail_start:
            raise RuntimeError("injected output start failure")
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True

    def emit(self, frames=2, status=None):
        output = np.zeros((frames, 1), dtype=np.float32)
        self.callback(output, frames, {}, status)
        return output[:, 0]


class PlaybackEngineTests(unittest.TestCase):
    def engine(self, sample_count=32, sample_rate=10, *, voices=(), enabled=()):
        return PlaybackEngine(buffers(sample_count, sample_rate=sample_rate), voices, initially_enabled=enabled)

    def test_all_enabled_tracks_read_the_same_sample_range(self):
        source = buffers(8)
        source.original[:] = np.arange(8, dtype=np.float32)
        source.a_per_voice["3rd"][:] = np.arange(8, dtype=np.float32) + 10
        source.a_per_voice["5th"][:] = np.arange(8, dtype=np.float32) + 20
        engine = PlaybackEngine(source, ("3rd", "5th"), initially_enabled=("3rd", "5th"))
        engine.enqueue(PlaybackPlay())

        first = engine.render_block(3)
        second = engine.render_block(2)

        np.testing.assert_allclose(first, np.mean([source.original[:3], source.a_per_voice["3rd"][:3], source.a_per_voice["5th"][:3]], axis=0))
        np.testing.assert_allclose(second, np.mean([source.original[3:5], source.a_per_voice["3rd"][3:5], source.a_per_voice["5th"][3:5]], axis=0))
        self.assertEqual(engine.sample_offset, 5)

    def test_play_pause_resume_preserves_exact_sample_offset(self):
        engine = self.engine()
        engine.enqueue(PlaybackPlay())
        engine.render_block(7)
        engine.enqueue(PlaybackPause())
        engine.render_block(20)
        self.assertEqual(engine.sample_offset, 7)
        self.assertFalse(engine.is_playing)

        engine.enqueue(PlaybackPlay())
        engine.render_block(3)
        self.assertEqual(engine.sample_offset, 10)
        self.assertTrue(engine.is_playing)

    def test_repeated_pause_resume_cycles_preserve_clock(self):
        engine = self.engine(sample_count=100)
        engine.enqueue(PlaybackPlay())
        for _ in range(20):
            engine.render_block(1)
            frozen = engine.sample_offset
            engine.enqueue(PlaybackPause())
            engine.render_block(5)
            self.assertEqual(engine.sample_offset, frozen)
            engine.enqueue(PlaybackPlay())
            engine.render_block(1)
            self.assertEqual(engine.sample_offset, frozen + 1)

    def test_seek_rounding_and_clamping(self):
        engine = self.engine(sample_count=10, sample_rate=10)
        for request, seconds, expected in (
            ("rounded", 0.24, 2),
            ("negative", -4.0, 0),
            ("past_end", 4.0, 10),
        ):
            with self.subTest(request=request):
                engine.enqueue(PlaybackSeek(request, seconds))
                engine.render_block(1)
                self.assertEqual(engine.sample_offset, expected)

    def test_seek_while_paused_moves_clock_without_starting(self):
        engine = self.engine()
        engine.enqueue(PlaybackSeek("paused-seek", 1.2))
        output = engine.render_block(4)

        self.assertEqual(engine.sample_offset, 12)
        self.assertFalse(engine.is_playing)
        np.testing.assert_array_equal(output, np.zeros(4, dtype=np.float32))

    def test_latest_of_25_rapid_seeks_wins_and_only_latest_is_acknowledged(self):
        engine = self.engine(sample_count=100, sample_rate=10)
        for index in range(25):
            engine.enqueue(PlaybackSeek(f"seek-{index}", index / 10))
        engine.render_block(1)

        self.assertEqual(engine.sample_offset, 24)
        seek_acks = [event for event in engine.drain_events() if isinstance(event, PlaybackSeekApplied)]
        self.assertEqual(len(seek_acks), 1)
        self.assertEqual(seek_acks[0].request_id, "seek-24")
        self.assertEqual(seek_acks[0].position_seconds, 2.4)

    def test_stale_seek_acknowledgement_cannot_clear_newer_request(self):
        session = SessionState(mode=SessionMode.UPLOAD)
        session.playback = PlaybackState(pending_seek_seconds=5.0, pending_seek_request_id="new")
        store = StateStore(AppState(session=session))

        store.publish_playback_event(PlaybackSeekApplied("old", 1.0, False))
        state = store.snapshot().session.playback

        self.assertEqual(state.pending_seek_request_id, "new")
        self.assertEqual(state.pending_seek_seconds, 5.0)
        self.assertEqual(state.position_seconds, 0.0)

    def test_voice_toggle_does_not_change_sample_position(self):
        engine = self.engine(voices=("3rd",), enabled=("3rd",))
        engine.enqueue(PlaybackPlay())
        engine.render_block(4)
        before = engine.sample_offset
        engine.enqueue(PlaybackSetVoices(()))
        engine.render_block(3)

        self.assertEqual(engine.sample_offset, before + 3)
        self.assertEqual(engine._target_gains["3rd"], 0.0)

    def test_voice_activation_and_deactivation_fade_over_150ms(self):
        source = buffers(1000, sample_rate=1000)
        source.original.fill(0.0)
        source.a_per_voice["3rd"].fill(1.0)
        engine = PlaybackEngine(source, ("3rd",))
        engine.enqueue(PlaybackSetVoices(("3rd",)))
        engine.enqueue(PlaybackPlay())
        engine.render_block(75)
        self.assertAlmostEqual(engine._gains["3rd"], 0.5)
        engine.render_block(75)
        self.assertEqual(engine._gains["3rd"], 1.0)
        engine.enqueue(PlaybackSetVoices(()))
        engine.render_block(75)
        self.assertAlmostEqual(engine._gains["3rd"], 0.5)
        engine.render_block(75)
        self.assertEqual(engine._gains["3rd"], 0.0)

    def test_unrendered_voice_cannot_be_activated_even_if_buffer_is_nonzero(self):
        engine = self.engine(voices=("3rd",))
        engine.enqueue(PlaybackSetVoices(("5th",)))
        engine.render_block(1)
        self.assertEqual(engine._target_gains["3rd"], 0.0)
        errors = [event for event in engine.drain_events() if getattr(event, "error_code", None) == "PlaybackVoiceError"]
        self.assertEqual(len(errors), 1)

    def test_zero_voices_outputs_original_only(self):
        source = buffers(4)
        source.original[:] = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        engine = PlaybackEngine(source, ())
        engine.enqueue(PlaybackPlay())

        np.testing.assert_array_equal(engine.render_block(4), source.original)

    def test_end_of_track_stops_at_exact_length_and_play_restarts_at_zero(self):
        source = buffers(5)
        source.original[:] = np.arange(5, dtype=np.float32)
        engine = PlaybackEngine(source, ())
        engine.enqueue(PlaybackPlay())
        np.testing.assert_array_equal(engine.render_block(4), source.original[:4])
        final = engine.render_block(4)

        np.testing.assert_array_equal(final, np.array([4, 0, 0, 0], dtype=np.float32))
        self.assertEqual(engine.sample_offset, 5)
        self.assertFalse(engine.is_playing)
        self.assertTrue(engine.ended)

        engine.enqueue(PlaybackPlay())
        np.testing.assert_array_equal(engine.render_block(2), source.original[:2])
        self.assertEqual(engine.sample_offset, 2)
        self.assertFalse(engine.ended)

    def test_invalid_and_mismatched_sources_fail_explicitly(self):
        valid = buffers(4)
        cases = []
        cases.append(("missing original", RecordedBuffers(None, valid.a_per_voice, 10), ()))
        cases.append(("bad sample rate", RecordedBuffers(valid.original, valid.a_per_voice, 0), ()))
        cases.append(("empty original", RecordedBuffers(np.zeros(0, dtype=np.float32), {}, 10), ()))
        cases.append(("multichannel", RecordedBuffers(np.zeros((4, 2), dtype=np.float32), {}, 10), ()))
        invalid_float = np.ones(4, dtype=np.float32)
        invalid_float[2] = np.nan
        cases.append(("non-finite", RecordedBuffers(invalid_float, {}, 10), ()))
        mismatched = dict(valid.a_per_voice)
        mismatched["3rd"] = np.zeros(3, dtype=np.float32)
        cases.append(("mismatched voice", RecordedBuffers(valid.original, mismatched, 10), ()))
        cases.append(("missing rendered buffer", RecordedBuffers(valid.original, {}, 10), ("3rd",)))
        for label, source, rendered in cases:
            with self.subTest(label=label), self.assertRaises(PlaybackSourceError):
                PlaybackEngine(source, rendered)

    def test_output_failure_aborts_prepared_block_and_freezes_clock(self):
        engine = self.engine(voices=())
        engine.enqueue(PlaybackPlay())
        engine.prepare_block(4)
        engine.fail_output(RuntimeError("output write failed"))

        self.assertEqual(engine.sample_offset, 0)
        self.assertFalse(engine.is_playing)
        engine.confirm_recovery()
        engine.enqueue(PlaybackPlay())
        engine.render_block(2)
        self.assertEqual(engine.sample_offset, 2)

    def test_callback_failure_recovers_from_last_committed_offset_without_snapshot(self):
        created = []

        def make_stream(callback, **kwargs):
            stream = FakeOutputStream(callback, **kwargs)
            created.append(stream)
            return stream

        session = SessionState(mode=SessionMode.UPLOAD)
        store = StateStore(AppState(session=session))
        controller = PlaybackController(store, buffers(20), (), output_stream_factory=make_stream)
        with patch.object(store, "snapshot", side_effect=AssertionError("callback must not snapshot")):
            controller.play()
            created[0].emit(3)
            self.assertEqual(controller.engine.sample_offset, 3)
            failed_output = created[0].emit(2, SimpleNamespace(output_underflow=True))
            np.testing.assert_array_equal(failed_output, np.zeros(2, dtype=np.float32))
            self.assertEqual(controller.engine.sample_offset, 3)
            controller.process_events()
            self.assertEqual(len(created), 2)
            self.assertEqual(controller.engine.sample_offset, 3)
            resumed = created[1].emit(2)

        np.testing.assert_array_equal(resumed, np.full(2, 0.2, dtype=np.float32))
        self.assertEqual(controller.engine.sample_offset, 5)
        self.assertTrue(created[0].closed)
        self.assertIsNone(store.snapshot().session.playback.error_code)
        controller.close()

    def test_state_store_routes_intents_without_snapshot_copy_and_rejects_unrendered_voice(self):
        session = SessionState(mode=SessionMode.UPLOAD)
        session.playback.available_voices = ("3rd",)
        store = StateStore(AppState(session=session))
        with patch.object(store, "snapshot", side_effect=AssertionError("transport must not snapshot")):
            store.dispatch_playback_intent(PlaybackSetVoices(("3rd",)))
            with self.assertRaises(ValueError):
                store.dispatch_playback_intent(PlaybackSetVoices(("octave",)))
            store.dispatch_playback_intent(PlaybackSeek("seek", 1.0))

        state = store.snapshot().session
        self.assertTrue(state.voices.active["3rd"])
        self.assertEqual(state.playback.pending_seek_request_id, "seek")


if __name__ == "__main__":
    unittest.main()