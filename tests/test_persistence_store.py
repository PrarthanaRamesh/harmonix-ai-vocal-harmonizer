import json
import multiprocessing
import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from harmonix.config.models import MAX_RECENT_SESSIONS
from harmonix.domain.models import GlobalSettings
from harmonix.domain.errors import PersistenceLockError, PersistenceReadError, PersistenceWriteError
from harmonix.persistence.store import JSONPersistenceStore, RecentSessionMetadata


def _try_lifetime_lock(base_dir: str, output) -> None:
    store = JSONPersistenceStore(base_dir)
    try:
        store.acquire_lifetime_lock()
    except PersistenceLockError:
        output.put("blocked")
    else:
        output.put("acquired")
        store.release_lifetime_lock()


class PersistenceStoreTests(unittest.TestCase):
    def test_missing_settings_returns_defaults_without_warning_or_write(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            store = JSONPersistenceStore(tempdir)

            loaded = store.load_settings()

            self.assertEqual(loaded.latency_buffer_frame_size, 2048)
            self.assertIsNone(store.settings_warning)
            self.assertFalse(store.settings_path.exists())

    def test_unreadable_settings_load_is_read_only_and_blocks_save(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            store = JSONPersistenceStore(tempdir)
            original = b'{"reduced_motion": false}'
            store.settings_path.write_bytes(original)

            with patch.object(store, "_load_json", side_effect=PermissionError("access denied")):
                loaded = store.load_settings()

            self.assertEqual(loaded, GlobalSettings())
            self.assertIsNotNone(store.settings_warning)
            self.assertEqual(store.settings_path.read_bytes(), original)
            real_open = Path.open

            def fail_settings_open(path, mode="r", *args, **kwargs):
                if path == store.settings_path and mode == "rb":
                    raise PermissionError("access denied")
                return real_open(path, mode, *args, **kwargs)

            with patch.object(Path, "open", autospec=True, side_effect=fail_settings_open):
                with self.assertRaises(PersistenceReadError):
                    store.save_settings(GlobalSettings(reduced_motion=True))
            self.assertEqual(store.settings_path.read_bytes(), original)

    def test_file_that_becomes_unreadable_before_save_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            store = JSONPersistenceStore(tempdir)
            store.save_settings(GlobalSettings(reduced_motion=False))
            original = store.settings_path.read_bytes()
            real_open = Path.open

            def fail_settings_open(path, mode="r", *args, **kwargs):
                if path == store.settings_path and mode == "rb":
                    raise PermissionError("access denied")
                return real_open(path, mode, *args, **kwargs)

            with patch.object(Path, "open", autospec=True, side_effect=fail_settings_open):
                with self.assertRaises(PersistenceReadError):
                    store.save_settings(GlobalSettings(reduced_motion=True))

            self.assertIsNotNone(store.settings_warning)
            self.assertEqual(store.settings_path.read_bytes(), original)
    def test_store_persists_settings_in_local_appdata_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            store = JSONPersistenceStore(tempdir)
            settings = GlobalSettings(
                audio_input_device="Realtek Mic",
                audio_output_device="Speakers",
                reduced_motion=True,
                export_default_format="wav",
                export_default_quality="high",
            )
            store.save_settings(settings)
            loaded = store.load_settings()
            self.assertEqual(loaded.audio_input_device, "Realtek Mic")
            self.assertEqual(loaded.audio_output_device, "Speakers")
            self.assertTrue(loaded.reduced_motion)
            self.assertEqual(loaded.export_default_format, "wav")
            self.assertEqual(loaded.export_default_quality, "high")
            self.assertTrue(Path(tempdir, "HarmonixV2", "settings.json").exists())

    def test_store_keeps_only_valid_recent_sessions_and_enforces_max_five(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            records = []
            for index in range(7):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
                    path = handle.name
                records.append(path)
            store = JSONPersistenceStore(tempdir)
            payload = [
                {"source_file": records[0], "last_used_at": "2024-01-01T00:00:00Z"},
                {"source_file": records[1], "last_used_at": "2024-01-02T00:00:00Z"},
                {"source_file": records[2], "last_used_at": "2024-01-03T00:00:00Z"},
                {"source_file": records[3], "last_used_at": "2024-01-04T00:00:00Z"},
                {"source_file": records[4], "last_used_at": "2024-01-05T00:00:00Z"},
                {"source_file": records[5], "last_used_at": "2024-01-06T00:00:00Z"},
                {"source_file": records[6], "last_used_at": "2024-01-07T00:00:00Z"},
            ]
            saved = store.save_recent_sessions(payload)
            self.assertLessEqual(len(saved), MAX_RECENT_SESSIONS)
            loaded = store.load_recent_sessions()
            self.assertEqual(len(loaded), MAX_RECENT_SESSIONS)
            self.assertEqual(
                [entry.last_used_at for entry in loaded],
                [f"2024-01-{day:02d}T00:00:00Z" for day in range(7, 2, -1)],
            )

    def test_store_ignores_missing_or_corrupt_recent_session_entries(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            store = JSONPersistenceStore(tempdir)
            path = Path(tempdir) / "HarmonixV2" / "recent_sessions.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps([{"source_file": "missing.wav"}, {"source_file": "not_a_file"}, {"voices": ["3rd"]}]), encoding="utf-8")
            self.assertEqual(store.load_recent_sessions(), [])

    def test_recent_metadata_round_trips_live_without_source_and_preserves_voice_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            store = JSONPersistenceStore(tempdir)
            live = RecentSessionMetadata(
                key_root="E",
                key_scale_type="minor",
                voices=("3rd", "octave", "legacy_voice"),
                style_preset="choir",
                last_used_at="2025-02-03T04:05:06+00:00",
                id="live-stable-id",
                display_name="Live session — 2025-02-03",
                mode="live",
                voice_toggles=(("3rd", True), ("5th", False), ("octave", True)),
            )

            saved = store.save_recent_sessions([live])
            loaded = JSONPersistenceStore(tempdir).load_recent_sessions()

            self.assertEqual(saved[0].id, live.id)
            self.assertEqual(len(loaded), 1)
            self.assertIsNone(loaded[0].source_file)
            self.assertEqual(loaded[0].voices, live.voices)
            self.assertEqual(loaded[0].voice_toggles, live.voice_toggles)
            self.assertEqual(loaded[0].to_summary().mode.value, "live")

    def test_recent_session_order_uses_actual_instants_and_preserves_timestamp_text(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            records = [
                RecentSessionMetadata(id="offset-earlier", mode="live", last_used_at="2024-01-01T00:30:00+02:00"),
                RecentSessionMetadata(id="utc-later", mode="live", last_used_at="2023-12-31T23:00:00Z"),
                RecentSessionMetadata(id="utc-newest", mode="live", last_used_at="2024-01-01T00:00:00+00:00"),
            ]

            saved = JSONPersistenceStore(tempdir).save_recent_sessions(records)
            loaded = JSONPersistenceStore(tempdir).load_recent_sessions()

            expected_ids = ["utc-newest", "utc-later", "offset-earlier"]
            self.assertEqual([item.id for item in saved], expected_ids)
            self.assertEqual([item.id for item in loaded], expected_ids)
            self.assertEqual(
                {item.id: item.last_used_at for item in loaded},
                {item.id: item.last_used_at for item in records},
            )

    def test_naive_and_malformed_recent_timestamps_sort_after_known_instants(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            records = [
                RecentSessionMetadata(id="malformed", mode="live", last_used_at="not-a-timestamp"),
                RecentSessionMetadata(id="naive", mode="live", last_used_at="2025-01-01T12:00:00"),
                RecentSessionMetadata(id="aware", mode="live", last_used_at="2024-12-31T23:00:00Z"),
            ]

            JSONPersistenceStore(tempdir).save_recent_sessions(records)
            loaded = JSONPersistenceStore(tempdir).load_recent_sessions()

            self.assertEqual([item.id for item in loaded], ["aware", "malformed", "naive"])
            self.assertEqual(
                [item.last_used_at for item in loaded],
                ["2024-12-31T23:00:00Z", "not-a-timestamp", "2025-01-01T12:00:00"],
            )

    def test_legacy_recent_record_keeps_supported_fields_and_stable_id(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            source_path = str(Path(tempdir) / "legacy.wav")
            Path(source_path).write_bytes(b"audio")
            store = JSONPersistenceStore(tempdir)
            store.recent_sessions_path.write_text(
                json.dumps([{
                    "source_file": source_path,
                    "key_root": "F#",
                    "key_scale_type": "minor",
                    "voices": ["3rd", "octave", "legacy_voice"],
                    "style_preset": "legacy-style-name",
                    "last_used_at": "2024-03-02T00:00:00Z",
                }]),
                encoding="utf-8",
            )

            first = store.load_recent_sessions()[0]
            second = JSONPersistenceStore(tempdir).load_recent_sessions()[0]

            self.assertEqual(first.id, second.id)
            self.assertEqual(first.mode, "upload")
            self.assertEqual(first.key_root, "F#")
            self.assertEqual(first.key_scale_type, "minor")
            self.assertEqual(first.voices, ("3rd", "octave", "legacy_voice"))
            self.assertEqual(first.to_summary().style_preset, "legacy-style-name")
            self.assertEqual(dict(first.to_summary().voice_toggles)["octave"], True)

    def test_unreadable_recent_metadata_load_is_read_only_and_save_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            store = JSONPersistenceStore(tempdir)
            original = b'{"prior":"metadata"}'
            store.recent_sessions_path.write_bytes(original)
            real_open = Path.open
            real_read_bytes = Path.read_bytes

            def deny_open(path, mode="r", *args, **kwargs):
                if path == store.recent_sessions_path:
                    raise PermissionError("access denied")
                return real_open(path, mode, *args, **kwargs)

            def deny_read(path):
                if path == store.recent_sessions_path:
                    raise PermissionError("access denied")
                return real_read_bytes(path)

            with patch.object(Path, "open", autospec=True, side_effect=deny_open), patch.object(
                Path, "read_bytes", autospec=True, side_effect=deny_read
            ):
                self.assertEqual(store.load_recent_sessions(), [])
                with self.assertRaises(PersistenceReadError):
                    store.save_recent_sessions([])

            self.assertEqual(store.recent_sessions_path.read_bytes(), original)

    def test_corrupt_settings_are_backed_up_before_explicit_save(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            store = JSONPersistenceStore(tempdir)
            path = store.settings_path
            original = b'{broken metadata\x00'
            path.write_bytes(original)

            self.assertEqual(store.load_settings().audio_input_device, None)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(path.parent.glob(path.name + ".corrupt-*.bak")), [])

            store.save_settings(GlobalSettings(audio_input_device="new device"))

            backups = list(path.parent.glob(path.name + ".corrupt-*.bak"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), original)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["audio_input_device"], "new device")

    def test_failed_corrupt_backup_preserves_active_bytes_and_reports_error(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            store = JSONPersistenceStore(tempdir)
            original = b"not json"
            store.settings_path.write_bytes(original)
            store.load_settings()

            with patch.object(store, "_preserve_corrupt_file", side_effect=PersistenceWriteError("backup failed")):
                with self.assertRaises(PersistenceWriteError):
                    store.save_settings(GlobalSettings())

            self.assertEqual(store.settings_path.read_bytes(), original)

    def test_corrupt_backup_uses_a_unique_name_without_overwriting_existing_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            store = JSONPersistenceStore(tempdir)
            original = b"corrupt original"
            store.settings_path.write_bytes(original)
            store.load_settings()
            existing = store.settings_path.with_name(store.settings_path.name + ".corrupt-fixed.bak")
            existing.write_bytes(b"prior backup")

            store.save_settings(GlobalSettings())

            self.assertEqual(existing.read_bytes(), b"prior backup")
            backups = list(store.settings_path.parent.glob(store.settings_path.name + ".corrupt-*.bak"))
            self.assertEqual(len(backups), 2)
            self.assertEqual(sum(backup.read_bytes() == original for backup in backups), 1)

    def test_atomic_replace_failure_preserves_corrupt_active_and_completed_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            store = JSONPersistenceStore(tempdir)
            original = b"{ corrupt"
            store.settings_path.write_bytes(original)
            store.load_settings()

            with patch("harmonix.persistence.store.os.replace", side_effect=OSError("replace failed")):
                with self.assertRaises(PersistenceWriteError):
                    store.save_settings(GlobalSettings())

            backups = list(store.settings_path.parent.glob(store.settings_path.name + ".corrupt-*.bak"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), original)
            self.assertEqual(store.settings_path.read_bytes(), original)
            self.assertTrue(store._corrupt_paths.__contains__(store.settings_path))

    def test_atomic_write_failure_preserves_existing_valid_file(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            store = JSONPersistenceStore(tempdir)
            store.save_settings(GlobalSettings(audio_input_device="old"))
            original = store.settings_path.read_bytes()

            with patch("harmonix.persistence.store.os.replace", side_effect=OSError("replace failed")):
                with self.assertRaises(PersistenceWriteError):
                    store.save_settings(GlobalSettings(audio_input_device="new"))

            self.assertEqual(store.settings_path.read_bytes(), original)

    def test_serialization_recursion_failure_is_typed_and_cleans_temp(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            store = JSONPersistenceStore(tempdir)
            store.save_settings(GlobalSettings(audio_input_device="previous"))
            original = store.settings_path.read_bytes()
            pattern = "." + store.settings_path.name + ".*.tmp"
            before_temps = set(store.settings_path.parent.glob(pattern))

            with patch("harmonix.persistence.store.json.dump", side_effect=RecursionError("too deep")):
                with self.assertRaises(PersistenceWriteError) as raised:
                    store.save_settings(GlobalSettings(audio_input_device="replacement"))

            self.assertIsInstance(raised.exception.__cause__, RecursionError)
            self.assertEqual(store.settings_path.read_bytes(), original)
            self.assertEqual(set(store.settings_path.parent.glob(pattern)), before_temps)

    def test_temporary_write_failure_is_typed_and_cleans_partial_temp(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            store = JSONPersistenceStore(tempdir)
            store.save_settings(GlobalSettings(audio_input_device="previous"))
            original = store.settings_path.read_bytes()
            pattern = "." + store.settings_path.name + ".*.tmp"
            before_temps = set(store.settings_path.parent.glob(pattern))

            def partial_write_then_fail(payload, handle, **kwargs):
                handle.write("partial")
                raise OSError("temporary write failed")

            with patch("harmonix.persistence.store.json.dump", side_effect=partial_write_then_fail):
                with self.assertRaises(PersistenceWriteError) as raised:
                    store.save_settings(GlobalSettings(audio_input_device="replacement"))

            self.assertIsInstance(raised.exception.__cause__, OSError)
            self.assertEqual(store.settings_path.read_bytes(), original)
            self.assertEqual(set(store.settings_path.parent.glob(pattern)), before_temps)

    def test_flush_failure_is_typed_and_cleans_temp(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            store = JSONPersistenceStore(tempdir)
            store.save_settings(GlobalSettings(audio_input_device="previous"))
            original = store.settings_path.read_bytes()
            pattern = "." + store.settings_path.name + ".*.tmp"
            before_temps = set(store.settings_path.parent.glob(pattern))
            real_factory = tempfile.NamedTemporaryFile

            class FlushFailureFile:
                def __init__(self, *args, **kwargs):
                    self._handle = real_factory(*args, **kwargs)
                    self.name = self._handle.name

                def __enter__(self):
                    self._handle.__enter__()
                    return self

                def __exit__(self, exc_type, exc, traceback):
                    return self._handle.__exit__(exc_type, exc, traceback)

                def write(self, value):
                    return self._handle.write(value)

                def flush(self):
                    raise OSError("flush failed")

                def fileno(self):
                    return self._handle.fileno()

            with patch("harmonix.persistence.store.tempfile.NamedTemporaryFile", FlushFailureFile):
                with self.assertRaises(PersistenceWriteError) as raised:
                    store.save_settings(GlobalSettings(audio_input_device="replacement"))

            self.assertIsInstance(raised.exception.__cause__, OSError)
            self.assertEqual(store.settings_path.read_bytes(), original)
            self.assertEqual(set(store.settings_path.parent.glob(pattern)), before_temps)

    def test_fsync_failure_is_typed_and_cleans_temp(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            store = JSONPersistenceStore(tempdir)
            store.save_settings(GlobalSettings(audio_input_device="previous"))
            original = store.settings_path.read_bytes()
            pattern = "." + store.settings_path.name + ".*.tmp"
            before_temps = set(store.settings_path.parent.glob(pattern))

            with patch("harmonix.persistence.store.os.fsync", side_effect=OSError("fsync failed")):
                with self.assertRaises(PersistenceWriteError) as raised:
                    store.save_settings(GlobalSettings(audio_input_device="replacement"))

            self.assertIsInstance(raised.exception.__cause__, OSError)
            self.assertEqual(store.settings_path.read_bytes(), original)
            self.assertEqual(set(store.settings_path.parent.glob(pattern)), before_temps)

    def test_temp_creation_failure_is_typed_and_preserves_target(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            store = JSONPersistenceStore(tempdir)
            store.save_settings(GlobalSettings(audio_input_device="previous"))
            original = store.settings_path.read_bytes()
            pattern = "." + store.settings_path.name + ".*.tmp"
            before_temps = set(store.settings_path.parent.glob(pattern))

            with patch("harmonix.persistence.store.tempfile.NamedTemporaryFile", side_effect=OSError("create failed")):
                with self.assertRaises(PersistenceWriteError) as raised:
                    store.save_settings(GlobalSettings(audio_input_device="replacement"))

            self.assertIsInstance(raised.exception.__cause__, OSError)
            self.assertEqual(store.settings_path.read_bytes(), original)
            self.assertEqual(set(store.settings_path.parent.glob(pattern)), before_temps)

    def test_malformed_settings_fields_are_backed_up_only_on_explicit_save(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            store = JSONPersistenceStore(tempdir)
            original = json.dumps(
                {
                    "audio_input_device": "partially valid",
                    "latency_buffer_frame_size": 99,
                    "reduced_motion": "false",
                },
                sort_keys=True,
            ).encode("utf-8")
            store.settings_path.write_bytes(original)

            loaded = store.load_settings()

            self.assertIsNone(loaded.audio_input_device)
            self.assertFalse(loaded.reduced_motion)
            self.assertEqual(store.settings_path.read_bytes(), original)
            self.assertEqual(list(store.settings_path.parent.glob(store.settings_path.name + ".corrupt-*.bak")), [])

            store.save_settings(GlobalSettings(audio_input_device="explicit save"))

            backups = list(store.settings_path.parent.glob(store.settings_path.name + ".corrupt-*.bak"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), original)
            self.assertEqual(json.loads(store.settings_path.read_text(encoding="utf-8"))["audio_input_device"], "explicit save")

    def test_malformed_recent_entry_is_backed_up_only_on_explicit_save(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            store = JSONPersistenceStore(tempdir)
            original = json.dumps(
                [{"source_file": "missing.wav", "voices": ["3rd", 5]}],
                sort_keys=True,
            ).encode("utf-8")
            store.recent_sessions_path.write_bytes(original)

            self.assertEqual(store.load_recent_sessions(), [])
            self.assertEqual(store.recent_sessions_path.read_bytes(), original)
            self.assertEqual(
                list(store.recent_sessions_path.parent.glob(store.recent_sessions_path.name + ".corrupt-*.bak")), []
            )

            store.save_recent_sessions([])

            backups = list(store.recent_sessions_path.parent.glob(store.recent_sessions_path.name + ".corrupt-*.bak"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_bytes(), original)
            self.assertEqual(json.loads(store.recent_sessions_path.read_text(encoding="utf-8")), [])

    def test_stale_lock_file_does_not_block_startup(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            store = JSONPersistenceStore(tempdir)
            store.lock_path.write_bytes(b"stale lock contents")
            store.acquire_lifetime_lock()
            store.release_lifetime_lock()

    def test_lifetime_lock_blocks_and_then_allows_another_process(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            owner = JSONPersistenceStore(tempdir)
            owner.acquire_lifetime_lock()
            try:
                owner.save_settings(GlobalSettings(audio_input_device="write while lock held"))
                context = multiprocessing.get_context("spawn")
                output = context.Queue()
                process = context.Process(target=_try_lifetime_lock, args=(tempdir, output))
                process.start()
                self.assertEqual(output.get(timeout=10), "blocked")
                process.join(timeout=10)
                self.assertEqual(process.exitcode, 0)
            finally:
                owner.release_lifetime_lock()

            output = context.Queue()
            process = context.Process(target=_try_lifetime_lock, args=(tempdir, output))
            process.start()
            self.assertEqual(output.get(timeout=10), "acquired")
            process.join(timeout=10)
            self.assertEqual(process.exitcode, 0)


if __name__ == "__main__":
    unittest.main()
