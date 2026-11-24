"""M8 live hardware validation runner.

The runner uses only LiveAudioCoordinator's real sounddevice duplex input. It
does not synthesize, inject, play, or loop back samples. A controlled trial
requires a repeatable pitched acoustic test signal with sharp onsets and quiet
gaps, physically presented to the selected microphone. The operator must
provide the stimulus source and actual routing description; this tool does not
assume a speaker, cable, loopback device, or other hardware route. A written
description does not independently prove the physical route.

Physical capture-to-DAC latency is reported only for verified capture
timestamps. A zero or invalid inputBufferAdcTime leaves physical latency
unavailable; callback-to-DAC pipeline timing is reported separately.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import threading
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import numpy as np
import sounddevice as sd

from harmonix.harmony_generator import HarmonyGenerator
from harmonix.input.adapter import CANONICAL_SAMPLE_RATE, DEFAULT_FRAME_SIZE, InputAdapter
from harmonix.key_scale_resolver import KeyScale, KeyScaleResolver
from harmonix.latency import LatencyMeasurement, LatencyProvenance
from harmonix.live_audio import LiveAudioConfig, LiveAudioCoordinator
from harmonix.pitch_shifter import PitchShifter

MIN_TRIALS_PER_CONDITION = 20
VOICE_SET_BY_COUNT = {
    0: (),
    1: ("3rd",),
    3: ("3rd", "5th", "octave"),
}


class ShiftMode(str, Enum):
    ZERO = "zero"
    DIATONIC = "diatonic"
    OCTAVE = "octave"
    DOWNWARD = "downward"


@dataclass(frozen=True)
class TrialCondition:
    voice_count: int
    voices: tuple[str, ...]
    shift_mode: ShiftMode | None
    shift_ratio: float | None

    @property
    def condition_id(self) -> str:
        mode = "not_applicable" if self.shift_mode is None else self.shift_mode.value
        return f"voices_{self.voice_count}__{mode}"


@dataclass(frozen=True)
class ShiftTarget:
    valid: bool
    target_frequency_hz: float | None
    was_clamped: bool


@dataclass(frozen=True)
class TrialRecord:
    condition_id: str
    trial_index: int
    onset_sample_index: int
    status: str
    reason: str | None
    measurement: dict[str, Any] | None


class ConditionHarmonyGenerator:
    """Apply a test shift to valid M6 targets without changing M6 source."""

    def __init__(
        self,
        generator: HarmonyGenerator,
        shift_mode: ShiftMode | None,
        downward_semitones: float,
    ) -> None:
        self.generator = generator
        self.shift_mode = shift_mode
        self.downward_semitones = downward_semitones

    def generate(
        self,
        fundamental_frequency_hz: float | None,
        effective_key: KeyScale | Any | None,
        voice: str,
        **kwargs: Any,
    ) -> Any:
        target = self.generator.generate(fundamental_frequency_hz, effective_key, voice, **kwargs)
        if self.shift_mode in (None, ShiftMode.DIATONIC) or not target.valid:
            return target
        if fundamental_frequency_hz is None or not math.isfinite(fundamental_frequency_hz):
            return target
        ratio = shift_ratio(self.shift_mode, self.downward_semitones)
        return ShiftTarget(
            valid=True,
            target_frequency_hz=float(fundamental_frequency_hz * ratio),
            was_clamped=False,
        )


class ObservedPitchShifter:
    """Record M7 outcomes while delegating all DSP to the real shifter."""

    def __init__(self, shifter: PitchShifter) -> None:
        self.shifter = shifter
        self.call_count = 0
        self.statuses: list[str] = []

    def process_block(self, *args: Any, **kwargs: Any) -> Any:
        self.call_count += 1
        result = self.shifter.process_block(*args, **kwargs)
        if result is not None:
            self.statuses.append(result.status.value)
        return result


class OnsetDetector:
    """Detect sharp amplitude onsets in worker-normalized microphone samples."""

    def __init__(
        self,
        sample_rate: int,
        threshold: float,
        minimum_gap_ms: float,
        window_ms: float = 10.0,
    ) -> None:
        if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
            raise ValueError("Onset threshold must be finite and in (0, 1].")
        if not math.isfinite(minimum_gap_ms) or minimum_gap_ms <= 0.0:
            raise ValueError("Minimum onset gap must be finite and positive.")
        self.window_samples = max(1, round(sample_rate * window_ms / 1000.0))
        self.minimum_gap_samples = round(sample_rate * minimum_gap_ms / 1000.0)
        self.threshold = threshold
        self.sample_cursor = 0
        self.last_onset_sample = -self.minimum_gap_samples
        self.active = False
        self.pending = np.empty(0, dtype=np.float32)
        self.onsets: list[int] = []

    def feed(self, audio: np.ndarray) -> tuple[int, ...]:
        array = np.asarray(audio, dtype=np.float32)
        if array.ndim != 1:
            raise ValueError("Onset input must be mono.")
        self.pending = np.concatenate((self.pending, array))
        found: list[int] = []
        while self.pending.size >= self.window_samples:
            frame = self.pending[: self.window_samples]
            self.pending = self.pending[self.window_samples :]
            frame_start = self.sample_cursor
            self.sample_cursor += frame.size
            rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float64))))
            if rms >= self.threshold:
                if not self.active and frame_start - self.last_onset_sample >= self.minimum_gap_samples:
                    self.last_onset_sample = frame_start
                    self.onsets.append(frame_start)
                    found.append(frame_start)
                self.active = True
            else:
                self.active = False
        return tuple(found)


class ObservingInputAdapter(InputAdapter):
    """Observe only samples delivered through the real adapter worker path."""

    def __init__(self, device_index: int, onset_detector: OnsetDetector) -> None:
        super().__init__(device_index=device_index)
        self.onset_detector = onset_detector
        self._observation_lock = threading.Lock()
        self._processed_chunk_count = 0

    def process_native_chunk(self, chunk: np.ndarray, sample_rate: int) -> np.ndarray:
        canonical = super().process_native_chunk(chunk, sample_rate)
        self.onset_detector.feed(canonical)
        with self._observation_lock:
            self._processed_chunk_count += 1
        return canonical

    def observation_snapshot(self) -> tuple[int, tuple[int, ...]]:
        with self._observation_lock:
            return self._processed_chunk_count, tuple(self.onset_detector.onsets)


class MonitoredLiveAudioCoordinator(LiveAudioCoordinator):
    """Observe worker progress while retaining the existing worker loop."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._progress_lock = threading.Lock()
        self.worker_processed_chunks = 0
        self.worker_last_progress_monotonic: float | None = None
        self.worker_exception: str | None = None

    def process_pending_once(self) -> bool:
        try:
            processed = super().process_pending_once()
        except Exception:
            with self._progress_lock:
                self.worker_exception = traceback.format_exc()
            raise
        if processed:
            with self._progress_lock:
                self.worker_processed_chunks += 1
                self.worker_last_progress_monotonic = time.monotonic()
        return processed

    def worker_snapshot(self) -> dict[str, Any]:
        with self._progress_lock:
            return {
                "processed_input_chunks": self.worker_processed_chunks,
                "last_progress_monotonic": self.worker_last_progress_monotonic,
                "exception": self.worker_exception,
            }


def shift_ratio(mode: ShiftMode, downward_semitones: float) -> float:
    if mode is ShiftMode.ZERO:
        return 1.0
    if mode is ShiftMode.OCTAVE:
        return 2.0
    if mode is ShiftMode.DOWNWARD:
        if not math.isfinite(downward_semitones) or not 0.0 < downward_semitones <= 24.0:
            raise ValueError("Downward shift must be from 0 (exclusive) through 24 semitones.")
        return 2.0 ** (-downward_semitones / 12.0)
    raise ValueError("Diatonic shift ratio depends on the detected pitch and key.")


def build_condition_matrix(downward_semitones: float) -> tuple[TrialCondition, ...]:
    down_ratio = shift_ratio(ShiftMode.DOWNWARD, downward_semitones)
    conditions = [TrialCondition(0, VOICE_SET_BY_COUNT[0], None, None)]
    ratios = {
        ShiftMode.ZERO: 1.0,
        ShiftMode.DIATONIC: None,
        ShiftMode.OCTAVE: 2.0,
        ShiftMode.DOWNWARD: down_ratio,
    }
    for voice_count in (1, 3):
        for mode in ShiftMode:
            conditions.append(TrialCondition(voice_count, VOICE_SET_BY_COUNT[voice_count], mode, ratios[mode]))
    return tuple(conditions)


def validate_trial_count(trials_per_condition: int) -> None:
    if trials_per_condition < MIN_TRIALS_PER_CONDITION:
        raise ValueError(f"At least {MIN_TRIALS_PER_CONDITION} trials per condition are required.")


def classify_measurement(measurement: LatencyMeasurement) -> dict[str, Any]:
    capture = finite_timestamp(measurement.capture_timestamp)
    output = finite_timestamp(measurement.output_timestamp)
    pipeline = finite_timestamp(measurement.pipeline_timestamp)
    verified = (
        measurement.provenance is LatencyProvenance.VERIFIED_CAPTURE_TO_DAC
        and capture is not None
        and capture > 0.0
        and output is not None
    )
    physical_latency_ms = (output - capture) * 1000.0 if verified else None
    pipeline_latency_ms = (output - pipeline) * 1000.0 if output is not None and pipeline is not None else None
    return {
        "provenance": measurement.provenance.value,
        "capture_timestamp": capture,
        "callback_timestamp": pipeline,
        "output_timestamp": output,
        "physical_capture_to_dac_latency_ms": physical_latency_ms,
        "physical_capture_to_dac_status": "measured" if verified else "unavailable",
        "pipeline_latency_ms": pipeline_latency_ms,
        "sample_start": measurement.sample_start,
        "sample_count": measurement.sample_count,
        "active_voice_count": measurement.active_voice_count,
    }


def finite_timestamp(value: float | None) -> float | None:
    if value is None:
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None


def pair_onsets_with_measurements(
    condition_id: str,
    onsets: Sequence[int],
    measurements: Sequence[LatencyMeasurement],
    trials_per_condition: int,
) -> tuple[TrialRecord, ...]:
    validate_trial_count(trials_per_condition)
    records: list[TrialRecord] = []
    unused = set(range(len(measurements)))
    for trial_index, onset in enumerate(onsets[:trials_per_condition], start=1):
        match_index = next(
            (
                index
                for index in sorted(unused)
                if measurements[index].sample_start <= onset
                < measurements[index].sample_start + measurements[index].sample_count
            ),
            None,
        )
        if match_index is None:
            records.append(
                TrialRecord(condition_id, trial_index, int(onset), "measurement_missing", "No output record covered the captured onset.", None)
            )
            continue
        unused.remove(match_index)
        records.append(
            TrialRecord(
                condition_id,
                trial_index,
                int(onset),
                "measured",
                None,
                classify_measurement(measurements[match_index]),
            )
        )
    return tuple(records)


def summarize_condition(
    condition: TrialCondition,
    trial_records: Sequence[TrialRecord],
    coordinator_report: Any | None,
    worker_health: dict[str, Any],
    shifter_outcomes: dict[str, Any],
    error: str | None,
) -> dict[str, Any]:
    records = [asdict(record) for record in trial_records]
    measured = [record.measurement for record in trial_records if record.measurement is not None]
    physical = [item["physical_capture_to_dac_latency_ms"] for item in measured if item["physical_capture_to_dac_latency_ms"] is not None]
    pipeline = [item["pipeline_latency_ms"] for item in measured if item["pipeline_latency_ms"] is not None]
    events = () if coordinator_report is None else coordinator_report.events
    return {
        "condition": condition_dict(condition),
        "status": "failed" if error else ("complete" if len(trial_records) and all(r.status == "measured" for r in trial_records) else "incomplete"),
        "error": error,
        "trial_count": len(trial_records),
        "measured_trial_count": sum(record.status == "measured" for record in trial_records),
        "physical_capture_to_dac_trials": len(physical),
        "physical_capture_to_dac_latency_mean_ms": mean(physical) if physical else None,
        "physical_capture_to_dac_latency_max_ms": max(physical) if physical else None,
        "physical_capture_to_dac_unavailable_trials": sum(
            item["physical_capture_to_dac_status"] == "unavailable" for item in measured
        ),
        "pipeline_latency_trials": len(pipeline),
        "pipeline_latency_mean_ms": mean(pipeline) if pipeline else None,
        "pipeline_latency_max_ms": max(pipeline) if pipeline else None,
        "input_queue_overflow_events": sum(event.input_queue_overflowed for event in events),
        "output_queue_overflow_events": sum(event.output_queue_overflowed for event in events),
        "queue_underflow_events": sum(event.output_queue_underflowed for event in events),
        "callback_input_overflow_events": sum(event.callback_input_overflow for event in events),
        "callback_output_underflow_events": sum(event.callback_output_underflow for event in events),
        "worker": worker_health,
        "pitch_shifter_outcomes": shifter_outcomes,
        "trials": records,
    }


def condition_dict(condition: TrialCondition) -> dict[str, Any]:
    return {
        "condition_id": condition.condition_id,
        "active_voice_count": condition.voice_count,
        "active_voices": list(condition.voices),
        "shift_mode": None if condition.shift_mode is None else condition.shift_mode.value,
        "configured_shift_ratio": condition.shift_ratio,
    }


def device_record(device_index: int) -> dict[str, Any]:
    devices = sd.query_devices()
    if not 0 <= device_index < len(devices):
        raise ValueError(f"Device index {device_index} is not present in sounddevice enumeration.")
    device = devices[device_index]
    host_apis = sd.query_hostapis()
    host_api_index = int(device["hostapi"])
    return {
        "index": device_index,
        "name": str(device["name"]),
        "host_api": str(host_apis[host_api_index]["name"]),
        "max_input_channels": int(device["max_input_channels"]),
        "max_output_channels": int(device["max_output_channels"]),
        "default_sample_rate_hz": float(device["default_samplerate"]),
    }


def collect_hardware_configuration(args: argparse.Namespace) -> dict[str, Any]:
    input_device = device_record(args.input_device)
    output_index = args.output_device
    if output_index is None:
        default_devices = sd.default.device
        if not isinstance(default_devices, tuple) or default_devices[1] is None:
            raise ValueError("No default output device; supply --output-device.")
        output_index = int(default_devices[1])
    output_device = device_record(output_index)
    if input_device["max_input_channels"] < 1:
        raise ValueError("Selected input device does not expose an input channel.")
    if output_device["max_output_channels"] < 1:
        raise ValueError("Selected output device does not expose an output channel.")
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "input_device": input_device,
        "output_device": output_device,
        "sample_rate_hz": CANONICAL_SAMPLE_RATE,
        "block_size_samples": DEFAULT_FRAME_SIZE,
        "stimulus_pitch_hz": args.stimulus_pitch_hz,
        "stimulus_description": args.stimulus_description,
        "physical_routing_description": args.routing_description,
        "onset_rms_threshold": args.onset_threshold,
        "minimum_onset_gap_ms": args.minimum_onset_gap_ms,
        "synthetic_input_used": False,
    }


def run_condition(
    condition: TrialCondition,
    args: argparse.Namespace,
) -> dict[str, Any]:
    detector = OnsetDetector(CANONICAL_SAMPLE_RATE, args.onset_threshold, args.minimum_onset_gap_ms)
    adapter = ObservingInputAdapter(args.input_device, detector)
    resolver = KeyScaleResolver()
    resolver.set_manual_key(args.key_root, args.scale)
    resolver.apply_processing_boundary()
    base_generator = HarmonyGenerator()
    generator = ConditionHarmonyGenerator(base_generator, condition.shift_mode, args.downward_semitones)
    shifters = {voice: ObservedPitchShifter(PitchShifter()) for voice in VOICE_SET_BY_COUNT[3]}
    coordinator = MonitoredLiveAudioCoordinator(
        adapter,
        key_resolver=resolver,
        harmony_generator=generator,
        pitch_shifters=shifters,
        config=LiveAudioConfig(),
    )
    coordinator.set_active_voices(set(condition.voices))
    error: str | None = None
    worker_thread: threading.Thread | None = None
    worker_alive_observed = False
    deadline = time.monotonic() + args.condition_timeout_seconds
    try:
        if args.output_device is not None:
            sd.default.device = (args.input_device, args.output_device)
        coordinator.start(args.input_device)
        worker_thread = coordinator._worker
        print(f"READY {condition.condition_id}: present the described physical stimulus; no samples are injected.", flush=True)
        while time.monotonic() < deadline:
            if worker_thread is not None and worker_thread.is_alive():
                worker_alive_observed = True
            processed_count, onsets = adapter.observation_snapshot()
            del processed_count
            current_report = coordinator.latency_report()
            trial_records = pair_onsets_with_measurements(
                condition.condition_id,
                onsets,
                current_report.measurements,
                args.trials_per_condition,
            )
            if len(trial_records) >= args.trials_per_condition and all(
                record.status == "measured" for record in trial_records
            ):
                break
            if worker_thread is not None and not worker_thread.is_alive() and coordinator.worker_snapshot()["exception"]:
                break
            time.sleep(0.05)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            coordinator.stop()
        except Exception as exc:
            cleanup_error = f"cleanup {type(exc).__name__}: {exc}"
            error = f"{error}; {cleanup_error}" if error else cleanup_error

    report = coordinator.latency_report()
    _, onsets = adapter.observation_snapshot()
    trials = pair_onsets_with_measurements(
        condition.condition_id,
        onsets,
        report.measurements,
        args.trials_per_condition,
    )
    worker = coordinator.worker_snapshot()
    worker["alive_during_run"] = worker_alive_observed
    worker["alive_after_stop"] = worker_thread.is_alive() if worker_thread is not None else False
    worker["input_queue_depth_after_stop"] = coordinator._input_queue.qsize()
    shift_statuses = Counter(status for shifter in shifters.values() for status in shifter.statuses)
    shift_outcomes = {
        "calls": sum(shifter.call_count for shifter in shifters.values()),
        "completed_results": sum(len(shifter.statuses) for shifter in shifters.values()),
        "status_counts": dict(shift_statuses),
    }
    return summarize_condition(condition, trials, report, worker, shift_outcomes, error)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run M8 over the real sounddevice duplex microphone stream. A repeatable pitched signal with sharp onsets "
            "must be physically presented to the selected microphone; this tool does not synthesize or route it."
        )
    )
    parser.add_argument("--input-device", type=int, required=True)
    parser.add_argument("--output-device", type=int)
    parser.add_argument("--trials-per-condition", type=int, default=MIN_TRIALS_PER_CONDITION)
    parser.add_argument("--downward-semitones", type=float, required=True)
    parser.add_argument("--key-root", default="C")
    parser.add_argument("--scale", choices=("major", "minor"), default="major")
    parser.add_argument("--stimulus-pitch-hz", type=float, required=True)
    parser.add_argument("--stimulus-description", required=True)
    parser.add_argument("--routing-description", required=True)
    parser.add_argument("--onset-threshold", type=float, default=0.02)
    parser.add_argument("--minimum-onset-gap-ms", type=float, default=300.0)
    parser.add_argument("--condition-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    validate_trial_count(args.trials_per_condition)
    shift_ratio(ShiftMode.DOWNWARD, args.downward_semitones)
    if not math.isfinite(args.stimulus_pitch_hz) or args.stimulus_pitch_hz <= 0.0:
        raise ValueError("Stimulus pitch must be finite and positive.")
    if not math.isfinite(args.condition_timeout_seconds) or args.condition_timeout_seconds <= 0.0:
        raise ValueError("Condition timeout must be finite and positive.")
    if not math.isfinite(args.onset_threshold) or not 0.0 < args.onset_threshold <= 1.0:
        raise ValueError("Onset threshold must be finite and in (0, 1].")
    if not args.stimulus_description.strip() or not args.routing_description.strip():
        raise ValueError("Stimulus and physical routing descriptions are required.")


def human_summary(report: dict[str, Any]) -> str:
    lines = [
        "M8 Hardware Validation Summary",
        f"Run status: {report['status']}",
        f"Hardware: input {report['hardware']['input_device']['name']} ({report['hardware']['input_device']['index']})",
        f"Output: {report['hardware']['output_device']['name']} ({report['hardware']['output_device']['index']})",
        "Physical capture-to-DAC latency is unavailable unless each record has verified capture provenance.",
        "Pipeline latency is callback-to-DAC timing and is not physical capture-to-DAC latency.",
    ]
    for condition in report["conditions"]:
        lines.append(
            f"{condition['condition']['condition_id']}: status={condition['status']}, "
            f"trials={condition['measured_trial_count']}/{report['trials_per_condition']}, "
            f"physical={condition['physical_capture_to_dac_trials']}, "
            f"pipeline={condition['pipeline_latency_trials']}, "
            f"input_overflow={condition['input_queue_overflow_events']}, "
            f"output_overflow={condition['output_queue_overflow_events']}, "
            f"underflow={condition['queue_underflow_events']}"
        )
    return "\n".join(lines)


def run_hardware_validation(args: argparse.Namespace) -> dict[str, Any]:
    validate_arguments(args)
    original_default = sd.default.device
    hardware = collect_hardware_configuration(args)
    if args.output_device is None:
        args.output_device = hardware["output_device"]["index"]
    sd.default.device = (args.input_device, args.output_device)
    conditions = [run_condition(condition, args) for condition in build_condition_matrix(args.downward_semitones)]
    complete = all(condition["status"] == "complete" for condition in conditions)
    report = {
        "schema_version": 1,
        "run_mode": "actual_sounddevice_duplex_live_stream",
        "status": "complete" if complete else "incomplete",
        "physical_latency_policy": "Only verified capture timestamps qualify; invalid/zero ADC time means unavailable.",
        "pipeline_latency_policy": "Reported separately; never labeled physical capture-to-DAC latency.",
        "trials_per_condition": args.trials_per_condition,
        "downward_semitones": args.downward_semitones,
        "key": {"root": args.key_root, "scale": args.scale},
        "hardware": hardware,
        "conditions": conditions,
    }
    try:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        args.summary_out.write_text(human_summary(report) + "\n", encoding="utf-8")
    finally:
        sd.default.device = original_default
    print(human_summary(report))
    print(f"JSON report: {args.json_out}")
    print(f"Text report: {args.summary_out}")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_hardware_validation(args)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())