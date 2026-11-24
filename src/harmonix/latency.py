"""Capture-to-output latency records and deterministic aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from statistics import mean


class LatencyProvenance(str, Enum):
    """What the capture-side timestamp can truthfully establish."""

    VERIFIED_CAPTURE_TO_DAC = "verified_capture_to_dac"
    PIPELINE_CALLBACK_TO_DAC = "pipeline_callback_to_dac"
    PHYSICAL_CAPTURE_UNAVAILABLE = "physical_capture_unavailable"


@dataclass(frozen=True)
class LatencyMeasurement:
    trial: int
    capture_timestamp: float | None
    output_timestamp: float
    pipeline_timestamp: float | None
    provenance: LatencyProvenance
    sample_start: int
    sample_count: int
    active_voice_count: int
    mode: str
    input_device: str | None
    output_device: str | None
    sample_rate: int
    block_size: int
    processing_segment_size: int
    queue_overflowed: bool = False
    queue_underflowed: bool = False

    @property
    def latency_ms(self) -> float | None:
        """Physical capture-to-DAC latency, only when its capture time is valid."""
        if self.provenance is not LatencyProvenance.VERIFIED_CAPTURE_TO_DAC or self.capture_timestamp is None:
            return None
        return (self.output_timestamp - self.capture_timestamp) * 1000.0

    @property
    def pipeline_latency_ms(self) -> float | None:
        """Callback-handoff-to-DAC timing; this is not physical capture latency."""
        if self.pipeline_timestamp is None:
            return None
        return (self.output_timestamp - self.pipeline_timestamp) * 1000.0


@dataclass(frozen=True)
class LatencyEvent:
    """A queue or callback event, recorded independently from emitted audio."""

    callback_timestamp: float | None
    output_timestamp: float | None
    input_queue_overflowed: bool = False
    output_queue_underflowed: bool = False
    callback_input_overflow: bool = False
    callback_output_underflow: bool = False
    output_queue_overflowed: bool = False


@dataclass(frozen=True)
class LatencyReport:
    measurements: tuple[LatencyMeasurement, ...]
    events: tuple[LatencyEvent, ...]
    mean_ms: float | None
    maximum_ms: float | None
    pipeline_mean_ms: float | None
    pipeline_maximum_ms: float | None
    physical_latency_unavailable: int
    overrun_events: int
    output_queue_overflow_events: int
    underflow_events: int
    callback_input_overflow_events: int
    callback_output_underflow_events: int


def summarize_latency(
    measurements: list[LatencyMeasurement], events: list[LatencyEvent] | None = None
) -> LatencyReport:
    callback_events = events or []
    values = [value for measurement in measurements if (value := measurement.latency_ms) is not None]
    pipeline_values = [value for measurement in measurements if (value := measurement.pipeline_latency_ms) is not None]
    return LatencyReport(
        measurements=tuple(measurements),
        events=tuple(callback_events),
        mean_ms=mean(values) if values else None,
        maximum_ms=max(values) if values else None,
        pipeline_mean_ms=mean(pipeline_values) if pipeline_values else None,
        pipeline_maximum_ms=max(pipeline_values) if pipeline_values else None,
        physical_latency_unavailable=sum(
            measurement.provenance is LatencyProvenance.PHYSICAL_CAPTURE_UNAVAILABLE
            for measurement in measurements
        ),
        overrun_events=sum(measurement.queue_overflowed for measurement in measurements)
        + sum(event.input_queue_overflowed for event in callback_events)
        + sum(event.output_queue_overflowed for event in callback_events),
        output_queue_overflow_events=sum(event.output_queue_overflowed for event in callback_events),
        underflow_events=sum(measurement.queue_underflowed for measurement in measurements)
        + sum(event.output_queue_underflowed for event in callback_events),
        callback_input_overflow_events=sum(event.callback_input_overflow for event in callback_events),
        callback_output_underflow_events=sum(event.callback_output_underflow for event in callback_events),
    )
