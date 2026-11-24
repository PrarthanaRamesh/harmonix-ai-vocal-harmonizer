
























import unittest

from harmonix.latency import LatencyEvent, LatencyMeasurement, LatencyProvenance, summarize_latency


class LatencyTests(unittest.TestCase):
    def measurement(self, trial: int, latency: float, *, overflow: bool = False, underflow: bool = False) -> LatencyMeasurement:
        return LatencyMeasurement(
            trial=trial,
            capture_timestamp=10.0,
            output_timestamp=10.0 + latency,
            pipeline_timestamp=None,
            provenance=LatencyProvenance.VERIFIED_CAPTURE_TO_DAC,
            sample_start=trial * 2048,
            sample_count=2048,
            active_voice_count=1,
            mode="live",
            input_device="input",
            output_device="output",
            sample_rate=44100,
            block_size=2048,
            processing_segment_size=8192,
            queue_overflowed=overflow,
            queue_underflowed=underflow,
        )

    def test_latency_is_capture_to_output_delta(self) -> None:
        measurement = self.measurement(1, 0.025)
        self.assertAlmostEqual(measurement.latency_ms, 25.0)

    def test_summary_reports_mean_max_and_events(self) -> None:
        report = summarize_latency([
            self.measurement(1, 0.010),
            self.measurement(2, 0.020, overflow=True),
            self.measurement(3, 0.030, underflow=True),
        ])

        self.assertAlmostEqual(report.mean_ms, 20.0)
        self.assertAlmostEqual(report.maximum_ms, 30.0)
        self.assertEqual(report.overrun_events, 1)
        self.assertEqual(report.underflow_events, 1)
        self.assertEqual(len(report.measurements), 3)

    def test_empty_summary_is_explicit(self) -> None:
        report = summarize_latency([])
        self.assertIsNone(report.mean_ms)
        self.assertIsNone(report.maximum_ms)
        self.assertEqual(report.measurements, ())

    def test_summary_keeps_input_and_output_queue_overflows_distinct(self) -> None:
        input_overflow = LatencyEvent(callback_timestamp=1.0, output_timestamp=2.0, input_queue_overflowed=True)
        output_overflow = LatencyEvent(callback_timestamp=None, output_timestamp=None, output_queue_overflowed=True)

        report = summarize_latency([], [input_overflow, output_overflow])

        self.assertEqual(report.overrun_events, 2)
        self.assertEqual(report.output_queue_overflow_events, 1)
        self.assertTrue(report.events[0].input_queue_overflowed)
        self.assertFalse(report.events[0].output_queue_overflowed)
        self.assertTrue(report.events[1].output_queue_overflowed)
        self.assertFalse(report.events[1].input_queue_overflowed)

    def test_unavailable_physical_capture_has_no_physical_latency(self) -> None:
        measurement = LatencyMeasurement(
            trial=1,
            capture_timestamp=None,
            output_timestamp=20.0,
            pipeline_timestamp=None,
            provenance=LatencyProvenance.PHYSICAL_CAPTURE_UNAVAILABLE,
            sample_start=0,
            sample_count=512,
            active_voice_count=1,
            mode="live",
            input_device="input",
            output_device="output",
            sample_rate=44100,
            block_size=2048,
            processing_segment_size=8192,
        )

        report = summarize_latency([measurement])
        self.assertIsNone(measurement.latency_ms)
        self.assertIsNone(report.mean_ms)
        self.assertEqual(report.physical_latency_unavailable, 1)


if __name__ == "__main__":
    unittest.main()
