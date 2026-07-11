"""Routing: who gets what, and how often it is computed.

The router is deliberately socket-free, which is what lets these tests exercise the
real dispatch logic against a fake sink instead of against a TCP connection.
"""

from __future__ import annotations

import pytest

from thalamus.core.router import Router
from thalamus.processing import PipelineSpecError
from thalamus.protocol import MISSING, Sample


class FakeSink:
    """A client that records what it was sent, instead of writing it to a socket."""

    def __init__(self, name="client"):
        self.id = name
        self.received = []
        self.delayed = []

    def send(self, message):
        self.received.append(message)

    def send_later(self, delay_ms, message):
        self.delayed.append((delay_ms, message))

    @property
    def samples(self):
        return [m for m in self.received if isinstance(m, Sample)]

    @property
    def controls(self):
        return [m for m in self.received if isinstance(m, dict)]


def sample(device="eeg", timestamp=1000, **channels):
    return Sample(device, timestamp, channels or {"x": 1.0})


class TestSubscriptions:
    def test_a_subscriber_receives_the_stream(self):
        router, sink = Router(), FakeSink()
        router.subscribe(sink, "eeg")
        router.route(sample(x=1.0))
        assert [s.data["x"] for s in sink.samples] == [1.0]

    def test_a_non_subscriber_receives_nothing(self):
        router, sink = Router(), FakeSink()
        router.subscribe(sink, "eeg")
        router.route(sample(device="eye"))
        assert sink.samples == []

    def test_you_can_subscribe_to_a_device_that_has_not_connected_yet(self):
        # Clients routinely start before devices do. Rejecting them would force an
        # ordering on the study that nothing else requires.
        router, sink = Router(), FakeSink()
        router.subscribe(sink, "not_here_yet")
        router.route(sample(device="not_here_yet", x=5.0))
        assert len(sink.samples) == 1

    def test_channels_are_filtered_per_subscription(self):
        router = Router()
        everything, just_fp1 = FakeSink("a"), FakeSink("b")
        router.subscribe(everything, "eeg")
        router.subscribe(just_fp1, "eeg", channels=["Fp1"])

        router.route(sample(Fp1=1.0, Fp2=2.0))

        assert everything.samples[0].data == {"Fp1": 1.0, "Fp2": 2.0}
        assert just_fp1.samples[0].data == {"Fp1": 1.0}

    def test_unsubscribing_stops_the_stream(self):
        router, sink = Router(), FakeSink()
        router.subscribe(sink, "eeg")
        assert router.unsubscribe(sink, "eeg") == 1
        router.route(sample())
        assert sink.samples == []

    def test_a_disconnecting_client_is_forgotten_entirely(self):
        router, sink = Router(), FakeSink()
        router.subscribe(sink, "eeg")
        router.remove_sink(sink)
        router.route(sample())
        assert sink.samples == []
        assert router.snapshot()["clients"] == 0

    def test_a_bad_pipeline_is_rejected_rather_than_silently_ignored(self):
        router, sink = Router(), FakeSink()
        with pytest.raises(PipelineSpecError):
            router.subscribe(sink, "eeg", pipeline=[{"stage": "does_not_exist"}])


class TestSharedPipelines:
    def test_two_clients_asking_for_the_same_processing_share_one_pipeline(self):
        # §3.1's efficiency argument, made concrete: the filter runs once, not twice.
        router = Router()
        a, b = FakeSink("a"), FakeSink("b")
        spec = [{"stage": "moving_average", "window": 3}]

        router.subscribe(a, "eeg", pipeline=spec)
        router.subscribe(
            b, "eeg", pipeline=list(reversed([dict(spec[0])]))
        )  # same, written differently

        assert len(router._pipelines["eeg"]) == 1

        router.route(sample(x=1.0))
        router.route(sample(timestamp=1010, x=10.0))
        assert [s.data["x"] for s in a.samples] == [s.data["x"] for s in b.samples]

    def test_two_clients_asking_for_different_processing_do_not_interfere(self):
        # One client takes the raw EEG while another takes it filtered. Stages are
        # stateful, so sharing one instance between them would corrupt both.
        router = Router()
        raw, filtered = FakeSink("raw"), FakeSink("filtered")

        router.subscribe(raw, "eeg")
        router.subscribe(filtered, "eeg", pipeline=[{"stage": "constant_noise", "offset": 100}])

        router.route(sample(x=1.0))

        assert raw.samples[0].data["x"] == 1.0
        assert filtered.samples[0].data["x"] == 101.0

    def test_a_shared_pipeline_is_torn_down_when_its_last_subscriber_leaves(self):
        # Otherwise a long-running Core accumulates dead pipelines that still do work
        # on every single sample.
        router = Router()
        a, b = FakeSink("a"), FakeSink("b")
        spec = [{"stage": "moving_average"}]

        router.subscribe(a, "eeg", pipeline=spec)
        router.subscribe(b, "eeg", pipeline=spec)

        router.unsubscribe(a, "eeg")
        assert len(router._pipelines["eeg"]) == 1  # b still wants it
        router.unsubscribe(b, "eeg")
        assert len(router._pipelines["eeg"]) == 0


class TestIngestPipeline:
    def test_a_devices_own_imperfections_reach_every_subscriber(self):
        # Nobody can opt out of a simulated sensor's noise, any more than they could
        # opt out of a real one's.
        router = Router()
        a, b = FakeSink("a"), FakeSink("b")
        router.set_ingest_pipeline("eye", [{"stage": "missing_inject", "probability": 1.0}])
        router.subscribe(a, "eye")
        router.subscribe(b, "eye")

        router.route(sample(device="eye", pupil=3.5))

        assert a.samples[0].data["pupil"] is MISSING
        assert b.samples[0].data["pupil"] is MISSING

    def test_ingest_runs_before_the_subscription_pipeline(self):
        router, sink = Router(), FakeSink()
        router.set_ingest_pipeline("eye", [{"stage": "missing_inject", "probability": 1.0}])
        router.subscribe(sink, "eye", pipeline=[{"stage": "missing_fill", "strategy": "zero"}])

        router.route(sample(device="eye", pupil=3.5))

        # The device lost the reading; the client chose to fill the hole with a zero.
        assert sink.samples[0].data["pupil"] == 0.0

    def test_a_sample_dropped_at_ingest_reaches_nobody(self):
        router, sink = Router(), FakeSink()
        router.set_ingest_pipeline("ecg", [{"stage": "dropout", "probability": 1.0}])
        router.subscribe(sink, "ecg")
        router.route(sample(device="ecg"))
        assert sink.samples == []


class TestLatency:
    def test_per_client_latency_delays_only_that_client(self):
        router = Router()
        near, far = FakeSink("near"), FakeSink("far")
        router.subscribe(near, "eeg")
        router.subscribe(far, "eeg", latency_ms=200)

        router.route(sample())

        assert len(near.samples) == 1 and near.delayed == []
        assert far.received == [] and [d for d, _ in far.delayed] == [200]


class TestDeviceRegistry:
    def test_a_devices_channels_are_learned_from_its_samples(self):
        # A device does not have to announce itself; three lines of JSON are enough.
        router = Router()
        router.route(sample(Fp1=1.0, Fp2=2.0))
        assert router.device_info("eeg").channels == ["Fp1", "Fp2"]

    def test_the_measured_rate_is_reported_not_the_claimed_one(self):
        router = Router()
        router.device_connected("eeg", {"sample_rate": 250})
        for n in range(200):
            router.route(sample(timestamp=1000 + n * 8))  # actually 125 Hz

        info = router.device_info("eeg")
        assert info.declared_rate == 250
        assert info.effective_rate == pytest.approx(125, rel=0.05)

    def test_clients_are_told_when_a_device_connects_and_dies(self):
        router, sink = Router(), FakeSink()
        router.subscribe(sink, "eeg")

        router.device_connected("eeg")
        router.device_disconnected("eeg")

        kinds = [m["type"] for m in sink.controls if "type" in m]
        assert "device_connected" in kinds
        assert "device_disconnected" in kinds

    def test_a_disconnect_flushes_what_a_delay_stage_was_holding(self):
        # Otherwise the tail of a delayed stream vanishes when the device unplugs.
        router, sink = Router(), FakeSink()
        router.subscribe(sink, "eeg", pipeline=[{"stage": "delay", "mode": "buffer", "samples": 5}])

        for n in range(3):
            router.route(sample(timestamp=1000 + n, x=float(n)))
        assert sink.samples == []

        router.device_disconnected("eeg")
        assert [s.data["x"] for s in sink.samples] == [0.0, 1.0, 2.0]


class TestSyncSubscription:
    def test_a_synced_subscriber_receives_frames_instead_of_samples(self):
        router, sink = Router(), FakeSink()
        router.subscribe_sync(sink, ["eeg", "eye"], reference="eeg", tolerance_ms=20)

        router.route(Sample("eye", 998, {"pupil": 3.5}))
        router.route(Sample("eeg", 1000, {"Fp1": 1.0}))
        router.route(Sample("eye", 1010, {"pupil": 3.6}))

        frames = [m for m in sink.controls if m.get("type") == "frame"]
        assert len(frames) == 1
        assert frames[0]["complete"]
        assert frames[0]["streams"]["eye"]["pupil"] == 3.5
