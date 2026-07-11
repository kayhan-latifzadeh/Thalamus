"""The hub: device registry, subscriptions, and dispatch.

This is Thalamus Core's brain, and it is deliberately free of sockets. It takes
parsed samples in and hands rendered messages to *sinks*; :mod:`thalamus.core.server`
is the only place that knows what a TCP connection is. That split is what makes
the routing logic — which is where the interesting behaviour lives — testable
without opening a port.

Two design points are worth stating explicitly.

**Pipelines are computed once and shared.** Ten clients subscribing to the same
device with the same processing spec get *one* pipeline instance, run once per
sample, fanned out to all ten. That is why a subscription's pipeline is keyed by its canonical spec
rather than by the client that asked for it. Two clients asking for *different*
processing still get one pipeline each, so a client can take raw EEG while another
takes it filtered — the two never interfere, because stages are stateful and each
shared pipeline owns its own.

**Clients can be devices.** A client connection may send data samples as well as
receive them, and they are routed exactly as if they had come from a recording
device. This is the feedback loop in the architecture diagram: the client that
feeds its own signal (a mouse trace, a classifier's output) back into the hub for
everyone else to consume.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence

from ..processing import Pipeline, PipelineSpecError, build_pipeline, pipeline_key
from ..protocol import Sample
from ..sync import Synchronizer

logger = logging.getLogger(__name__)


class Sink(Protocol):
    """Anything the router can push a message to. A TCP client, or a test double."""

    id: str

    def send(self, message: Any) -> None:
        """Deliver one message. Must not block; drop rather than stall the hub."""

    def send_later(self, delay_ms: float, message: Any) -> None:
        """Deliver after ``delay_ms`` of wall-clock time, preserving order."""


@dataclass
class DeviceInfo:
    """What the Core knows about one device, learned as it streams.

    A device may announce itself with a ``hello``, but it does not have to — the
    channel list and rate are inferred from the samples themselves, so a
    three-line device that just writes JSON to a socket still shows up correctly
    in ``thalamus devices``.
    """

    device_id: str
    channels: List[str] = field(default_factory=list)
    declared_rate: Optional[float] = None
    connected: bool = False
    samples: int = 0
    first_seen: Optional[int] = None
    last_seen: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    #: Exponentially-weighted mean inter-sample interval, in ms.
    _mean_interval: Optional[float] = None

    @property
    def effective_rate(self) -> Optional[float]:
        """Measured sample rate in Hz, as opposed to the one the device claims.

        Worth watching: a device that says 250 Hz but delivers 190 Hz is precisely
        the kind of thing device stress-testing is meant to
        catch, and you would rather catch it in a dry run than in the analysis.
        """
        if not self._mean_interval:
            return None
        return round(1000.0 / self._mean_interval, 1)

    def observe(self, sample: Sample) -> None:
        if self.last_seen is not None and sample.timestamp > self.last_seen:
            interval = sample.timestamp - self.last_seen
            if self._mean_interval is None:
                self._mean_interval = float(interval)
            else:
                self._mean_interval = 0.98 * self._mean_interval + 0.02 * interval

        if self.first_seen is None:
            self.first_seen = sample.timestamp
        self.last_seen = sample.timestamp
        self.samples += 1

        for channel in sample.data:
            if channel not in self.channels:
                self.channels.append(channel)

    def to_wire(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "channels": self.channels,
            "connected": self.connected,
            "samples": self.samples,
            "declared_rate": self.declared_rate,
            "effective_rate": self.effective_rate,
            "metadata": self.metadata,
        }


@dataclass
class Subscription:
    """One client's interest in one device."""

    sink: Sink
    device_id: str
    channels: Optional[List[str]] = None
    latency_ms: float = 0.0
    key: str = "[]"

    def deliver(self, sample: Sample) -> None:
        message = sample.select(self.channels)
        if self.latency_ms > 0:
            self.sink.send_later(self.latency_ms, message)
        else:
            self.sink.send(message)


class SharedPipeline:
    """One pipeline instance, and every subscription that wants its output."""

    def __init__(self, key: str, spec: Optional[Sequence[Dict[str, Any]]]) -> None:
        self.key = key
        self.spec = list(spec or ())
        self.pipeline: Pipeline = build_pipeline(spec)
        self.subscriptions: List[Subscription] = []

    def dispatch(self, sample: Sample) -> None:
        for output in self.pipeline.process(sample):
            for subscription in self.subscriptions:
                subscription.deliver(output)


class SyncSubscription:
    """A client's interest in several devices *aligned to each other*."""

    def __init__(self, sink: Sink, synchronizer: Synchronizer, latency_ms: float = 0.0) -> None:
        self.sink = sink
        self.synchronizer = synchronizer
        self.latency_ms = latency_ms

    @property
    def devices(self) -> List[str]:
        return self.synchronizer.devices

    def push(self, sample: Sample) -> None:
        self._emit(self.synchronizer.push(sample))

    def tick(self) -> None:
        self._emit(self.synchronizer.tick())

    def _emit(self, frames) -> None:
        for frame in frames:
            message = frame.to_wire()
            if self.latency_ms > 0:
                self.sink.send_later(self.latency_ms, message)
            else:
                self.sink.send(message)


class Router:
    """Routes samples from devices to the clients that asked for them."""

    def __init__(self) -> None:
        self.devices: Dict[str, DeviceInfo] = {}
        #: device_id -> pipeline_key -> SharedPipeline
        self._pipelines: Dict[str, Dict[str, SharedPipeline]] = {}
        #: device_id -> pipeline applied at ingest, before anyone subscribes
        self._ingest: Dict[str, Pipeline] = {}
        #: sink id -> its subscriptions, so a disconnect can be undone cleanly
        self._by_sink: Dict[str, List[Subscription]] = {}
        self._sync_subs: Dict[str, List[SyncSubscription]] = {}
        self._sinks: Dict[str, Sink] = {}
        self.samples_in = 0
        self.samples_out = 0

    def set_ingest_pipeline(self, device_id: str, spec: Optional[Sequence[Dict[str, Any]]]) -> None:
        """Give a device a pipeline that runs before *anything* subscribes to it.

        This is where a *simulated device's own imperfections* belong — the eye
        tracker that loses the pupil when the participant blinks, the electrode with
        a DC offset, the flaky link that drops packets. Every subscriber sees them,
        because they are part of what the device *is*; a client cannot opt out of
        them any more than it could opt out of a real sensor's noise. Contrast with
        a subscription pipeline, which is processing a *client* chose to apply to
        what it received, and which other clients neither see nor pay for.

        Not settable by clients, only by the study configuration. A client that
        could rewrite the ingest pipeline could silently corrupt every other
        client's stream.
        """
        if spec:
            self._ingest[device_id] = build_pipeline(spec)
        else:
            self._ingest.pop(device_id, None)

    # -- devices ---------------------------------------------------------------

    def device_info(self, device_id: str) -> DeviceInfo:
        info = self.devices.get(device_id)
        if info is None:
            info = DeviceInfo(device_id=device_id)
            self.devices[device_id] = info
        return info

    def device_connected(self, device_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        info = self.device_info(device_id)
        info.connected = True
        if metadata:
            info.metadata.update(metadata)
            if "channels" in metadata and isinstance(metadata["channels"], list):
                for channel in metadata["channels"]:
                    if channel not in info.channels:
                        info.channels.append(channel)
            rate = metadata.get("sample_rate") or metadata.get("rate")
            if isinstance(rate, (int, float)):
                info.declared_rate = float(rate)

        logger.info("device connected: %s", device_id)
        self.broadcast({"type": "device_connected", "device": info.to_wire()})

    def device_disconnected(self, device_id: str) -> None:
        info = self.devices.get(device_id)
        if info is None:
            return
        info.connected = False
        logger.info("device disconnected: %s (%d samples)", device_id, info.samples)

        # Let buffered stages release what they are holding, so a delayed stream
        # is not silently truncated by the disconnect.
        for shared in self._pipelines.get(device_id, {}).values():
            for output in shared.pipeline.flush():
                for subscription in shared.subscriptions:
                    subscription.deliver(output)
            shared.pipeline.reset()

        self.broadcast({"type": "device_disconnected", "device_id": device_id})

    # -- dispatch --------------------------------------------------------------

    def route(self, sample: Sample) -> None:
        """The hot path: one sample in, zero or more messages out."""
        info = self.device_info(sample.device_id)
        info.observe(sample)
        self.samples_in += 1

        # The device's own imperfections come first: they are part of the signal
        # everyone receives. An ingest stage may drop the sample outright (packet
        # loss) or turn one into several, so fan out over whatever comes back.
        ingest = self._ingest.get(sample.device_id)
        samples = ingest.process(sample) if ingest else [sample]

        for processed in samples:
            for shared in self._pipelines.get(sample.device_id, {}).values():
                shared.dispatch(processed)

            for sync_sub in self._sync_subs.get(sample.device_id, ()):
                sync_sub.push(processed)

    def tick(self) -> None:
        """Nudge every synchronizer so a stalled device cannot hold frames forever."""
        seen = set()
        for subs in self._sync_subs.values():
            for sub in subs:
                if id(sub) not in seen:
                    seen.add(id(sub))
                    sub.tick()

    def broadcast(self, message: Dict[str, Any]) -> None:
        """Send a control message to every connected client."""
        for sink in list(self._sinks.values()):
            sink.send(message)

    # -- subscriptions ---------------------------------------------------------

    def add_sink(self, sink: Sink) -> None:
        self._sinks[sink.id] = sink
        self._by_sink.setdefault(sink.id, [])

    def subscribe(
        self,
        sink: Sink,
        device_id: str,
        *,
        channels: Optional[Sequence[str]] = None,
        pipeline: Optional[Sequence[Dict[str, Any]]] = None,
        latency_ms: float = 0.0,
    ) -> Subscription:
        """Subscribe ``sink`` to ``device_id``.

        Subscribing to a device that has not connected yet is allowed and normal:
        clients routinely start before devices do, and the subscription simply
        starts producing when the device shows up.

        Raises :class:`~thalamus.processing.PipelineSpecError` if the pipeline
        spec is bad, so a client gets a clear error rather than a silent no-op.
        """
        if latency_ms < 0:
            raise PipelineSpecError(f"latency_ms must be >= 0, got {latency_ms}")

        key = pipeline_key(pipeline)
        shared = self._pipelines.setdefault(device_id, {}).get(key)
        if shared is None:
            shared = SharedPipeline(key, pipeline)  # may raise PipelineSpecError
            self._pipelines[device_id][key] = shared

        subscription = Subscription(
            sink=sink,
            device_id=device_id,
            channels=list(channels) if channels else None,
            latency_ms=latency_ms,
            key=key,
        )
        shared.subscriptions.append(subscription)
        self._by_sink.setdefault(sink.id, []).append(subscription)
        self.add_sink(sink)

        logger.info(
            "client %s subscribed to %s (channels=%s, pipeline=%s, latency=%sms)",
            sink.id,
            device_id,
            channels or "all",
            [s.get("stage") for s in (pipeline or ())] or "none",
            latency_ms,
        )
        return subscription

    def subscribe_sync(
        self,
        sink: Sink,
        devices: Sequence[str],
        *,
        reference: Optional[str] = None,
        tolerance_ms: float = 20.0,
        timeout_ms: float = 500.0,
        latency_ms: float = 0.0,
    ) -> SyncSubscription:
        """Subscribe ``sink`` to a time-aligned view of several devices."""
        synchronizer = Synchronizer(
            devices,
            reference=reference,
            tolerance_ms=tolerance_ms,
            timeout_ms=timeout_ms,
        )
        sync_sub = SyncSubscription(sink, synchronizer, latency_ms=latency_ms)
        for device_id in devices:
            self._sync_subs.setdefault(device_id, []).append(sync_sub)
        self.add_sink(sink)

        logger.info(
            "client %s subscribed to synced view of %s (reference=%s, tolerance=%sms)",
            sink.id,
            list(devices),
            synchronizer.reference,
            tolerance_ms,
        )
        return sync_sub

    def unsubscribe(self, sink: Sink, device_id: Optional[str] = None) -> int:
        """Remove ``sink``'s subscriptions — to one device, or (``None``) to all.

        Also tears down any shared pipeline left with no subscribers, which is what
        stops a long-running Core from accumulating dead pipelines that still do
        work on every sample.
        """
        removed = 0
        keep: List[Subscription] = []

        for subscription in self._by_sink.get(sink.id, []):
            if device_id is not None and subscription.device_id != device_id:
                keep.append(subscription)
                continue

            shared = self._pipelines.get(subscription.device_id, {}).get(subscription.key)
            if shared and subscription in shared.subscriptions:
                shared.subscriptions.remove(subscription)
                removed += 1
                if not shared.subscriptions:
                    del self._pipelines[subscription.device_id][subscription.key]

        self._by_sink[sink.id] = keep

        if device_id is None:
            for device, subs in list(self._sync_subs.items()):
                remaining = [s for s in subs if s.sink.id != sink.id]
                removed += len(subs) - len(remaining)
                if remaining:
                    self._sync_subs[device] = remaining
                else:
                    del self._sync_subs[device]

        return removed

    def remove_sink(self, sink: Sink) -> None:
        """Drop a client entirely. Safe to call for a client that never subscribed."""
        self.unsubscribe(sink)
        self._by_sink.pop(sink.id, None)
        self._sinks.pop(sink.id, None)

    # -- introspection ---------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Everything a client or the CLI needs to know about the Core's state."""
        return {
            "devices": [info.to_wire() for info in self.devices.values()],
            "clients": len(self._sinks),
            "subscriptions": sum(len(subs) for subs in self._by_sink.values()),
            "samples_in": self.samples_in,
        }
