"""Timestamp-based stream synchronization.

The problem: an EEG cap runs at 250 Hz, an eye tracker at 150 Hz, and a webcam at
30 Hz. Their samples never land on the same instant, they arrive over separate
sockets in an order that means nothing, and any of them may stall. To ask "what
was the pupil diameter when this EEG spike happened", something has to align them
on their UTC timestamps. That is this module.

The strategy is *nearest-neighbour against a reference stream*. One device is the
reference (typically the fastest, or the one whose events you care about); each of
its samples becomes a frame, and every other stream contributes the sample nearest
in time, provided that sample is within ``tolerance_ms``. A stream with nothing
close enough contributes ``None`` — an honest gap, rather than an interpolated
guess.

The subtlety is *when it is safe to emit*. A frame for reference time ``t`` cannot
be emitted the moment ``t`` arrives, because a closer sample from another stream
may still be in flight. So a frame is only released once every other stream has
delivered a sample at or after ``t``, which proves nothing closer is coming. That
gives correct alignment at the cost of a bounded lag. If a stream dies, waiting
forever would stall the whole synchronizer, so :meth:`Synchronizer.tick` releases
frames that have waited longer than ``timeout_ms`` with whatever is available.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Sequence

from ..protocol import Sample


@dataclass
class Frame:
    """Several devices' readings, aligned to one instant.

    ``timestamp`` is the reference sample's timestamp. ``streams`` maps every
    device in the synchronizer to its nearest sample, or to ``None`` if that
    device had nothing within tolerance.
    """

    timestamp: int
    streams: Dict[str, Optional[Sample]] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        """True if every device contributed a sample."""
        return all(s is not None for s in self.streams.values())

    def to_wire(self) -> Dict[str, Any]:
        return {
            "type": "frame",
            "timestamp": self.timestamp,
            "complete": self.complete,
            "streams": {
                device: (sample.to_wire() if sample is not None else None)
                for device, sample in self.streams.items()
            },
        }


class Synchronizer:
    """Aligns several device streams onto a reference stream's timeline.

    Feed it every sample with :meth:`push`, which returns whatever frames became
    releasable. Call :meth:`tick` periodically with the current wall-clock time so
    that a stalled device cannot hold frames hostage indefinitely.

    Not thread-safe; drive it from a single task.
    """

    def __init__(
        self,
        devices: Sequence[str],
        *,
        reference: Optional[str] = None,
        tolerance_ms: float = 20.0,
        timeout_ms: float = 500.0,
        max_buffer: int = 10_000,
    ) -> None:
        if len(devices) < 2:
            raise ValueError("synchronizing needs at least two devices")
        if reference is not None and reference not in devices:
            raise ValueError(f"reference {reference!r} is not among devices {list(devices)}")
        if tolerance_ms < 0:
            raise ValueError(f"tolerance_ms must be >= 0, got {tolerance_ms}")

        self.devices = list(devices)
        #: Default to the first device listed; the caller is expected to list the
        #: stream whose timeline matters (usually the fastest one) first.
        self.reference = reference or self.devices[0]
        self.tolerance_ms = tolerance_ms
        self.timeout_ms = timeout_ms
        self.max_buffer = max_buffer

        self._others = [d for d in self.devices if d != self.reference]
        self._buffers: Dict[str, Deque[Sample]] = {d: deque() for d in self.devices}
        #: Wall-clock arrival time of each pending reference sample, so that
        #: timeouts are measured against real time rather than signal time.
        self._arrivals: Deque[float] = deque()
        self._dropped_late = 0

    @property
    def dropped_late(self) -> int:
        """Samples discarded because they arrived after their frame had gone out."""
        return self._dropped_late

    def push(self, sample: Sample, *, now_ms: Optional[float] = None) -> List[Frame]:
        """Add one sample and return every frame that is now safe to release."""
        if sample.device_id not in self._buffers:
            return []

        buffer = self._buffers[sample.device_id]

        # Streams can arrive slightly out of order (separate sockets, jitter, a
        # delay stage). Insert in timestamp order rather than assuming sorted
        # input, walking back from the end because disorder is always local.
        if buffer and sample.timestamp < buffer[-1].timestamp:
            if sample.device_id != self.reference and sample.timestamp < self._oldest_pending():
                # Too late to matter: its frame has already been emitted.
                self._dropped_late += 1
                return []
            position = len(buffer)
            while position > 0 and buffer[position - 1].timestamp > sample.timestamp:
                position -= 1
            buffer.insert(position, sample)
        else:
            buffer.append(sample)

        if sample.device_id == self.reference:
            self._arrivals.append(now_ms if now_ms is not None else _monotonic_ms())

        self._enforce_buffer_cap(sample.device_id)
        return self._drain(now_ms=now_ms)

    def tick(self, now_ms: Optional[float] = None) -> List[Frame]:
        """Release frames whose wait has exceeded ``timeout_ms``.

        Call this on a timer. Without it, one dead device silently stops the whole
        synchronizer — which, in a toolkit whose job is surfacing exactly that
        kind of failure, would be an unfortunate way to fail.
        """
        return self._drain(now_ms=now_ms, allow_timeout=True)

    def flush(self) -> List[Frame]:
        """Emit every pending frame with whatever data is on hand. For end of stream."""
        frames = []
        while self._buffers[self.reference]:
            frames.append(self._emit_next())
        return frames

    # -- internals -------------------------------------------------------------

    def _oldest_pending(self) -> float:
        reference_buffer = self._buffers[self.reference]
        if not reference_buffer:
            return float("-inf")
        return reference_buffer[0].timestamp - self.tolerance_ms

    def _drain(self, *, now_ms: Optional[float], allow_timeout: bool = False) -> List[Frame]:
        frames: List[Frame] = []
        reference_buffer = self._buffers[self.reference]

        while reference_buffer:
            target = reference_buffer[0].timestamp

            # Safe to emit once every other stream has moved past `target`: only
            # then is the nearest sample known to be the nearest sample.
            settled = all(
                self._buffers[d] and self._buffers[d][-1].timestamp >= target for d in self._others
            )

            if not settled:
                if not allow_timeout:
                    break
                waited = (now_ms if now_ms is not None else _monotonic_ms()) - self._arrivals[0]
                if waited < self.timeout_ms:
                    break
                # Give up on the stragglers and emit with what we have.

            frames.append(self._emit_next())

        return frames

    def _emit_next(self) -> Frame:
        reference_sample = self._buffers[self.reference].popleft()
        if self._arrivals:
            self._arrivals.popleft()

        target = reference_sample.timestamp
        frame = Frame(timestamp=target, streams={self.reference: reference_sample})

        for device in self._others:
            buffer = self._buffers[device]
            nearest = self._take_nearest(buffer, target)
            frame.streams[device] = nearest

        return frame

    def _take_nearest(self, buffer: Deque[Sample], target: int) -> Optional[Sample]:
        """The sample nearest ``target``, or ``None`` if none is within tolerance.

        The nearest sample is always one of two: the newest sample at or before
        ``target``, or the oldest one after it. So we discard everything older than
        that left neighbour — no later frame can want it, since reference
        timestamps only increase — and compare the two survivors.

        Note that the chosen sample is *not* consumed. When the reference stream is
        faster than this one (EEG at 250 Hz against an eye tracker at 150 Hz) the
        same eye-tracking sample really is the nearest neighbour of several
        consecutive EEG samples, and must be attached to each of their frames. That
        is nearest-neighbour upsampling, and it is the intended behaviour — but it
        does mean a slow stream's values repeat across frames, which anything
        computing statistics over frames needs to know.
        """
        while len(buffer) >= 2 and buffer[1].timestamp <= target:
            buffer.popleft()

        if not buffer:
            return None

        best = buffer[0]
        if best.timestamp <= target and len(buffer) >= 2:
            right = buffer[1]
            if abs(right.timestamp - target) < abs(best.timestamp - target):
                best = right

        if abs(best.timestamp - target) > self.tolerance_ms:
            return None
        return best

    def _enforce_buffer_cap(self, device: str) -> None:
        buffer = self._buffers[device]
        while len(buffer) > self.max_buffer:
            buffer.popleft()
            self._dropped_late += 1
            if device == self.reference and self._arrivals:
                self._arrivals.popleft()


def _monotonic_ms() -> float:
    import time

    return time.monotonic() * 1000
