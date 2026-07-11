"""Delay and packet loss.

Delay can be simulated either by shifting timestamps or by holding a buffer window,
and both are here. It is worth being precise about what each one does, because
they break *different* downstream code:

``timestamp`` mode
    Shifts the timestamp a sample carries, without holding the sample back. The
    stream arrives as fast as ever, but claims to be older than it is. This is
    what a device with a lagging clock or an internal processing pipeline looks
    like, and it is what breaks cross-device *synchronization* logic.

``buffer`` mode
    Holds ``samples`` samples back and releases the oldest as each new one
    arrives. Delivery genuinely lags by that many samples, and timestamps are
    untouched. This is what an under-read socket buffer looks like, and it is what
    breaks *real-time* logic that assumes the newest sample is the current state.

A third kind of delay — the network being slow — is *not* a stage, because it is a
property of the transport rather than of the signal. Ask for it on the
subscription instead (``latency_ms``), where the Core can hold delivery on the
event loop without stalling any other client. See :mod:`thalamus.core.router`.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, Optional, Sequence

from ..protocol import Sample
from .base import SeededStage, Stage, register


@register("delay")
class DelayStage(SeededStage):
    """Delay a stream, either by shifting its timestamps or by buffering it.

    ``jitter_ms`` adds a random ± spread to each timestamp shift, which models the
    variable latency of a real link rather than a constant one. It applies to
    ``timestamp`` mode only — jitter on a sample buffer is not a thing a buffer
    can do.
    """

    def __init__(
        self,
        *,
        mode: str = "timestamp",
        delay_ms: float = 100.0,
        jitter_ms: float = 0.0,
        samples: int = 10,
        seed: Optional[int] = None,
        channels: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(seed=seed, channels=channels)
        if mode not in ("timestamp", "buffer"):
            raise ValueError(f"mode must be 'timestamp' or 'buffer', got {mode!r}")
        if delay_ms < 0:
            raise ValueError(f"delay_ms must be >= 0, got {delay_ms}")
        if jitter_ms < 0:
            raise ValueError(f"jitter_ms must be >= 0, got {jitter_ms}")
        if mode == "buffer" and samples < 1:
            raise ValueError(f"samples must be >= 1 in buffer mode, got {samples}")

        self.mode = mode
        self.delay_ms = delay_ms
        self.jitter_ms = jitter_ms
        self.samples = samples
        self._buffer: Deque[Sample] = deque()

    def apply(self, sample: Sample) -> Iterable[Sample]:
        if self.mode == "timestamp":
            shift = self.delay_ms
            if self.jitter_ms:
                shift += self._rng.uniform(-self.jitter_ms, self.jitter_ms)
            shifted = sample.copy()
            shifted.timestamp = int(sample.timestamp + shift)
            return [shifted]

        self._buffer.append(sample)
        if len(self._buffer) <= self.samples:
            return []
        return [self._buffer.popleft()]

    def flush(self) -> Iterable[Sample]:
        """Release the held tail, so a delayed stream still ends with all its data."""
        held = list(self._buffer)
        self._buffer.clear()
        return held

    def reset(self) -> None:
        super().reset()
        self._buffer.clear()


@register("dropout")
class DropoutStage(SeededStage):
    """Drop whole samples, simulating packet loss on an unreliable link.

    Distinct from :class:`~thalamus.processing.missing.MissingInjectStage`: that
    one delivers a sample whose *channels* are empty (the eye tracker is running,
    the participant just looked away), whereas this one delivers *nothing at all*
    (the sample never made it). Client code that only handles the first case will
    happily miscompute a sampling rate when the second happens, which is exactly
    the kind of bug this toolkit exists to surface before a study runs.

    ``burst`` makes losses arrive in runs of consecutive samples, as real
    connection stalls do, rather than sprinkled independently.
    """

    def __init__(
        self,
        *,
        probability: float = 0.01,
        burst: int = 1,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(seed=seed)
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"probability must be in [0, 1], got {probability}")
        if burst < 1:
            raise ValueError(f"burst must be >= 1, got {burst}")
        self.probability = probability
        self.burst = burst
        self._remaining = 0

    def apply(self, sample: Sample) -> Iterable[Sample]:
        if self._remaining > 0:
            self._remaining -= 1
            return []
        if self._rng.random() < self.probability:
            self._remaining = self.burst - 1
            return []
        return [sample]

    def reset(self) -> None:
        super().reset()
        self._remaining = 0


@register("passthrough")
class PassthroughStage(Stage):
    """Does nothing. Useful as a placeholder in a config you are bisecting."""

    def apply(self, sample: Sample) -> Iterable[Sample]:
        return [sample]
