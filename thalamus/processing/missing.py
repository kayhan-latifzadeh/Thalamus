"""Missing values (paper §2.4.1).

Two directions, and a study needs both.

*Injecting* gaps is how you find out whether your analysis code survives them.
The paper's example is the right one: an eye tracker records nothing while the
participant looks off-screen. Note that real gaps are not independent coin flips
— a blink blanks a run of consecutive samples — so
:class:`MissingInjectStage` models them as bursts.

*Filling* gaps is Figure 2 of the paper: a recording arrives with ``"NA"`` in it
and the client needs numbers. :class:`MissingFillStage` replaces them, and
insists you choose *how*, because every choice is a lie of some kind and the
right one depends on what you are measuring. Zero-filling a pupil diameter makes
a plausible-looking trace with a hole punched in it; holding the last value makes
a plausible-looking trace with no hole at all, which is more dangerous.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

from ..protocol import MISSING, Sample
from .base import SeededStage, Stage, register


@register("missing_inject")
class MissingInjectStage(SeededStage):
    """Blank out channels at random, in bursts, to simulate dropped readings.

    On each sample, with probability ``probability``, a gap *starts*: this sample
    and the next ``burst - 1`` samples have their target channels set to
    :data:`~thalamus.protocol.MISSING`. Pass ``burst`` as a two-element
    ``[min, max]`` to draw each gap's length uniformly from that range, which is
    the realistic setting — blinks and off-screen glances are not all the same
    length.

    At 150 Hz, a blink lasting 100-400 ms is ``burst: [15, 60]``.
    """

    def __init__(
        self,
        *,
        probability: float = 0.01,
        burst: Union[int, Sequence[int]] = 1,
        seed: Optional[int] = None,
        channels: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(seed=seed, channels=channels)
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"probability must be in [0, 1], got {probability}")

        self.probability = probability
        self.burst: Tuple[int, int] = _parse_burst(burst)
        self._remaining = 0

    def apply(self, sample: Sample) -> Iterable[Sample]:
        if self._remaining == 0 and self._rng.random() < self.probability:
            self._remaining = self._rng.randint(*self.burst)

        if self._remaining == 0:
            return [sample]

        self._remaining -= 1
        out = sample.copy()
        for channel in self.targets(sample):
            out.data[channel] = MISSING
        return [out]

    def reset(self) -> None:
        super().reset()
        self._remaining = 0


@register("missing_fill")
class MissingFillStage(Stage):
    """Replace gaps with something a numeric client can consume.

    ``strategy``:

    ``"zero"`` (default)
        Substitute ``0``. What Figure 2 of the paper does, and what most existing
        pipelines expect. Remember that a zero is indistinguishable from a real
        measurement of zero downstream.

    ``"value"``
        Substitute ``value``. Use a sentinel your analysis can actually detect,
        e.g. ``-1`` for a pupil diameter that can never legitimately be negative.

    ``"hold"``
        Repeat the channel's last valid reading (last-observation-carried-forward).
        Keeps the signal continuous for filters that cannot cope with a step, at
        the cost of hiding the gap completely. A channel that has never had a
        valid reading falls back to ``value``.

    ``"drop"``
        Discard the whole sample if *any* target channel is missing. The honest
        choice when you would rather have a shorter clean stream than a complete
        dirty one — but it makes the stream irregularly sampled, so anything
        downstream must be timestamp-driven, not index-driven.
    """

    STRATEGIES = ("zero", "value", "hold", "drop")

    def __init__(
        self,
        *,
        strategy: str = "zero",
        value: Any = 0.0,
        channels: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(channels=channels)
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"strategy must be one of {', '.join(self.STRATEGIES)}, got {strategy!r}"
            )
        self.strategy = strategy
        self.value = 0.0 if strategy == "zero" else value
        self._last_valid: Dict[str, Any] = {}

    def targets(self, sample: Sample) -> list:
        # Unlike a filter, this stage must look at channels that are *missing* —
        # and a missing channel is not numeric, so the numeric default in
        # Stage.targets would skip exactly the ones we care about.
        if self.channels is None:
            return list(sample.data)
        return [c for c in self.channels if c in sample.data]

    def apply(self, sample: Sample) -> Iterable[Sample]:
        targets = self.targets(sample)

        if self.strategy == "drop":
            if any(sample.data[c] is MISSING for c in targets):
                return []
            return [sample]

        out = sample.copy()
        for channel in targets:
            if out.data[channel] is MISSING:
                if self.strategy == "hold":
                    out.data[channel] = self._last_valid.get(channel, self.value)
                else:
                    out.data[channel] = self.value
            elif self.strategy == "hold":
                self._last_valid[channel] = out.data[channel]
        return [out]

    def reset(self) -> None:
        self._last_valid.clear()


def _parse_burst(burst: Union[int, Sequence[int]]) -> Tuple[int, int]:
    """Accept ``5`` or ``[5, 60]`` and normalize to an inclusive ``(min, max)``."""
    if isinstance(burst, int):
        if burst < 1:
            raise ValueError(f"burst must be >= 1, got {burst}")
        return (burst, burst)

    try:
        low, high = burst
    except (TypeError, ValueError) as exc:
        raise ValueError(f"burst must be an int or a [min, max] pair, got {burst!r}") from exc

    low, high = int(low), int(high)
    if low < 1 or high < low:
        raise ValueError(f"burst must satisfy 1 <= min <= max, got {burst!r}")
    return (low, high)
