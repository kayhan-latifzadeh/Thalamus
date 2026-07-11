"""Missing values.

Two directions, and a study needs both.

*Injecting* gaps is how you find out whether your analysis code survives them.
The canonical example: an eye tracker records nothing while the participant
looks off-screen. Note that real gaps are not independent coin flips
— a blink blanks a run of consecutive samples — so
:class:`MissingInjectStage` models them as bursts.

*Filling* gaps is the other direction: a recording arrives with ``"NA"`` in it
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

    ``flag``
        The name of the device's validity channel, if it has one. During a gap it is
        set to ``flag_invalid`` (default ``0``) instead of being blanked. This is what
        real hardware does: a Gazepoint does not stop emitting rows during a blink, it
        emits rows with ``BPOGV=0``.

    ``mode``
        What the channels do during the gap.

        ``"blank"`` (default)
            Set them to :data:`~thalamus.protocol.MISSING`. The honest, easy case: a
            client cannot help but notice.

        ``"hold"``
            *Freeze them at their last valid values.* This is the ugly case, and it is
            what a real Gazepoint GP3 does. In a 26-minute reference recording, 115
            of the 116 multi-sample blinks have every column byte-identical to the
            others in the run, and 116 of the 118 blink onsets repeat the last valid
            sample exactly. So a blink is not an absence — it is 131 ms of perfectly
            plausible, perfectly unchanging numbers, with a ``0`` in ``BPOGV`` beside
            them that nothing forces you to read.

            Simulate with ``"blank"`` and your client sees an obvious gap that the
            hardware will never give it. Simulate with ``"hold"`` and the only thing
            standing between you and a pupil baseline polluted by 118 frozen blinks is
            :class:`ValidityMaskStage` — which is exactly the situation you will be in
            on the day, and exactly what a dry run is for.
    """

    MODES = ("blank", "hold")

    def __init__(
        self,
        *,
        probability: float = 0.01,
        burst: Union[int, Sequence[int]] = 1,
        seed: Optional[int] = None,
        channels: Optional[Sequence[str]] = None,
        flag: Optional[str] = None,
        flag_invalid: Any = 0,
        mode: str = "blank",
    ) -> None:
        super().__init__(seed=seed, channels=channels)
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"probability must be in [0, 1], got {probability}")
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {', '.join(self.MODES)}, got {mode!r}")

        self.probability = probability
        self.burst: Tuple[int, int] = _parse_burst(burst)
        self.flag = flag
        self.flag_invalid = flag_invalid
        self.mode = mode
        self._remaining = 0
        self._held: Dict[str, Any] = {}
        self._last_valid: Dict[str, Any] = {}

    def apply(self, sample: Sample) -> Iterable[Sample]:
        targets = [c for c in self.targets(sample) if c != self.flag]

        if self._remaining == 0 and self._rng.random() < self.probability:
            self._remaining = self._rng.randint(*self.burst)
            if self.mode == "hold":
                # Freeze at the reading *before* the gap, and hold that for the whole
                # run. Not this sample's own value — the last one the device believed.
                # In the recording, 116 of 118 blink onsets repeat the preceding valid
                # row exactly, so an off-by-one here would leave the first sample of
                # every blink carrying a fresh, plausible, wrong measurement.
                self._held = {c: self._last_valid.get(c, sample.data[c]) for c in targets}

        if self._remaining == 0:
            self._last_valid = {c: sample.data[c] for c in targets}
            return [sample]

        self._remaining -= 1
        out = sample.copy()
        for channel in targets:
            if self.mode == "hold":
                out.data[channel] = self._held.get(channel, sample.data[channel])
            else:
                out.data[channel] = MISSING
        if self.flag is not None and self.flag in out.data:
            out.data[self.flag] = self.flag_invalid
        return [out]

    def reset(self) -> None:
        super().reset()
        self._remaining = 0
        self._held = {}
        self._last_valid = {}


@register("validity_mask")
class ValidityMaskStage(Stage):
    """Turn a device's own validity flag into honest gaps.

    Real sensors tell you when they failed, in a side channel: the Gazepoint GP3
    writes ``BPOGV=0`` during a blink, the Unicorn writes ``ValidationIndicator=0``
    for a corrupt sample. What they do *not* reliably do is blank the data columns.
    A GP3 export mid-blink still has numbers in ``BPOGX`` and ``LPD`` — stale, zeroed,
    or simply meaningless — and they average into your pupil trace as if they were
    measurements. Nothing downstream can tell the difference, because by then there
    is no difference to tell.

    This stage reads the flag and blanks what it does not vouch for, so that a gap in
    the *recording* becomes a :data:`~thalamus.protocol.MISSING` on the wire and stays
    one all the way to the client. It is the first stage to put on any replay of real
    hardware, and with a ``profile:`` set the config can write itself::

        - stage: validity_mask        # blanks BPOGX/BPOGY/LPD/RPD when BPOGV != 1

    ``flag``
        The validity channel. Defaults to the device profile's, and is required if
        there is no profile.

    ``valid``
        What the flag reads when the sample is good. Default ``1``.

    ``channels``
        What the flag vouches for. Defaults to the profile's list, or to every channel
        except the flag itself.

    ``drop``
        Discard the whole sample instead of blanking it. Honest, and it makes the
        stream irregularly sampled — which is a real property of the data, and better
        faced now than assumed away.

    ``keep_flag``
        Leave the flag channel on the wire (default). Set ``false`` to drop it once it
        has done its job.
    """

    def __init__(
        self,
        *,
        flag: Optional[str] = None,
        valid: Any = 1,
        channels: Optional[Sequence[str]] = None,
        drop: bool = False,
        keep_flag: bool = True,
        profile: Optional[str] = None,
    ) -> None:
        super().__init__(channels=channels)

        if profile:
            from ..devices.profiles import get_profile

            spec = get_profile(profile)
            flag = flag or spec.validity_flag
            if channels is None and spec.validity_flag:
                self.channels = spec.covered_by_flag()

        if not flag:
            raise ValueError(
                "validity_mask needs flag= (the device's validity channel, "
                "e.g. BPOGV), or a profile= that declares one"
            )

        self.flag = flag
        self.valid = valid
        self.drop = drop
        self.keep_flag = keep_flag

    def targets(self, sample: Sample) -> list:
        if self.channels is None:
            return [c for c in sample.data if c != self.flag]
        return [c for c in self.channels if c in sample.data and c != self.flag]

    def apply(self, sample: Sample) -> Iterable[Sample]:
        if self.flag not in sample.data:
            # The flag is not here. Say nothing and pass the sample through: a device
            # that does not report validity is not a device that is reporting invalid.
            return [sample]

        value = sample.data[self.flag]
        if value is not MISSING and _matches(value, self.valid):
            if self.keep_flag:
                return [sample]
            out = sample.copy()
            out.data.pop(self.flag, None)
            return [out]

        if self.drop:
            return []

        out = sample.copy()
        for channel in self.targets(sample):
            out.data[channel] = MISSING
        if not self.keep_flag:
            out.data.pop(self.flag, None)
        return [out]


def _matches(value: Any, valid: Any) -> bool:
    """``1``, ``1.0``, and ``"1"`` all mean valid. CSV does not preserve the type."""
    if value == valid:
        return True
    try:
        return float(value) == float(valid)
    except (TypeError, ValueError):
        return False


@register("missing_fill")
class MissingFillStage(Stage):
    """Replace gaps with something a numeric client can consume.

    ``strategy``:

    ``"zero"`` (default)
        Substitute ``0``. The most common choice, and what most existing
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
