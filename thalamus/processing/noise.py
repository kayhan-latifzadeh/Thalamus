"""Noise injection.

Three kinds: fixed, random (uniform), and Gaussian. All of them
draw from a per-stage :class:`random.Random`, so passing ``seed`` makes a noisy
run exactly reproducible — a colleague re-running your study config gets the same
corrupted stream you did, which is the difference between a demonstration and an
experiment.

Noise is never added to a channel that is :data:`~thalamus.protocol.MISSING`: a
gap in the recording is not a value that can be perturbed.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

from ..protocol import Sample
from .base import SeededStage, register


@register("gaussian_noise")
class GaussianNoiseStage(SeededStage):
    """Add zero-mean Gaussian noise of standard deviation ``sigma``.

    The realistic default for sensor noise: thermal and electronic noise is Gaussian.

    With ``relative=True``, ``sigma`` is read as a *fraction of the current value*
    rather than an absolute amount, which is the right model when a sensor's error
    scales with its reading (``sigma=0.02`` is then "2% noise") and saves you from
    having to know a channel's units to noise it sensibly.
    """

    def __init__(
        self,
        *,
        sigma: float = 1.0,
        mu: float = 0.0,
        relative: bool = False,
        seed: Optional[int] = None,
        channels: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(seed=seed, channels=channels)
        if sigma < 0:
            raise ValueError(f"sigma must be >= 0, got {sigma}")
        self.sigma = sigma
        self.mu = mu
        self.relative = relative

    def apply(self, sample: Sample) -> Iterable[Sample]:
        def perturb(channel: str, value: float) -> float:
            scale = self.sigma * abs(value) if self.relative else self.sigma
            return value + self._rng.gauss(self.mu, scale) if scale else value

        return [self.map_values(sample, perturb)]


@register("uniform_noise")
class UniformNoiseStage(SeededStage):
    """Add noise drawn uniformly from ``[low, high]``: "random" noise.

    Useful for quantization-like error and for worst-case bounded perturbation,
    where Gaussian tails would be unrealistic.
    """

    def __init__(
        self,
        *,
        low: float = -1.0,
        high: float = 1.0,
        seed: Optional[int] = None,
        channels: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(seed=seed, channels=channels)
        if low > high:
            raise ValueError(f"low ({low}) must be <= high ({high})")
        self.low = low
        self.high = high

    def apply(self, sample: Sample) -> Iterable[Sample]:
        return [self.map_values(sample, lambda _c, v: v + self._rng.uniform(self.low, self.high))]


@register("constant_noise")
class ConstantNoiseStage(SeededStage):
    """Add a fixed ``offset`` to every sample: "fixed" noise.

    Models a miscalibrated or drifting sensor: a DC offset on an EEG electrode, an
    eye tracker that is systematically off by a few pixels. Constant bias breaks
    analysis code in ways random noise does not, so it is worth testing against.

    Set ``drift`` to have the offset grow by that much per sample, which simulates
    the thermal drift real electrodes show over a long session.
    """

    def __init__(
        self,
        *,
        offset: float = 1.0,
        drift: float = 0.0,
        channels: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(channels=channels)
        self.offset = offset
        self.drift = drift
        self._elapsed = 0

    def apply(self, sample: Sample) -> Iterable[Sample]:
        shift = self.offset + self.drift * self._elapsed
        self._elapsed += 1
        return [self.map_values(sample, lambda _c, v: v + shift)]

    def reset(self) -> None:
        super().reset()
        self._elapsed = 0
