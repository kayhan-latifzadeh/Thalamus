"""Devices that need no recording at all.

The paper's premise is that you should be able to prototype a study "even without
the need to install or purchase a specific device". A replay device gets you most
of the way, but it still needs a file — and the sample recordings for this project
live behind a download link, which means a fresh clone of the repository cannot
actually run anything. That is a bad first five minutes.

So: signal generators. ``thalamus demo`` streams a plausible 8-channel EEG cap, an
eye tracker that blinks, and a mouse, from nothing but arithmetic. Nobody should
mistake these for real physiology — they are for exercising *your* code paths
(does the client cope with 250 Hz? with gaps? with a device that dies?), which is
what the toolkit is for. When you want real signals, replay a real dataset.
"""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Dict, Iterator, Optional

from .base import Reading, RecordingDevice
from .profiles import available_profiles, get_profile

#: kind -> factory(params, rng, rate) -> f(t_seconds) -> value
_GENERATORS: Dict[str, Callable[..., Callable[[float], float]]] = {}


def generator(kind: str):
    def decorator(factory):
        _GENERATORS[kind] = factory
        return factory

    return decorator


@generator("sine")
def _sine(
    *, amplitude: float = 1.0, freq: float = 1.0, offset: float = 0.0, phase: float = 0.0, **_
):
    return lambda t: offset + amplitude * math.sin(2 * math.pi * freq * t + phase)


@generator("square")
def _square(*, amplitude: float = 1.0, freq: float = 1.0, offset: float = 0.0, **_):
    return lambda t: offset + amplitude * (1.0 if math.sin(2 * math.pi * freq * t) >= 0 else -1.0)


@generator("sawtooth")
def _sawtooth(*, amplitude: float = 1.0, freq: float = 1.0, offset: float = 0.0, **_):
    return lambda t: offset + amplitude * (2.0 * ((t * freq) % 1.0) - 1.0)


@generator("constant")
def _constant(*, value: float = 0.0, **_):
    return lambda _t: value


@generator("noise")
def _noise(*, sigma: float = 1.0, offset: float = 0.0, rng: random.Random, **_):
    return lambda _t: offset + rng.gauss(0.0, sigma)


@generator("random_walk")
def _random_walk(
    *,
    step: float = 1.0,
    start: float = 0.0,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    pull: float = 0.0,
    centre: Optional[float] = None,
    rng: random.Random,
    **_,
):
    """A drifting value, clamped to a range. The closest thing here to a gaze trace.

    ``pull`` makes the walk mean-reverting: each step is tugged back towards
    ``centre`` (which defaults to ``start``) by this fraction of its distance from it.

    It matters. A free random walk has no equilibrium — its spread grows without
    bound — so a 60-second gaze channel simulated as one inevitably wanders into a
    screen edge and *sticks* there against the clamp, and you have simulated a
    participant staring at the top of their monitor for a minute. Real gaze wanders
    around something. A ``pull`` of a few thousandths gives a trace that roams most
    of the screen and never pins to a corner.
    """
    origin = start if centre is None else centre
    state = {"value": start}

    def next_value(_t: float) -> float:
        value = state["value"]
        if pull:
            value += pull * (origin - value)
        value += rng.gauss(0.0, step)
        if minimum is not None:
            value = max(minimum, value)
        if maximum is not None:
            value = min(maximum, value)
        state["value"] = value
        return value

    return next_value


@generator("ramp")
def _ramp(
    *,
    start: float = 0.0,
    per_second: float = 1.0,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    quantize: Optional[float] = None,
    **_,
):
    """A value that climbs (or falls) steadily. A battery discharging, a drifting DC.

    ``quantize`` snaps the output to multiples of itself, which is how a real battery
    gauge behaves: the Unicorn reports in fifteenths, so a 26-minute recording contains
    exactly two distinct battery values, not a smooth slope. Code that watches for a
    *change* in battery level and code that watches for a *slope* are different code,
    and the difference only shows up if the simulation steps.
    """

    def value(t: float) -> float:
        v = start + per_second * t
        if minimum is not None:
            v = max(minimum, v)
        if maximum is not None:
            v = min(maximum, v)
        if quantize:
            v = round(v / quantize) * quantize
        return v

    return value


@generator("counter")
def _counter(*, start: float = 0.0, step: float = 1.0, rate: float = 1.0, **_):
    """A sample counter, as real amplifiers emit.

    Worth having because it is the one channel that can *prove* a packet was lost:
    a dropped sample leaves a hole in the count, whereas a dropped value does not.
    Simulate a flaky link with the ``dropout`` stage and this channel will show it.
    """
    return lambda t: start + step * round(t * rate)


@generator("eeg")
def _eeg(
    *,
    amplitude: float = 20.0,
    alpha_freq: float = 10.0,
    phase: float = 0.3,
    rng: random.Random,
    **_,
):
    """A caricature of an EEG channel: an alpha rhythm buried in 1/f-ish noise.

    Not a model of the brain. It is shaped like EEG — microvolt scale, dominant
    alpha, broadband background — so that plots look right and filters have
    something to bite on.
    """
    background = [(rng.uniform(1.0, 40.0), rng.uniform(0, 2 * math.pi)) for _ in range(12)]

    def value(t: float) -> float:
        alpha = amplitude * math.sin(2 * math.pi * alpha_freq * t + phase)
        noise = sum(
            (amplitude / (2.0 * freq)) * math.sin(2 * math.pi * freq * t + ph)
            for freq, ph in background
        )
        return alpha + noise + rng.gauss(0.0, amplitude / 10.0)

    return value


@generator("ecg")
def _ecg(*, heart_rate: float = 70.0, amplitude: float = 1.0, rng: random.Random, **_):
    """A caricature of an ECG lead: P-QRS-T as a sum of Gaussian bumps.

    Enough to drive the telemedicine broadcast scenario of §3.5, where what matters
    is that every cardiologist's tablet receives the identical stream — not that the
    stream is diagnostically real.
    """
    # (centre within the beat, width, height) for each deflection.
    waves = (
        (0.20, 0.025, 0.10),
        (0.38, 0.008, -0.15),
        (0.40, 0.008, 1.00),
        (0.42, 0.010, -0.25),
        (0.60, 0.040, 0.25),
    )

    def value(t: float) -> float:
        period = 60.0 / heart_rate
        phase = (t % period) / period
        total = sum(
            height * math.exp(-((phase - centre) ** 2) / (2 * width**2))
            for centre, width, height in waves
        )
        return amplitude * total + rng.gauss(0.0, 0.01)

    return value


class SyntheticDevice(RecordingDevice):
    """A device whose signals are generated, not recorded.

    Either name a real device and get its real columns::

        SyntheticDevice("eeg", profile="unicorn_hybrid_black").run()
        # -> EEG1..EEG8, AccelerometerX/Y/Z, GyroscopeX/Y/Z, BatteryLevel,
        #    Counter, ValidationIndicator, at 250 Hz

    or describe the signals yourself::

        SyntheticDevice(
            "eye_tracker",
            signals={
                "pupil": {"kind": "sine", "freq": 0.2, "amplitude": 0.5, "offset": 3.5},
                "gaze_x": {"kind": "random_walk", "step": 8, "start": 960, "minimum": 0,
                           "maximum": 1920},
            },
            rate=150,
        ).run()

    Prefer the first form. The channel names it produces are the ones the real
    hardware writes, so the analysis code you write against the simulation is the
    analysis code you run against the recording — which is the entire point of a
    dry run, and is lost the moment you invent your own column names.

    Every signal spec takes two options on top of its generator's own:

    ``noise``
        Standard deviation of Gaussian noise added on top. Real sensors are never
        this clean, and a filter has nothing to do without it.

    ``digits``
        Round to this many decimals; ``0`` makes the channel an integer, which is
        what a flag or a counter actually is on the wire.

    ``seed`` makes the stream reproducible, which matters more than it sounds: a
    bug that only shows up on a particular noise realization is one you want to be
    able to reproduce on a colleague's machine.
    """

    def __init__(
        self,
        device_id: str,
        signals: Optional[Dict[str, Dict[str, Any]]] = None,
        *,
        profile: Optional[str] = None,
        duration_s: Optional[float] = None,
        seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.profile = get_profile(profile) if profile else None

        # A profile supplies the rate and the channels; anything passed explicitly
        # still wins, so you can borrow a Unicorn's columns and run them at 500 Hz to
        # see what breaks.
        if self.profile is not None:
            signals = signals if signals is not None else self.profile.signals()
            kwargs.setdefault("rate", self.profile.rate)

        super().__init__(device_id, **kwargs)
        if not signals:
            raise ValueError(
                f"[{device_id}] needs either signals= or a profile=; "
                f"profiles: {', '.join(available_profiles())}"
            )
        if self.rate is None:
            raise ValueError(f"[{device_id}] a synthetic device needs a rate (Hz)")

        self.duration_s = duration_s
        self.seed = seed
        self._rng = random.Random(seed)
        self._signals = {name: self._build(name, spec) for name, spec in signals.items()}

    def _build(self, name: str, spec: Dict[str, Any]) -> Callable[[float], float]:
        params = dict(spec)
        kind = params.pop("kind", "sine")
        sigma = params.pop("noise", 0.0)
        digits = params.pop("digits", None)

        if kind not in _GENERATORS:
            raise ValueError(
                f"[{self.device_id}] unknown signal kind {kind!r} for channel {name!r}; "
                f"available: {', '.join(sorted(_GENERATORS))}"
            )

        # Each channel gets its own RNG stream, derived from the device seed, so that
        # adding a channel does not change the values of the existing ones.
        rng = random.Random(self._rng.random())
        try:
            base = _GENERATORS[kind](rng=rng, rate=self.rate, **params)
        except TypeError as exc:
            raise ValueError(f"[{self.device_id}] channel {name!r}: {exc}") from exc

        if not sigma and digits is None:
            return base

        def shaped(t: float) -> Any:
            value = base(t)
            if sigma:
                value += rng.gauss(0.0, sigma)
            if digits is None:
                return value
            return int(round(value)) if digits == 0 else round(value, digits)

        return shaped

    def describe(self) -> Dict[str, Any]:
        info = super().describe()
        if self.profile is not None:
            info.update(self.profile.describe())
        info.setdefault("channels", list(self._signals))
        info["synthetic"] = True
        return info

    def samples(self) -> Iterator[Reading]:
        interval = 1.0 / self.rate
        index = 0
        while True:
            t = index * interval
            if self.duration_s is not None and t >= self.duration_s:
                return
            yield {name: fn(t) for name, fn in self._signals.items()}
            index += 1
