"""Signal filters (paper §2.4.2).

Every filter here is *causal* by default: it computes each output from the
samples seen so far and never from the future. That matters because Thalamus
filters a live stream — a filter that needs lookahead necessarily delays
delivery, and a researcher prototyping a real-time system needs to know which
kind they are getting. Where a non-causal variant is worth having (Savitzky-Golay
smooths noticeably better when centered), it is available behind ``mode`` and its
latency is stated.

A channel whose window contains a gap (:data:`~thalamus.protocol.MISSING`) is
passed through unfiltered for that sample rather than being interpolated across:
silently inventing data is the last thing a simulation toolkit should do. Run a
:class:`~thalamus.processing.missing.MissingFillStage` first if you want gaps
filled before filtering.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

from ..protocol import MISSING, Sample
from .base import Stage, register


@register("moving_average")
class MovingAverageStage(Stage):
    """Unweighted mean over the last ``window`` samples of each channel.

    The cheapest useful smoother, and the one to reach for when you just want to
    see whether downstream code copes with a smoothed stream. Zero lag in the
    sense that it emits one sample per input, though it does of course attenuate
    and phase-shift the signal like any moving average.
    """

    def __init__(self, *, window: int = 5, channels: Optional[Sequence[str]] = None) -> None:
        super().__init__(channels=channels)
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self.window = window
        self._history: Dict[str, Deque[float]] = {}

    def apply(self, sample: Sample) -> Iterable[Sample]:
        def smooth(channel: str, value: float) -> float:
            history = self._history.setdefault(channel, deque(maxlen=self.window))
            history.append(value)
            return sum(history) / len(history)

        return [self.map_values(sample, smooth)]

    def reset(self) -> None:
        self._history.clear()


@register("exponential")
class ExponentialStage(Stage):
    """Exponential moving average: ``y[n] = a*x[n] + (1-a)*y[n-1]``.

    Constant memory and no window to size, which makes it the pragmatic choice
    for high-rate streams (EEG at 250 Hz and up).
    """

    def __init__(self, *, alpha: float = 0.3, channels: Optional[Sequence[str]] = None) -> None:
        super().__init__(channels=channels)
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self._state: Dict[str, float] = {}

    def apply(self, sample: Sample) -> Iterable[Sample]:
        def smooth(channel: str, value: float) -> float:
            previous = self._state.get(channel)
            current = (
                value if previous is None else self.alpha * value + (1 - self.alpha) * previous
            )
            self._state[channel] = current
            return current

        return [self.map_values(sample, smooth)]

    def reset(self) -> None:
        self._state.clear()


@register("kalman")
class KalmanStage(Stage):
    """Scalar Kalman filter, one independent estimator per channel [Li et al. 2015].

    The model is a random walk: the signal is assumed to drift by ``process_noise``
    per step and to be observed through sensor noise of variance
    ``measurement_noise``. Their *ratio* is what matters — raising
    ``measurement_noise`` tells the filter to trust its own prediction over the
    incoming sample, and so smooths harder at the cost of lagging real changes.

    Deliberately implemented in plain Python: a scalar Kalman filter is four lines
    of arithmetic, and keeping it dependency-free means the Core can filter
    without numpy or scipy installed.
    """

    def __init__(
        self,
        *,
        process_noise: float = 1e-3,
        measurement_noise: float = 1e-1,
        channels: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(channels=channels)
        if process_noise < 0 or measurement_noise <= 0:
            raise ValueError("process_noise must be >= 0 and measurement_noise must be > 0")
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self._estimate: Dict[str, float] = {}
        self._covariance: Dict[str, float] = {}

    def apply(self, sample: Sample) -> Iterable[Sample]:
        def step(channel: str, measurement: float) -> float:
            if channel not in self._estimate:
                # Seed from the first observation: with no prior, the measurement
                # *is* the best estimate, and its uncertainty is the sensor's.
                self._estimate[channel] = measurement
                self._covariance[channel] = self.measurement_noise
                return measurement

            # Predict: a random walk adds process noise to our uncertainty.
            covariance = self._covariance[channel] + self.process_noise
            # Update: weight the residual by how much we trust the sensor.
            gain = covariance / (covariance + self.measurement_noise)
            estimate = self._estimate[channel] + gain * (measurement - self._estimate[channel])

            self._estimate[channel] = estimate
            self._covariance[channel] = (1 - gain) * covariance
            return estimate

        return [self.map_values(sample, step)]

    def reset(self) -> None:
        self._estimate.clear()
        self._covariance.clear()


@register("savgol")
class SavitzkyGolayStage(Stage):
    """Savitzky-Golay filter [Savitzky & Golay 1964]: least-squares polynomial smoothing.

    Fits a degree-``polyorder`` polynomial to a sliding window and evaluates it at
    one point of that window. Unlike a moving average it preserves peak height and
    width, which is why it is the standard smoother for physiological traces.

    ``mode`` picks which point of the window is evaluated, and this is the whole
    trade-off:

    ``"causal"`` (default)
        Evaluate at the *newest* sample. No lag: output sample ``n`` is emitted on
        input sample ``n``. This is the only correct choice for a real-time
        pipeline, and it is what a filter running inside the Core must do.

    ``"centered"``
        Evaluate at the *middle* of the window. Smooths visibly better and is what
        offline analysis does, but output sample ``n`` is only emitted once input
        sample ``n + window//2`` arrives, so the stream is delayed by
        ``window // 2`` samples. The emitted sample carries the *center* sample's
        timestamp, so it stays correctly aligned in time with other streams.

    Needs SciPy: ``pip install thalamus[filters]``.
    """

    def __init__(
        self,
        *,
        window: int = 11,
        polyorder: int = 3,
        mode: str = "causal",
        channels: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__(channels=channels)
        if mode not in ("causal", "centered"):
            raise ValueError(f"mode must be 'causal' or 'centered', got {mode!r}")
        if window < 3:
            raise ValueError(f"window must be >= 3, got {window}")
        if polyorder >= window:
            raise ValueError(f"polyorder ({polyorder}) must be < window ({window})")
        if mode == "centered" and window % 2 == 0:
            raise ValueError(f"a centered window must be odd, got {window}")

        self.window = window
        self.polyorder = polyorder
        self.mode = mode

        self._savgol_coeffs = _require_savgol()
        self._buffer: Deque[Sample] = deque(maxlen=window)
        self._coeff_cache: Dict[Tuple[int, int], Any] = {}

    def _coefficients(self, length: int, position: int):
        key = (length, position)
        if key not in self._coeff_cache:
            self._coeff_cache[key] = self._savgol_coeffs(
                length, self.polyorder, pos=position, use="dot"
            )
        return self._coeff_cache[key]

    def _smooth(self, window: List[Sample], position: int) -> Sample:
        """Evaluate the fitted polynomial at ``window[position]`` for every channel."""
        target = window[position]
        out = target.copy()
        coefficients = self._coefficients(len(window), position)

        for channel in self.targets(target):
            values = [s.data.get(channel, MISSING) for s in window]
            if any(v is MISSING or not isinstance(v, (int, float)) for v in values):
                continue  # a gap in the window: leave this channel's raw value alone
            out.data[channel] = float(sum(c * float(v) for c, v in zip(coefficients, values)))
        return out

    def apply(self, sample: Sample) -> Iterable[Sample]:
        self._buffer.append(sample)
        window = list(self._buffer)

        if self.mode == "causal":
            # Start filtering as soon as a fit is defined (polyorder+1 points),
            # growing the window until it reaches full size. Before that, pass
            # the sample through raw rather than dropping or holding it.
            if len(window) < self.polyorder + 1:
                return [sample]
            return [self._smooth(window, len(window) - 1)]

        if len(window) < self.window:
            return []  # centered mode needs a full window before it can emit
        return [self._smooth(window, self.window // 2)]

    def flush(self) -> Iterable[Sample]:
        """Release the tail that centered mode is still holding, unfiltered."""
        if self.mode == "causal":
            return []
        tail = list(self._buffer)[self.window // 2 + 1 :]
        self._buffer.clear()
        return tail

    def reset(self) -> None:
        self._buffer.clear()


def _require_savgol():
    """Import scipy's coefficient helper, with an error a researcher can act on."""
    try:
        from scipy.signal import savgol_coeffs
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "the 'savgol' stage needs SciPy. Install it with: pip install 'thalamus[filters]'"
        ) from exc
    return savgol_coeffs
