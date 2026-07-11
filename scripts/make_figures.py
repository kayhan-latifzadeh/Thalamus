"""Regenerate the figures in the README from real runs of the toolkit.

    python scripts/make_figures.py

Nothing here is drawn from made-up numbers. Each figure starts a real Thalamus Core
on a real port, connects a real device to it, subscribes real clients with real
processing pipelines, and plots what actually came back out of the socket. If a stage
is broken, the figure is wrong — which is the point: a demo you cannot trust is worse
than no demo, and the old GIFs illustrated features that did not exist yet.

Needs: pip install matplotlib pillow  (plus thalamus[filters] for Savitzky-Golay)
"""

from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thalamus import MISSING, RecordingDevice, SyntheticDevice, ThalamusCore  # noqa: E402
from thalamus.protocol import LineDecoder, Sample, encode  # noqa: E402

ASSETS = Path(__file__).resolve().parent.parent / "assets"

# A mouse cursor trace: a signal whose shape makes filtering and delay obvious.
RAW = "#1f4e9c"
PROCESSED = "#c0392b"
GRID = "#dfe3e8"
FPS = 20


class MouseDevice(RecordingDevice):
    """A mouse cursor moving the way a mouse cursor moves: rest, dart, rest.

    A pure random walk would not do — real cursor traces sit still, accelerate, and
    settle, and it is precisely that shape (flat plateaus joined by steep ramps) that
    makes it obvious what a smoother, a delay, or a noise source is doing to it. The
    small tremor on top is real too: no hand holds a mouse perfectly still.
    """

    # (time_s, x_pixels) waypoints; the cursor eases between them.
    WAYPOINTS = [
        (0.0, 120),
        (0.35, 130),
        (0.8, 300),
        (1.15, 310),
        (1.5, 250),
        (1.85, 260),
        (2.1, 560),
        (2.35, 520),
        (2.5, 450),
    ]

    def __init__(self, device_id: str, *, duration_s: float = 2.5, jitter: float = 6.0, **kw):
        super().__init__(device_id, **kw)
        self.duration_s = duration_s
        self.jitter = jitter
        import random

        self._rng = random.Random(4)

    def samples(self):
        interval = 1.0 / self.rate
        n = int(self.duration_s * self.rate)
        for i in range(n):
            t = i * interval
            yield {"cursor_x": self._position(t) + self._rng.gauss(0, self.jitter)}

    def _position(self, t: float) -> float:
        points = self.WAYPOINTS
        for (t0, x0), (t1, x1) in zip(points, points[1:]):
            if t0 <= t <= t1:
                # Smoothstep between waypoints: eases out of rest and back into it.
                u = (t - t0) / (t1 - t0)
                return x0 + (x1 - x0) * (u * u * (3 - 2 * u))
        return points[-1][1]


# --------------------------------------------------------------------------- #
# capture: run the real thing and keep what comes off the socket
# --------------------------------------------------------------------------- #


async def _read_n(reader, want: int, kind: str = "sample", timeout: float = 20.0) -> List[Any]:
    """Read until ``want`` messages of the requested kind have arrived."""
    decoder = LineDecoder()
    out: List[Any] = []
    while len(out) < want:
        chunk = await asyncio.wait_for(reader.read(65536), timeout=timeout)
        if not chunk:
            break
        for obj in decoder.feed(chunk):
            if kind == "sample" and "type" not in obj:
                out.append(Sample.from_wire(obj))
            elif obj.get("type") == kind:
                out.append(obj)
    return out


async def capture(device, subscriptions: Dict[str, Dict[str, Any]], want: int) -> Dict[str, List]:
    """Stream ``device`` through a live Core and collect each subscription's output.

    ``subscriptions`` maps a name to the subscribe message that client should send.
    """
    core = ThalamusCore(host="127.0.0.1", device_port=0, client_port=0)
    await core.start()

    readers = {}
    writers = []
    for name, request in subscriptions.items():
        reader, writer = await asyncio.open_connection("127.0.0.1", core.client_port)
        await _read_n(reader, 1, kind="welcome")
        writer.write(encode(request))
        await writer.drain()
        await _read_n(reader, 1, kind="subscribed")
        readers[name] = reader
        writers.append(writer)

    device.host, device.port = "127.0.0.1", core.device_port
    thread = device.run_in_thread()

    kinds = {n: ("frame" if "sync" in r else "sample") for n, r in subscriptions.items()}
    results = dict(
        zip(
            readers,
            await asyncio.gather(*(_read_n(r, want, kinds[n]) for n, r in readers.items())),
        )
    )

    device.stop()
    thread.join(timeout=2)
    for writer in writers:
        writer.close()
    await core.stop()
    return results


def series(samples: List[Sample], channel: str):
    """(seconds since start, values) with gaps as NaN so matplotlib breaks the line."""
    t0 = samples[0].timestamp
    xs = [(s.timestamp - t0) / 1000.0 for s in samples]
    ys = [
        float("nan") if s.data.get(channel, MISSING) is MISSING else s.data[channel]
        for s in samples
    ]
    return xs, ys


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #


def two_panel(path: Path, left, right, *, titles, ylabel, xlabel="Time (s)", caption=""):
    """Animate two traces side by side, drawing left to right, as the old GIFs did."""
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.1), dpi=110, sharey=True)
    fig.patch.set_facecolor("white")

    panels = []
    for ax, (xs, ys), title, colour in zip(axes, (left, right), titles, (RAW, PROCESSED)):
        ax.set_title(title, fontsize=10, pad=8, color="#2c3e50")
        ax.set_xlabel(xlabel, fontsize=8.5, color="#5b6770")
        ax.set_xlim(min(xs), max(xs))
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.tick_params(labelsize=7.5, colors="#5b6770")
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)
        (line,) = ax.plot([], [], color=colour, linewidth=1.5)
        panels.append((xs, ys, line))

    axes[0].set_ylabel(ylabel, fontsize=8.5, color="#5b6770")
    finite = [v for _, ys, _ in panels for v in ys if not math.isnan(v)]
    pad = (max(finite) - min(finite)) * 0.12 or 1.0
    axes[0].set_ylim(min(finite) - pad, max(finite) + pad)

    if caption:
        fig.text(0.5, 0.015, caption, ha="center", fontsize=8, color="#8593a0")
    fig.tight_layout(rect=(0, 0.06 if caption else 0, 1, 1))

    total = max(len(ys) for _, ys, _ in panels)
    steps = 70

    def draw(frame):
        cut = int(total * (frame + 1) / steps)
        for xs, ys, line in panels:
            line.set_data(xs[:cut], ys[:cut])
        return [line for _, _, line in panels]

    anim = FuncAnimation(fig, draw, frames=steps, interval=1000 / FPS, blit=True)
    anim.save(path, writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print(f"  wrote {path.relative_to(ASSETS.parent)}  ({path.stat().st_size // 1024} KB)")


# --------------------------------------------------------------------------- #
# the figures
# --------------------------------------------------------------------------- #


async def filter_figure():
    """A live mouse trace, and the same trace Savitzky-Golay'd inside the Core."""
    raw, smooth = (
        await capture(
            MouseDevice("mouse", rate=60),
            {
                "raw": {"type": "subscribe", "devices": ["mouse"]},
                "savgol": {
                    "type": "subscribe",
                    "devices": [
                        {
                            "device_id": "mouse",
                            "pipeline": [{"stage": "savgol", "window": 21, "polyorder": 3}],
                        }
                    ],
                },
            },
            want=150,
        )
    ).values()

    two_panel(
        ASSETS / "filter_example.gif",
        series(raw, "cursor_x"),
        series(smooth, "cursor_x"),
        titles=("Original", "Savitzky-Golay (window 21, order 3)"),
        ylabel="Mouse cursor X (px)",
        caption="The filter runs inside Thalamus Core, once, and is shared by every "
        "client that asks for it.",
    )


async def noise_figure():
    """The same trace with Gaussian noise added by the Core."""
    raw, noisy = (
        await capture(
            MouseDevice("mouse", rate=60),
            {
                "raw": {"type": "subscribe", "devices": ["mouse"]},
                "noisy": {
                    "type": "subscribe",
                    "devices": [
                        {
                            "device_id": "mouse",
                            "pipeline": [{"stage": "gaussian_noise", "sigma": 18.0, "seed": 3}],
                        }
                    ],
                },
            },
            want=150,
        )
    ).values()

    two_panel(
        ASSETS / "noise_example.gif",
        series(raw, "cursor_x"),
        series(noisy, "cursor_x"),
        titles=("Original", "Gaussian noise (sigma 18 px, seed 3)"),
        ylabel="Mouse cursor X (px)",
        caption="Seeded, so the same corrupted stream can be reproduced on another machine.",
    )


async def delay_figure():
    """The same trace, arriving 300 ms late."""
    raw, delayed = (
        await capture(
            MouseDevice("mouse", rate=60),
            {
                "raw": {"type": "subscribe", "devices": ["mouse"]},
                "delayed": {
                    "type": "subscribe",
                    "devices": [
                        {
                            "device_id": "mouse",
                            "pipeline": [{"stage": "delay", "mode": "timestamp", "delay_ms": 300}],
                        }
                    ],
                },
            },
            want=150,
        )
    ).values()

    raw_xs, raw_ys = series(raw, "cursor_x")
    # Both streams share one clock, so the delayed one has to be plotted against the
    # *raw* origin — plotting it against its own would hide the very shift it shows.
    t0 = raw[0].timestamp
    delayed_xs = [(s.timestamp - t0) / 1000.0 for s in delayed]
    delayed_ys = [s.data["cursor_x"] for s in delayed]

    two_panel(
        ASSETS / "delay_example.gif",
        (raw_xs, raw_ys),
        (delayed_xs, delayed_ys),
        titles=("Original", "Delayed (+300 ms)"),
        ylabel="Mouse cursor X (px)",
        caption="Same clock on both axes: the delayed stream really does arrive "
        "300 ms into the past.",
    )


async def missing_figure():
    """What a GP3 blink actually looks like, and what `validity_mask` does about it.

    Not the figure you would draw from intuition. A blink is not an absence — the
    tracker freezes, holding the last gaze and pupil it believed and dropping BPOGV to
    0. The left panel is what the device hands you: flat plateaus that are
    indistinguishable from a very still eye. The right is the same stream after the
    Core has been told to read the flag.
    """
    device = SyntheticDevice("eye", profile="gp3", seed=5)

    # The real thing, only more often so a 2-second figure catches a few. `mode: hold`
    # is what makes the plateaus.
    blinks = [
        {
            "stage": "missing_inject",
            "mode": "hold",
            "probability": 0.012,
            "burst": [12, 30],
            "channels": ["BPOGX", "BPOGY", "LPD", "RPD"],
            "flag": "BPOGV",
            "seed": 21,
        }
    ]

    frozen, masked = (
        await capture(
            device,
            {
                "frozen": {
                    "type": "subscribe",
                    "devices": [{"device_id": "eye", "pipeline": blinks}],
                },
                "masked": {
                    "type": "subscribe",
                    "devices": [
                        {
                            "device_id": "eye",
                            "pipeline": blinks + [{"stage": "validity_mask", "profile": "gp3"}],
                        }
                    ],
                },
            },
            want=300,
        )
    ).values()

    two_panel(
        ASSETS / "missing_example.gif",
        series(frozen, "LPD"),
        series(masked, "LPD"),
        titles=(
            "As the GP3 records it (blink: BPOGV=0, values frozen)",
            "validity_mask: the blink becomes a gap",
        ),
        ylabel="Left pupil diameter LPD (px)",
        caption="The flat stretches on the left are blinks. Nothing in the data says "
        "so — only BPOGV does.",
    )


async def sync_figure():
    """EEG at 250 Hz and an eye tracker at 150 Hz, aligned onto one timeline."""
    core = ThalamusCore(host="127.0.0.1", device_port=0, client_port=0)
    await core.start()

    reader, writer = await asyncio.open_connection("127.0.0.1", core.client_port)
    await _read_n(reader, 1, kind="welcome")
    writer.write(
        encode(
            {
                "type": "subscribe",
                "sync": {
                    "devices": ["eeg", "eye"],
                    "reference": "eeg",
                    "tolerance_ms": 12,
                },
            }
        )
    )
    await writer.drain()
    await _read_n(reader, 1, kind="subscribed")

    # Two devices at different rates, with their real channel names.
    eeg = SyntheticDevice(
        "eeg",
        profile="unicorn_hybrid_black",
        seed=11,
        host="127.0.0.1",
        port=core.device_port,
    )
    eye = SyntheticDevice(
        "eye",
        profile="gp3",
        seed=12,
        host="127.0.0.1",
        port=core.device_port,
    )
    threads = [eeg.run_in_thread(), eye.run_in_thread()]

    frames = await _read_n(reader, 500, kind="frame")

    for device in (eeg, eye):
        device.stop()
    for thread in threads:
        thread.join(timeout=2)
    writer.close()
    await core.stop()

    complete = [f for f in frames if f["complete"]]
    t0 = complete[0]["timestamp"]
    times = [(f["timestamp"] - t0) / 1000.0 for f in complete]
    eeg_y = [f["streams"]["eeg"]["EEG1"] for f in complete]
    pupil_y = [f["streams"]["eye"]["LPD"] for f in complete]

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8.4, 3.6), dpi=110, sharex=True)
    fig.patch.set_facecolor("white")

    for ax, label, colour in (
        (top, "Unicorn EEG1 / Fz (uV)", RAW),
        (bottom, "GP3 pupil LPD (px)", PROCESSED),
    ):
        ax.set_ylabel(label, fontsize=8.5, color="#5b6770")
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.tick_params(labelsize=7.5, colors="#5b6770")
        ax.set_xlim(times[0], times[-1])
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(GRID)
        del colour

    top.set_title(
        "One frame per EEG sample: Unicorn 250 Hz + GP3 150 Hz, aligned on UTC timestamps",
        fontsize=10,
        pad=8,
        color="#2c3e50",
    )
    bottom.set_xlabel("Time (s)", fontsize=8.5, color="#5b6770")
    top.set_ylim(min(eeg_y) * 1.15, max(eeg_y) * 1.15)
    span = max(pupil_y) - min(pupil_y)
    bottom.set_ylim(min(pupil_y) - span * 0.3, max(pupil_y) + span * 0.3)

    (eeg_line,) = top.plot([], [], color=RAW, linewidth=1.0)
    (pupil_line,) = bottom.plot([], [], color=PROCESSED, linewidth=1.6)
    cursors = [
        ax.axvline(times[0], color="#95a5a6", linewidth=0.9, linestyle="--") for ax in (top, bottom)
    ]

    fig.text(
        0.5,
        0.015,
        "Every frame carries one reading from each device, taken at the same instant.",
        ha="center",
        fontsize=8,
        color="#8593a0",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    steps = 70

    def draw(frame):
        cut = int(len(times) * (frame + 1) / steps)
        eeg_line.set_data(times[:cut], eeg_y[:cut])
        pupil_line.set_data(times[:cut], pupil_y[:cut])
        for cursor in cursors:
            cursor.set_xdata([times[max(cut - 1, 0)]] * 2)
        return [eeg_line, pupil_line, *cursors]

    anim = FuncAnimation(fig, draw, frames=steps, interval=1000 / FPS, blit=True)
    path = ASSETS / "synchronisation_example.gif"
    anim.save(path, writer=PillowWriter(fps=FPS))
    plt.close(fig)
    print(f"  wrote {path.relative_to(ASSETS.parent)}  ({path.stat().st_size // 1024} KB)")

    # The figure claims the streams are aligned; check that they are, and say so.
    dropped = len(frames) - len(complete)
    print(f"    {len(frames)} frames, {dropped} incomplete, tolerance 12 ms")


async def main():
    print("Regenerating README figures from live Thalamus runs...\n")
    await filter_figure()
    await noise_figure()
    await delay_figure()
    await missing_figure()
    await sync_figure()
    print("\nDone. Every trace above came off a real socket.")


if __name__ == "__main__":
    asyncio.run(main())
