"""Four clients, one per thing a client can do.

Start a Core with data flowing first — `thalamus demo` is the easy way — then:

    python examples/clients.py basic       # subscribe and print
    python examples/clients.py filtered    # ask the Core to filter before sending
    python examples/clients.py synced      # several devices, aligned in time
    python examples/clients.py feedback    # a client that is also a recording device
"""

from __future__ import annotations

import sys

from thalamus import MISSING, ThalamusClient


def basic() -> None:
    """Subscribe to a device and print what arrives. The original client_example.py."""
    with ThalamusClient() as client:
        print("devices on offer:", [d["device_id"] for d in client.welcome["devices"]])
        client.subscribe("eye_tracker")

        for sample in client.stream():
            pupil = sample.data.get("pupil")
            # A gap is not a zero. The Core hands you MISSING and lets you decide;
            # if you would rather it decided for you, add a missing_fill stage.
            shown = "blink" if pupil is MISSING else f"{pupil:.2f} mm"
            print(f"{sample.timestamp}  pupil={shown}")


def filtered() -> None:
    """Have the Core filter the stream *before* it reaches us.

    The processing runs once, inside the Core, and is shared with any other client
    that asked for the same thing — so ten clients plotting a filtered EEG cost one
    filter, not ten (paper §3.1). Note the pipeline also fills the blinks first:
    stages run in order, and smoothing across a gap would be meaningless.
    """
    with ThalamusClient() as client:
        client.subscribe(
            "eye_tracker",
            channels=["pupil"],
            pipeline=[
                {"stage": "missing_fill", "strategy": "hold"},
                {"stage": "savgol", "window": 31, "polyorder": 3},
            ],
        )
        for sample in client.stream():
            print(f"{sample.timestamp}  pupil={sample.data['pupil']:.3f} mm  (smoothed)")


def synced() -> None:
    """Receive several devices aligned onto one timeline — Figure 4 of the paper.

    Instead of samples you get frames: one reading per device, taken at (nearly) the
    same instant. `complete` tells you whether every device actually had something
    within tolerance; when it does not, you get None rather than an invented value.
    """
    with ThalamusClient() as client:
        client.subscribe_synced(["eeg", "eye_tracker", "ecg"], reference="eeg", tolerance_ms=10)

        for frame in client.frames():
            eeg = frame["streams"]["eeg"]
            eye = frame["streams"]["eye_tracker"]
            if not frame["complete"]:
                print(f"{frame['timestamp']}  incomplete frame")
                continue
            print(
                f"{frame['timestamp']}  "
                f"Fp1={eeg['ch_Fp1']:8.2f} uV   "
                f"pupil={eye['pupil'] if eye['pupil'] is not None else float('nan'):.2f} mm"
            )


def feedback() -> None:
    """A client that is also a recording device — Figure 1's client #1 / device #5.

    We consume the eye tracker, compute something from it, and push the result back
    into Thalamus as a new device that *other* clients can subscribe to. This is how
    a live classifier joins the study: its output becomes just another stream.

    Watch it from another terminal with:  thalamus monitor attention_estimate
    """
    with ThalamusClient() as client:
        client.subscribe("eye_tracker", channels=["pupil"])
        client.send_event("session_start", note="feedback client online")

        baseline, count = 0.0, 0
        for sample in client.stream():
            pupil = sample.data.get("pupil")
            if pupil is MISSING:
                continue

            count += 1
            baseline += (pupil - baseline) / min(count, 500)  # running mean
            if count < 50:
                continue

            # Pupil dilation above baseline as a (very) crude arousal proxy.
            client.send_sample(
                "attention_estimate",
                {"arousal": round((pupil - baseline) / max(baseline, 1e-6), 4)},
                timestamp=sample.timestamp,
            )
            if count % 150 == 0:
                print(f"pushed {count} estimates; baseline pupil = {baseline:.3f} mm")


CLIENTS = {"basic": basic, "filtered": filtered, "synced": synced, "feedback": feedback}

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "basic"
    if name not in CLIENTS:
        sys.exit(f"unknown client {name!r}; try one of: {', '.join(CLIENTS)}")
    try:
        CLIENTS[name]()
    except KeyboardInterrupt:
        print("\nstopped.")
    except ConnectionRefusedError:
        sys.exit("no Thalamus Core on localhost:9001. Start one with:  thalamus demo")
