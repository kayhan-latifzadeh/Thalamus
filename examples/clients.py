"""Four clients, one per thing a client can do.

Start a Core with data flowing first — `thalamus demo` is the easy way — then:

    python examples/clients.py basic       # subscribe and print
    python examples/clients.py filtered    # ask the Core to filter before sending
    python examples/clients.py synced      # several devices, aligned in time
    python examples/clients.py feedback    # a client that is also a recording device

The channel names below — BPOGX, LPD, EEG1, BPOGV — are the ones the real Gazepoint
GP3 and g.tec Unicorn write, because the simulated devices are declared with those
devices' profiles. That is the whole trick: this code does not know or care whether
the samples came from hardware or from arithmetic, so it will run against either.
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
            # The GP3 reports a diameter per eye, in camera pixels. It also reports
            # whether it believes itself: BPOGV goes to 0 during a blink, and the gaze
            # and pupil columns for those samples mean nothing. Reading the flag is not
            # optional — it is the difference between a pupil baseline and a lie.
            if sample.data.get("BPOGV") != 1:
                print(f"{sample.timestamp}  blink")
                continue

            left, right = sample.data.get("LPD"), sample.data.get("RPD")
            if left is MISSING or right is MISSING:
                continue
            gaze = (sample.data["BPOGX"], sample.data["BPOGY"])
            print(
                f"{sample.timestamp}  pupil={(left + right) / 2:5.2f} px  "
                f"gaze=({gaze[0]:.3f}, {gaze[1]:.3f})"
            )


def filtered() -> None:
    """Have the Core filter the stream *before* it reaches us.

    The processing runs once, inside the Core, and is shared with any other client
    that asked for the same thing — so ten clients plotting a filtered pupil cost one
    filter, not ten (paper §3.1).

    Read the pipeline top to bottom; the order is the argument.

    1. ``validity_mask`` turns the tracker's own BPOGV=0 into real gaps. Skip it and
       everything below happily smooths garbage into your signal.
    2. ``missing_fill`` bridges those gaps, because a Savitzky-Golay window that
       straddles one has nothing to fit.
    3. ``savgol`` smooths what is left.

    Doing (3) without (1) is the single easiest way to publish a wrong pupil trace.
    """
    with ThalamusClient() as client:
        client.subscribe(
            "eye_tracker",
            channels=["LPD", "RPD", "BPOGV"],
            pipeline=[
                {"stage": "validity_mask", "profile": "gp3"},
                {"stage": "missing_fill", "strategy": "hold"},
                {"stage": "savgol", "window": 31, "polyorder": 3},
            ],
        )
        for sample in client.stream():
            pupil = (sample.data["LPD"] + sample.data["RPD"]) / 2
            print(f"{sample.timestamp}  pupil={pupil:.3f} px  (validated, smoothed)")


def synced() -> None:
    """Receive several devices aligned onto one timeline — Figure 4 of the paper.

    Instead of samples you get frames: one reading per device, taken at (nearly) the
    same instant. `complete` tells you whether every device actually had something
    within tolerance; when it does not, you get None rather than an invented value.

    This is the question the paper is really about — "what was the pupil doing when
    this EEG spike happened?" — and it is unanswerable without a common clock.
    """
    with ThalamusClient() as client:
        client.subscribe_synced(["eeg", "eye_tracker", "ecg"], reference="eeg", tolerance_ms=10)

        for frame in client.frames():
            if not frame["complete"]:
                print(f"{frame['timestamp']}  incomplete frame")
                continue

            eeg = frame["streams"]["eeg"]
            eye = frame["streams"]["eye_tracker"]
            fz = eeg["EEG1"]  # EEG1 is Fz on the Unicorn's cap
            pupil = eye["LPD"]
            print(
                f"{frame['timestamp']}  "
                f"Fz={fz:8.2f} uV   "
                f"pupil={float('nan') if pupil is None else pupil:5.2f} px"
            )


def feedback() -> None:
    """A client that is also a recording device — Figure 1's client #1 / device #5.

    We consume the eye tracker, compute something from it, and push the result back
    into Thalamus as a new device that *other* clients can subscribe to. This is how
    a live classifier joins the study: its output becomes just another stream.

    Watch it from another terminal with:  thalamus monitor attention_estimate
    """
    with ThalamusClient() as client:
        # Let the Core do the validity check for us, so the samples that arrive here
        # are only ever ones the tracker stands behind.
        client.subscribe(
            "eye_tracker",
            channels=["LPD", "RPD", "BPOGV"],
            pipeline=[{"stage": "validity_mask", "profile": "gp3"}],
        )
        client.send_event("session_start", note="feedback client online")

        baseline, count = 0.0, 0
        for sample in client.stream():
            left, right = sample.data["LPD"], sample.data["RPD"]
            if left is MISSING or right is MISSING:
                continue  # a blink: no pupil to measure, so measure nothing

            pupil = (left + right) / 2
            count += 1
            baseline += (pupil - baseline) / min(count, 500)  # running mean
            if count < 50:
                continue

            # Pupil dilation above baseline as a (very) crude arousal proxy. Crude
            # partly because GP3 pupil units are camera pixels, so they shift with head
            # distance — which is exactly why this is a *relative* measure.
            client.send_sample(
                "attention_estimate",
                {"arousal": round((pupil - baseline) / max(baseline, 1e-6), 4)},
                timestamp=sample.timestamp,
            )
            if count % 150 == 0:
                print(f"pushed {count} estimates; baseline pupil = {baseline:.3f} px")


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
