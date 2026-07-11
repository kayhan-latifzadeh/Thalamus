"""The three devices from the original release, rewritten against the new SDK.

Each was ~35 lines of hand-written socket loop. Each is now one line, because the
socket loop lives in RecordingDevice and is written exactly once.

    python examples/devices.py eeg          # Unicorn Hybrid Black, 250 Hz
    python examples/devices.py eye          # Gazepoint GP3, 150 Hz
    python examples/devices.py webcam       # Logitech C505e, 30 Hz  [needs thalamus[video]]
    python examples/devices.py real         # a real device: see RealDevice below

Start `thalamus serve` first.
"""

from __future__ import annotations

import random
import sys
import time

from thalamus import RecordingDevice, ReplayDevice

DATA = "examples/data"


def unicorn_hybrid_black_eeg() -> RecordingDevice:
    """https://www.unicorn-bi.com/unicorn-hybrid-black — 8 channels at 250 Hz."""
    return ReplayDevice("unicorn_hybrid_black_eeg", f"{DATA}/eeg.csv", rate=250, loop=True)


def gp3_eye_tracker() -> RecordingDevice:
    """https://www.gazept.com/product/gazepoint-gp3-eye-tracker — 150 Hz."""
    return ReplayDevice("gp3_eye_tracker", f"{DATA}/eye-tracking.csv", rate=150, loop=True)


def logitech_c505e_webcam() -> RecordingDevice:
    """https://www.logitech.com/products/webcams/c505e-business-webcam — 30 fps."""
    from thalamus.devices import VideoDevice

    return VideoDevice("logitech_c505e_webcam", f"{DATA}/webcam.mp4", loop=True, max_width=640)


class RealDevice(RecordingDevice):
    """What integrating an *actual* sensor looks like.

    This is the whole contract: yield readings, one dict per sample. Thalamus does
    the connecting, the reconnecting, the JSON, the UTC timestamps, and the pacing.

    The paper (§2.2) notes the two things that make real devices awkward — no UTC
    timestamp, and no clean access to the data stream — and both are handled here,
    in a controller you write once per device:

    * No timestamp? Omit it and Thalamus stamps the sample as it goes out. (Better,
      if the device gives you *some* clock, is to convert it to UTC ms yourself and
      include it, since that removes your own transport latency from the number.)
    * Data only reachable through a file the vendor's software writes? Tail the
      file, as below. Through a proprietary SDK? Call it, and yield what it returns.
    """

    def samples(self):
        while True:
            # Replace with: self.sdk.read(), or a line read from the vendor's file.
            yield {
                "temperature": 36.5 + random.gauss(0, 0.05),
                "skin_conductance": 4.0 + random.gauss(0, 0.2),
            }
            time.sleep(0.1)  # only because this fake source has no natural blocking


DEVICES = {
    "eeg": unicorn_hybrid_black_eeg,
    "eye": gp3_eye_tracker,
    "webcam": logitech_c505e_webcam,
    "real": lambda: RealDevice("skin_sensor", rate=10),
}

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "eeg"
    if name not in DEVICES:
        sys.exit(f"unknown device {name!r}; try one of: {', '.join(DEVICES)}")

    device = DEVICES[name]()
    print(f"streaming {device.device_id} -> {device.host}:{device.port}  (Ctrl-C to stop)")
    try:
        device.run()
    except KeyboardInterrupt:
        print(f"\nstopped after {device.sent} samples")
