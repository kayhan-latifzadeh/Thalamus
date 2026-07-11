<p align="center">
  <img src="assets/logo.jpg" alt="Thalamus" width="200">
  <br>
  <b>A Multimodal Sensing and Simulation Toolkit</b>
  <br>
  <sub>Prototype a physiological study before you buy the hardware, book the lab, or pay a participant.</sub>
</p>

<p align="center">
  <a href="https://doi.org/10.1145/3708319.3733687"><img src="https://img.shields.io/badge/paper-UMAP%20Adjunct%20'25-blue" alt="paper"></a>
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="license">
</p>

---

Running a study with EEG, eye tracking, and physiological sensors is expensive and
slow, and most of what goes wrong goes wrong in the software. Thalamus lets you find
that out first: it streams real or simulated signals from any number of devices,
synchronizes them on UTC timestamps, and does to them all the things a real study
does (drops packets, loses the pupil mid-blink, adds noise, lags the network), so
that your recording and analysis code meets those problems in a dry run rather than
in front of a participant.

## Sixty seconds

```shell
pip install -e .
thalamus demo
```

That is a complete three-device study: a g.tec Unicorn Hybrid Black EEG cap at 250 Hz on
a lossy link, a Gazepoint GP3 eye tracker at 150 Hz that blinks, and an ECG. No data
files, no hardware. In another terminal:

```shell
thalamus devices                                    # what's connected, at what real rate
thalamus monitor eye_tracker                        # watch a stream (and its blinks)
thalamus monitor eeg eye_tracker ecg --sync         # all three, aligned on one timeline
thalamus record eeg eye_tracker --out ./rec         # write it to CSV
```

```
DEVICE                       RATE    SAMPLES  CHANNELS
eeg                      250.0 Hz        978  EEG1, EEG2, EEG3, EEG4, EEG5, EEG6, E...
ecg                      127.8 Hz        504  lead_ii
eye_tracker              150.2 Hz        590  BPOGX, BPOGY, BPOGV, LPD, RPD
```

Note the rate column: that is the rate the device is *actually* achieving, not the one
it claims. If it says `190/250 Hz!`, you have learned something.

And note the channel names. They are not `gaze_x` and `ch_Fp1`. They are `BPOGX` and
`EEG1`, the columns those two devices really write. See below.

## The columns are the real ones

<!-- The single most useful thing in the toolkit, so it goes first. -->

A simulation whose channels are called `pupil` and `gaze_x` is a simulation you have to
rewrite the day the hardware arrives, because the Gazepoint writes `BPOGX`, `BPOGY`,
`BPOGV`, `LPD`, `RPD`, and that rewrite is exactly the wasted work this toolkit exists
to prevent. So devices can name real hardware:

```yaml
- id: eye_tracker
  type: synthetic              # nothing plugged in
  profile: gp3
```

```shell
thalamus profiles              # what's known
thalamus profiles gp3          # every column, its unit, and what it means
```

You get the real column names, the real sampling rate, realistic value ranges (pupil
diameters around 16 px, gaze normalized 0..1 rather than in pixels, an accelerometer
that reads 1 g because the head is upright), and the device's real failure modes.
Swap `type: synthetic` for `type: replay` when the recording exists, and **the client
code does not change**. That is the whole point.

| profile | device | rate | channels |
|---|---|---|---|
| `unicorn_hybrid_black` | g.tec Unicorn Hybrid Black | 250 Hz | `EEG1`-`EEG8`, IMU, `BatteryLevel`, `Counter`, `ValidationIndicator` |
| `gp3` | Gazepoint GP3 HD | 150 Hz | `BPOGX`, `BPOGY`, `BPOGV`, `LPD`, `RPD` |
| `c505e` | Logitech C505e | 30 fps | `frame` |

These are the three devices used in the paper. The channel names and rates were taken
from the recordings themselves, and [a test pins them there](tests/test_profiles.py). If
someone renames a channel to something tidier, the suite fails, because the hardware
will not rename it back.

### Validity flags, and what a blink really looks like

Real sensors tell you when they failed, in a side channel: the GP3 sets `BPOGV=0` during
a blink, the Unicorn sets `ValidationIndicator=0` for a corrupt sample.

What they do *not* do is stop producing data. Measured over the paper's own 26-minute
GP3 recording (230,974 samples, 118 blinks):

- **115 of the 116 multi-sample blinks have every column identical throughout**, and
- **116 of the 118 blink onsets repeat the preceding valid row exactly.**

The tracker does not blank the pupil during a blink. It *freezes*, holding the last
value it believed for a median of 131 ms, and the only thing that says so is `BPOGV`.
Those rows look exactly like a very still eye. Average them into a pupil baseline and
nothing downstream can tell, because by then there is nothing to tell.

So Thalamus simulates the blink the device actually has, not the kind one you would
draw: `missing_inject` with `mode: hold`. And `validity_mask` is what saves you:
it reads the flag and blanks what the flag does not vouch for, turning a failure in the
*recording* into a real gap on the wire:

```python
client.subscribe("eye_tracker", pipeline=[
    {"stage": "validity_mask", "profile": "gp3"},   # BPOGV=0 -> BPOGX/BPOGY/LPD/RPD are gaps
    {"stage": "missing_fill", "strategy": "hold"},  # ...now bridge them
    {"stage": "savgol", "window": 31, "polyorder": 3},
])
```

It is the first stage to put on any replay of real hardware. The left panel of the
figure below is what you get without it.

## How it fits together

<img src="assets/architecture.png" alt="Architecture of Thalamus">

**Thalamus Core** is the hub. Devices connect to one port and stream samples in;
clients connect to another and subscribe to whatever they want. Everything is JSON,
one object per line, over plain TCP, so a client does not have to be Python, or on
the same machine, or even on the same continent.

**Recording devices** are real or simulated. A simulated one replays a CSV recording
or generates signals from nothing; a real one is a ~10-line subclass. Every sample
carries a UTC millisecond timestamp, which is what makes cross-device synchronization
possible at all.

**Clients** subscribe to devices, and may also *send* samples. A client that streams
its classifier's output back into the hub becomes a recording device that other
clients can subscribe to (Recording Device #5 in the figure).

## Describe a study in one file

```yaml
# study.yaml, run it with: thalamus run study.yaml
devices:
  - id: eeg
    type: replay
    path: data/eeg.csv           # no `rate:`, so replay honours the file's own timing,
    profile: unicorn_hybrid_black   # jitter and all, which is the honest simulation
    loop: true
    simulate:
      - stage: constant_noise    # an electrode drifting as the gel dries
        offset: 0.5
        drift: 0.0002
        channels: [EEG1, EEG2, EEG3, EEG4, EEG5, EEG6, EEG7, EEG8]   # not the battery!

  - id: eye_tracker
    type: synthetic              # generated: no recording needed
    profile: gp3                 # ...but with the GP3's real columns
    seed: 2                      # a seed makes the whole run reproducible
    simulate:
      - stage: missing_inject    # blinks, as measured: one per ~13 s, median 131 ms
        mode: hold               # the tracker FREEZES; it does not blank. See below.
        probability: 0.0005
        burst: [6, 34]
        channels: [BPOGX, BPOGY, LPD, RPD]
        flag: BPOGV              # the tracker doesn't go silent; it says BPOGV=0
```

A device that names a profile and says nothing about what goes wrong with it inherits
what *actually* goes wrong with it, so the two lines `type: synthetic` / `profile: gp3`
already give you the blinks above. `thalamus run` prints the stages each device ended up
with on startup, so it is a shortcut, never a secret. Write `simulate: []` to turn it off.

Note what the Unicorn inherits: **nothing**. It dropped no packets and flagged no bad
samples in 26 minutes of real recording, so its profile simulates neither. Failure modes
here are measured, not assumed. If you want a worse Bluetooth link than the paper had,
ask for one with a `dropout` stage.

`simulate:` is what is wrong with *this device*, and every client sees it, because it is
part of what the device is. What a *client* does with the signal afterwards is a
separate thing, requested per subscription (below).

The config is validated completely before anything starts listening, so a typo fails
in the first second rather than forty minutes into a dry run.

## Built-in features

The Core processes a stream **once** and shares the result with every client that asked
for the same thing, so ten clients plotting a filtered EEG cost one filter, not ten.

Every figure below was generated by [`scripts/make_figures.py`](scripts/make_figures.py),
which starts a real Core, streams a real device through it, and plots what actually came
back off the socket. If a stage breaks, the figures break with it.

### Missing values

<p align="center"><img src="assets/missing_example.gif" alt="Missing-value handling" width="500"></p>

The flat stretches on the left are blinks. Nothing in the data says so; only `BPOGV`
does, which is why finding the gaps comes before filling them.

Once found, Thalamus distinguishes a gap from a real zero all the way through: it
arrives as `NA`, travels as JSON `null`, and is only turned into a number if you ask.

```python
client.subscribe("eye_tracker", pipeline=[{"stage": "missing_fill", "strategy": "zero"}])
```

`zero` (paper Fig. 2), `hold` (carry the last reading forward), `value` (a sentinel you
can detect), or `drop` (discard the sample). Each is a different lie; pick deliberately.

### Filters

<p align="center"><img src="assets/filter_example.gif" alt="Savitzky-Golay filtering" width="500"></p>

`savgol`, `kalman`, `moving_average`, `exponential`. All causal by default: they use
only the past, because a live stream has no future. Savitzky-Golay also offers a
`centered` mode, which smooths better at the cost of a stated `window // 2` lag.

```python
client.subscribe("eeg", pipeline=[{"stage": "savgol", "window": 11, "polyorder": 3}])
```

### Synchronization

<p align="center"><img src="assets/synchronisation_example.gif" alt="Stream synchronization" width="500"></p>

Ask for several devices aligned onto one timeline and you receive *frames* instead of
samples: one reading per device, taken at (nearly) the same instant.

```python
client.subscribe_synced(["eeg", "eye_tracker"], reference="eeg", tolerance_ms=10)
for frame in client.frames():
    frame["streams"]["eeg"]["EEG1"], frame["streams"]["eye_tracker"]["LPD"]
```

A stream with nothing close enough contributes `None`, not an interpolated guess. A
device that dies does not hold the other streams hostage.

### Noise

<p align="center"><img src="assets/noise_example.gif" alt="Noise injection" width="500"></p>

`gaussian_noise`, `uniform_noise`, `constant_noise` (a fixed offset, optionally
drifting). All seedable, because a study whose noise cannot be reproduced is a study
whose results cannot be reproduced.

### Delay and loss

<p align="center"><img src="assets/delay_example.gif" alt="Delay simulation" width="500"></p>

Three different failures, which break three different things:

| | what it does | what it breaks |
|---|---|---|
| `delay` (`mode: timestamp`) | the sample claims to be older than it is | synchronization logic |
| `delay` (`mode: buffer`) | delivery lags by N samples | real-time logic |
| `dropout` | the sample never arrives at all | anything that counts samples |
| `latency_ms` on a subscription | this one client's link is slow | only that client |

## Writing a device

Implement `samples()`. The socket, reconnection, timestamps, and drift-free pacing are
already written.

```python
from thalamus import RecordingDevice

class MyThermometer(RecordingDevice):
    def samples(self):
        while True:
            yield {"temperature": self.sdk.read()}

MyThermometer("thermometer", rate=10).run()
```

The paper notes the two things that make real devices awkward, and both are handled in
a controller you write once: a device with no UTC clock just omits the timestamp and
gets stamped on arrival; a device that only writes to a file gets tailed. See
[`examples/devices.py`](examples/devices.py).

For simulated devices you usually need no class at all:

```python
from thalamus import ReplayDevice, SyntheticDevice

ReplayDevice("eeg", "unicorn.csv", profile="unicorn_hybrid_black", loop=True).run()
SyntheticDevice("eye", profile="gp3").run()                        # no recording needed
SyntheticDevice("ecg", {"lead_ii": {"kind": "ecg", "heart_rate": 72}}, rate=128).run()
```

`ReplayDevice` takes `channels=[...]`, which answers the paper's "try before you buy"
question directly: replay a 62-channel SEED recording as if it were a 14-channel
headset, and find out whether 14 would have been enough, without buying either.

## Writing a client

```python
from thalamus import MISSING, ThalamusClient

with ThalamusClient() as client:
    client.subscribe("eye_tracker", channels=["LPD", "RPD", "BPOGV"], pipeline=[
        {"stage": "validity_mask", "profile": "gp3"},   # blinks -> gaps
        {"stage": "missing_fill", "strategy": "hold"},
        {"stage": "savgol", "window": 31, "polyorder": 3},
    ])
    for sample in client.stream():
        print(sample.timestamp, sample.data["LPD"])
```

A client can also push samples back in, which makes it a recording device too:

```python
client.send_sample("attention_estimate", {"arousal": 0.7})
client.send_event("stimulus_onset")     # a marker every client sees
```

See [`examples/clients.py`](examples/clients.py) for all four patterns.

## The protocol

You do not need Python. Open a TCP socket, send newline-terminated JSON.

```jsonc
// device -> Core (port 9000). Any key that isn't reserved is a channel.
{"device_id": "eeg", "timestamp": 1690535469479, "Fp1": 12.3, "Fp2": -4.1}

// client -> Core (port 9001)
{"type": "subscribe", "devices": [
    {"device_id": "eeg",
     "channels": ["Fp1"],
     "pipeline": [{"stage": "savgol", "window": 11}],
     "latency_ms": 50}
]}
{"type": "list_devices"}

// Core -> client
{"type": "welcome", "devices": [...], "stages": [...]}   // sent on connect
{"device_id": "eeg", "timestamp": 1690535469479, "Fp1": 12.3}
{"type": "device_disconnected", "device_id": "eeg"}      // a sensor just died
```

A gap is JSON `null`, never `0`. Timestamps are UTC milliseconds.

## Commands

| | |
|---|---|
| `thalamus demo` | a three-device study, no data files needed |
| `thalamus run study.yaml` | your study: the Core plus every device in it |
| `thalamus serve` | just the Core, for devices running elsewhere |
| `thalamus devices` | what is connected, and at what *measured* rate |
| `thalamus monitor <ids>` | print a live stream (`--sync`, `--filter`, `--validity`, ...) |
| `thalamus record <ids>` | write streams to CSV, plus an events file |
| `thalamus stages` | what processing is available |
| `thalamus profiles [name]` | the hardware it knows: real channels, rates, quirks |
| `thalamus make-data` | generate sample recordings to replay |

## Installing

```shell
pip install -e .              # the Core, devices, clients, and every dependency-free stage
pip install -e ".[filters]"   # + Savitzky-Golay (SciPy)
pip install -e ".[video]"     # + webcam replay (OpenCV, Pillow)
pip install -e ".[all]"
```

The Core has no scientific dependencies at all. It runs anywhere Python does, and
`kalman`, `moving_average`, `exponential`, and every noise, delay, and missing-value
stage are pure Python. Only Savitzky-Golay needs SciPy, and only video needs OpenCV.

## Coming from the pre-1.0 version?

The wire protocol has not changed, so **existing devices and clients keep working
unmodified**, including clients that send their subscription without a trailing
newline, which the original example did.

What changed:

| before | now |
|---|---|
| `python3 thalamus.py --device-port 9000` | `thalamus serve --device-port 9000` |
| subclass `RecordingDevice`, write the socket loop yourself | subclass it and implement `samples()` |
| `from device_interface import RecordingDevice` | `from thalamus import RecordingDevice` (the old import still works, with a warning) |
| `run_dev_*.py` | [`examples/devices.py`](examples/devices.py), or a line of `study.yaml` |

The features the paper describes (filters, noise, delay, synchronization, missing
values) were not implemented in the original release. They are now, and are covered
by the test suite.

## Development

```shell
pip install -e ".[dev,filters]"
pytest                  # 122 tests, including end-to-end over real sockets
ruff check . && ruff format --check .
```

The test suite covers each layer in isolation and then starts a real Core on real
ports and drives real devices and clients through it, because in a networked toolkit
the bugs live in the seams.

## Citation

```bibtex
@inproceedings{latifzadeh2025thalamus,
  title     = {Thalamus: A User Simulation Toolkit for Prototyping Multimodal Sensing Studies},
  author    = {Latifzadeh, Kayhan and Leiva, Luis A.},
  booktitle = {Adjunct Proceedings of the 33rd ACM Conference on User Modeling,
               Adaptation and Personalization (UMAP Adjunct '25)},
  year      = {2025},
  doi       = {10.1145/3708319.3733687},
}
```

## Acknowledgments

Supported by the Horizon 2020 FET program of the European Union through the ERA-NET
Cofund funding (BANANA, grant CHIST-ERA-20-BCI-001) and Horizon Europe's European
Innovation Council through the Pathfinder program (SYMBIOTIK, grant 101071147).

## License

MIT.
