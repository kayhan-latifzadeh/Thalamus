<p align="center">
  <img src="assets/logo.jpg" alt="Thalamus" width="200">
  <br>
  <b>A Multimodal Sensing and Simulation Toolkit</b>
  <br>
  <sub>Prototype a physiological study before you buy the hardware, book the lab, or pay a participant.</sub>
</p>

<p align="center">
  <a href="https://doi.org/10.1145/3708319.3733687"><img src="https://img.shields.io/badge/paper-UMAP%20'25-blue" alt="paper"></a>
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="license">
</p>

---

Multimodal studies with EEG, eye tracking, and physiological sensors are costly to run,
and a substantial share of the problems that arise are in the software rather than the
hardware. Thalamus supports prototyping that software beforehand. It streams real or
simulated signals from any number of devices, synchronizes them on UTC timestamps, and
can introduce the conditions a real session produces: dropped packets, missing values,
sensor noise, and network delay. Recording and analysis code can therefore be tested
against those conditions in a dry run rather than during data collection.

## Quick start

```shell
pip install -e .
thalamus demo
```

This starts a complete three-device study with no data files and no hardware: an EEG cap
at 250 Hz on a lossy link, an eye tracker at 150 Hz that blinks, and an ECG. The first
two model specific commercial devices, but any device can be described the same way; see
[device profiles](#device-profiles). In another terminal:

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

The rate column reports the rate each device is measured to be achieving, not the rate it
declares. A device reporting `190/250 Hz!` is delivering samples more slowly than it
claims, which is a condition worth detecting before data collection rather than after.

The channel names (`BPOGX`, `EEG1`) are those the corresponding hardware actually writes,
rather than placeholders such as `gaze_x`. Analysis code must read the former, so the
simulation emits them. The same applies to any device you define.

## Device profiles

If a simulation emits channels named `pupil` and `gaze_x`, any code written against it
must be rewritten once the hardware arrives, since the device will use different names.
A profile avoids this: describe the device once, in the study file, and the simulation
emits the columns the device actually writes.

```yaml
profiles:
  my_tracker:                    # defined here; no Python required
    rate: 600
    validity_flag: validity      # the column that says "this sample is bad"
    channels:
      gaze_x:   {unit: px, signal: {kind: random_walk, step: 3, start: 960}}
      gaze_y:   {unit: px, signal: {kind: random_walk, step: 3, start: 540}}
      validity: {unit: flag, digits: 0, signal: {kind: constant, value: 1}}

devices:
  - id: eye
    type: synthetic              # nothing plugged in
    profile: my_tracker
```

With the profile in place, `type: synthetic` generates the data and `type: replay` reads
a recording, and **the client code is identical in both cases**. Analysis can therefore
be written before the hardware is available and run unchanged afterwards.

A profile may be as small as `channels: [force, torque]` together with a rate, or it may
additionally specify units, value ranges, a validity flag, and the device's characteristic
failure modes. Only `rate` and `channels` are required.

### Built-in profiles

Three profiles are included as worked examples. They have no special status: a profile
defined in a study file has the same capabilities.

| profile | device | rate | channels |
|---|---|---|---|
| `unicorn_hybrid_black` | g.tec Unicorn Hybrid Black | 250 Hz | `EEG1`-`EEG8`, IMU, `BatteryLevel`, `Counter`, `ValidationIndicator` |
| `gp3` | Gazepoint GP3 HD | 150 Hz | `BPOGX`, `BPOGY`, `BPOGV`, `LPD`, `RPD` |
| `c505e` | Logitech C505e | 30 fps | `frame` |

```shell
thalamus profiles              # what's defined
thalamus profiles gp3          # every column, its unit, and what it means
```

Their parameters are measured rather than estimated: channel names, sampling rates, value
ranges, and failure modes are all derived from 26-minute recordings of the devices, and
[the test suite checks them against those recordings](tests/test_profiles.py). They are
useful as references when defining a profile of your own. The following section describes
one such measured property.

### Validity flags

Many sensors provide a channel indicating whether a sample is valid: an eye tracker flags
blinks, an amplifier flags corrupt packets. Declaring it with `validity_flag:` in a
profile allows Thalamus to act on it.

Such sensors do not generally stop emitting samples when the measurement fails, which is
the difficulty. A Gazepoint GP3 illustrates this. In a 26-minute recording (230,974
samples, 118 blinks):

- **115 of the 116 multi-sample blinks have every column identical throughout**, and
- **116 of the 118 blink onsets repeat the preceding valid row exactly.**

The tracker does not blank the pupil during a blink. It holds the last valid reading for
the duration of the blink (median 131 ms), and only the validity flag distinguishes those
rows from genuine measurements of a stationary eye. If they are included in a pupil
baseline, no downstream check can identify them. In this recording, ignoring the flag
shifts the mean pupil diameter by 0.44%. Other devices exhibit comparable behaviour, and
it should not be assumed absent without checking.

Thalamus can therefore reproduce this failure mode rather than an idealized one, using
`missing_inject` with `mode: hold`. The `validity_mask` stage performs the corresponding
correction: it reads the flag and blanks the channels the flag does not vouch for,
converting an invalid reading in the recording into an explicit gap on the wire.

```python
client.subscribe("eye_tracker", pipeline=[
    {"stage": "validity_mask", "profile": "gp3"},   # BPOGV=0 -> BPOGX/BPOGY/LPD/RPD are gaps
    {"stage": "missing_fill", "strategy": "hold"},  # ...now bridge them
    {"stage": "savgol", "window": 31, "polyorder": 3},
])
```

It should generally be the first stage applied when replaying real hardware. The left
panel of the figure below shows the result of omitting it.

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
    path: data/eeg.csv           # no `rate:`, so the file's own timing drives the
    profile: unicorn_hybrid_black   # replay, reproducing its jitter rather than an
    loop: true                      # idealized constant rate
    simulate:
      - stage: constant_noise    # electrode drift as the gel dries
        offset: 0.5
        drift: 0.0002
        channels: [EEG1, EEG2, EEG3, EEG4, EEG5, EEG6, EEG7, EEG8]   # EEG channels only

  - id: eye_tracker
    type: synthetic              # generated; no recording required
    profile: gp3                 # but with the GP3's real column names
    seed: 2                      # a seed makes the run reproducible
    simulate:
      - stage: missing_inject    # blinks: one per ~13 s, median 131 ms (measured)
        mode: hold               # the tracker holds the last value; it does not blank
        probability: 0.0005
        burst: [6, 34]
        channels: [BPOGX, BPOGY, LPD, RPD]
        flag: BPOGV              # the tracker reports BPOGV=0 rather than stopping
```

A device that names a profile and does not specify `simulate:` inherits the failure modes
declared by that profile, so `type: synthetic` with `profile: gp3` produces the blinks
shown above without further configuration. `thalamus run` prints the stages applied to
each device at startup, and `simulate: []` disables them.

The Unicorn profile declares no failure modes, because the reference recording contains
no dropped packets and no invalid samples. Failure modes in the built-in profiles are
measured rather than assumed. A less reliable link can be simulated by adding a `dropout`
stage explicitly.

`simulate:` describes properties of the device itself, so every client observes them.
Processing that a particular client wants applied to the signal is a separate matter,
requested per subscription (see below).

The configuration is validated in full before any port is opened, so an error in the file
is reported immediately rather than partway through a run.

## Built-in features

The Core computes each distinct pipeline **once** and fans the result out to every client
that requested it, so ten clients displaying the same filtered EEG incur the cost of one
filter rather than ten.

The figures below were generated by [`scripts/make_figures.py`](scripts/make_figures.py),
which starts a Core, streams a device through it, and plots the data returned over the
socket. They therefore reflect the behaviour of the current implementation.

### Missing values

<p align="center"><img src="assets/missing_example.gif" alt="Missing-value handling" width="500"></p>

The flat segments in the left panel are blinks. They are not identifiable from the data
itself; only the `BPOGV` flag distinguishes them. Detecting gaps therefore precedes
filling them.

Once detected, Thalamus preserves the distinction between a gap and a measured zero
throughout: a gap arrives as `NA`, is transmitted as JSON `null`, and is converted to a
number only when requested.

```python
client.subscribe("eye_tracker", pipeline=[{"stage": "missing_fill", "strategy": "zero"}])
```

The available strategies are `zero`, `hold` (carry the last valid reading forward),
`value` (substitute a detectable sentinel), and `drop` (discard the sample). Each
introduces a different distortion, and the appropriate choice depends on the measure.

### Filters

<p align="center"><img src="assets/filter_example.gif" alt="Savitzky-Golay filtering" width="500"></p>

The available filters are `savgol`, `kalman`, `moving_average`, and `exponential`. All
are causal by default, using only past samples, since a live stream provides no lookahead.
Savitzky-Golay additionally supports a `centered` mode, which improves smoothing at the
cost of a `window // 2` lag.

```python
client.subscribe("eeg", pipeline=[{"stage": "savgol", "window": 11, "polyorder": 3}])
```

### Synchronization

<p align="center"><img src="assets/synchronisation_example.gif" alt="Stream synchronization" width="500"></p>

Requesting several devices aligned onto a single timeline yields *frames* rather than
samples: one reading per device, taken at approximately the same instant.

```python
client.subscribe_synced(["eeg", "eye_tracker"], reference="eeg", tolerance_ms=10)
for frame in client.frames():
    frame["streams"]["eeg"]["EEG1"], frame["streams"]["eye_tracker"]["LPD"]
```

A stream with no sample within the tolerance contributes `None` rather than an
interpolated estimate. A device that stops responding does not block the remaining
streams.

### Noise

<p align="center"><img src="assets/noise_example.gif" alt="Noise injection" width="500"></p>

The available noise sources are `gaussian_noise`, `uniform_noise`, and `constant_noise`
(a fixed offset, optionally drifting). All accept a seed, so that a run can be reproduced
exactly.

### Delay and loss

<p align="center"><img src="assets/delay_example.gif" alt="Delay simulation" width="500"></p>

These model distinct failures, which affect different parts of a system:

| stage | effect | primarily affects |
|---|---|---|
| `delay` (`mode: timestamp`) | the sample is reported as older than it is | synchronization |
| `delay` (`mode: buffer`) | delivery is held back by N samples | real-time processing |
| `dropout` | the sample is never delivered | anything that counts samples |
| `latency_ms` on a subscription | one client's link is slow | that client only |

## Connecting your own device

A device requires a single method: **`samples()`, a generator yielding one dict per
reading.** Subclassing `RecordingDevice` provides the socket handling, reconnection,
timestamping, and drift-free pacing.

```python
from thalamus import RecordingDevice

class MyThermometer(RecordingDevice):
    def samples(self):
        while True:
            yield {"temperature": self.sdk.read()}   # keys become channels

MyThermometer("thermometer", rate=10).run()
```

That constitutes the entire interface. The behaviour of each yielded value:

| yielded value | resulting behaviour |
|---|---|
| `{"temp": 21.5}` | stamped with the current UTC time on arrival |
| `{"timestamp": 1690535469479, "temp": 21.5}` | the supplied timestamp is used, which is preferable when the device has its own clock |
| `{"temp": None}` or `{"temp": "NA"}` | recorded as a gap, and remains distinguishable from a measured `0` |
| nothing (generator returns) | the stream ends and clients are notified of the disconnection |

Setting `rate=` causes Thalamus to pace the generator, scheduling against an absolute
origin so that a device declared at 250 Hz does not gradually drift below it. If the
device produces samples at its own pace (for instance, blocking on a driver or tailing a
file), leave `rate` unset and `samples()` will be drained as quickly as it yields.

Two common cases are handled by the controller rather than by device code: a device with
no clock of its own omits `timestamp` and is stamped on arrival, and a device that only
writes to a file is tailed. Both are demonstrated in
[`examples/devices.py`](examples/devices.py).

Python is not required. A device is anything that opens a TCP socket and writes one JSON
object per line, which can be implemented in roughly twenty lines of C++,
MATLAB, or JavaScript will do. See [the protocol](#the-protocol).

### Simulating a device instead

If the hardware is not yet available, no subclass is generally required:

```python
from thalamus import ReplayDevice, SyntheticDevice

ReplayDevice("eeg", "session.csv", profile="my_amp", loop=True).run()  # from a recording
SyntheticDevice("eye", profile="my_tracker").run()                     # from nothing
SyntheticDevice("ecg", {"lead_ii": {"kind": "ecg", "heart_rate": 72}}, rate=128).run()
```

`ReplayDevice` also accepts `channels=[...]`, which allows a 62-channel recording to be
replayed as though it came from a 14-channel headset. This makes it possible to assess
whether a lower-specification device would be sufficient before purchasing either.

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

A client may also send samples back to the Core, which makes it a recording device in its
own right:

```python
client.send_sample("attention_estimate", {"arousal": 0.7})
client.send_event("stimulus_onset")     # a marker every client sees
```

See [`examples/clients.py`](examples/clients.py) for all four patterns.

## The protocol

Python is not required. A client or device opens a TCP socket and exchanges
newline-terminated JSON.

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
{"type": "device_disconnected", "device_id": "eeg"}      // the device stopped responding
```

A gap is transmitted as JSON `null`, never as `0`. Timestamps are UTC milliseconds.

## Commands

| | |
|---|---|
| `thalamus demo` | a three-device study requiring no data files |
| `thalamus run study.yaml` | a study: the Core plus every device it defines |
| `thalamus serve` | the Core alone, for devices running elsewhere |
| `thalamus devices` | connected devices and their *measured* rates |
| `thalamus monitor <ids>` | print a live stream (`--sync`, `--filter`, `--validity`, ...) |
| `thalamus record <ids>` | write streams to CSV, together with an events file |
| `thalamus stages` | list the available processing stages |
| `thalamus profiles [name]` | list the defined device profiles |
| `thalamus make-data` | generate sample recordings for replay |

## Installing

```shell
pip install -e .              # the Core, devices, clients, and every dependency-free stage
pip install -e ".[filters]"   # + Savitzky-Golay (SciPy)
pip install -e ".[video]"     # + webcam replay (OpenCV, Pillow)
pip install -e ".[all]"
```

The Core has no scientific dependencies and runs on any Python installation. The
`kalman`, `moving_average`, and `exponential` filters, together with all noise, delay,
and missing-value stages, are implemented in pure Python. Only Savitzky-Golay requires
SciPy, and only video replay requires OpenCV.

## Coming from the pre-1.0 version?

The wire protocol is unchanged, so **existing devices and clients continue to work
without modification**, including clients that send a subscription without a trailing
newline, as the original example did.

What changed:

| before | now |
|---|---|
| `python3 thalamus.py --device-port 9000` | `thalamus serve --device-port 9000` |
| subclass `RecordingDevice` and implement the socket loop | subclass it and implement `samples()` |
| `from device_interface import RecordingDevice` | `from thalamus import RecordingDevice` (the old import still works, with a warning) |
| `run_dev_*.py` | [`examples/devices.py`](examples/devices.py), or a line of `study.yaml` |

Filters, noise, delay, synchronization, and missing-value handling were documented but
not implemented in the original release. They are implemented here, and are covered by
the test suite.

## Development

```shell
pip install -e ".[dev,filters]"
pytest                  # 171 tests, including end-to-end tests over real sockets
ruff check . && ruff format --check .
```

The test suite covers each layer in isolation, and additionally starts a Core on real
ports and drives devices and clients through it over TCP, since defects in a networked
system tend to arise at the interfaces between components.

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
