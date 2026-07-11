# Recordings

You do not need any of this to run Thalamus. `thalamus demo` generates its signals
from nothing, and [`examples/data/`](../examples/data) holds small generated CSVs that
[`examples/study.yaml`](../examples/study.yaml) replays out of the box. Regenerate them
at any time with:

```shell
thalamus make-data --out examples/data
```

Those generated files have the **same columns as the real recordings below**, because
both are described by the same device profile. That is the point: swap one for the
other and no client code changes.

## The real recordings from the paper

The EEG, eye-tracking, and webcam recordings used in the paper's figures are here:

https://drive.google.com/drive/folders/1bSI1wsgmD8lBhbxTuVIhsLqM6ISn3P4s

Download them into this directory and point a device at them. Name the hardware with
`profile:` and Thalamus knows the units, the nominal rate, and the validity flag. It
will also tell you if the file is missing a column it should have:

```yaml
devices:
  - id: eeg
    type: replay
    path: ../pre_recorded_files/eeg.csv
    profile: unicorn_hybrid_black
```

### g.tec Unicorn Hybrid Black: EEG, 250 Hz

Exactly 4 ms between samples. Seventeen columns, of which **only the first eight are
brain**:

| column | unit | meaning |
|---|---|---|
| `EEG1`-`EEG8` | µV | scalp potential at Fz, C3, Cz, C4, Pz, PO7, Oz, PO8 |
| `AccelerometerX/Y/Z` | g | head acceleration. `Y` sits near 1 g: that is gravity, the head is upright |
| `GyroscopeX/Y/Z` | °/s | head rotation rate |
| `BatteryLevel` | % | falls through the session |
| `Counter` | (none) | increments by 1 per sample. **A jump means the Bluetooth link dropped packets** |
| `ValidationIndicator` | flag | `1` when the sample is valid |

```csv
timestamp,EEG1,EEG2,...,BatteryLevel,Counter,ValidationIndicator
1690535476895,-0.6228964,4.2587385,...,93.333,206206,1
1690535476899,-3.2294390,0.8543182,...,93.333,206207,1
```

`Counter` is the one channel that can *prove* a packet was lost: a dropped sample leaves
a hole in the count, whereas a dropped *value* does not. Anything that infers a sampling
rate by counting rows will get it wrong; the counter says exactly what went missing.

Measured over the full recording (382,988 samples, 25.5 min): **exactly 4 ms between
every pair of samples** (not one interval differs), so 250.00 Hz, and the `Counter`
runs 206206 → 589193 unbroken. **This device lost nothing and flagged nothing**, which
is why its profile simulates no packet loss by default. `BatteryLevel` went 93.333 →
86.667, and those are the only two values in the file: the gauge reports in fifteenths,
so the battery is a step function, not a slope. EEG sits at σ ≈ 15 µV with excursions to
±450 µV (blinks, movement, cable).

### Gazepoint GP3 HD: eye tracking, 150 Hz

Samples every 6-7 ms, and genuinely jittery: the intervals are not constant, which is
why replaying the file's own timestamps (no `rate:`) is more honest than imposing a
perfect 150 Hz.

| column | unit | meaning |
|---|---|---|
| `BPOGX`, `BPOGY` | norm | best point of gaze. **Normalized to the screen, 0..1, origin top-left. Not pixels.** Multiply by your resolution |
| `BPOGV` | flag | `1` when the gaze point is valid; `0` during blinks and tracking loss |
| `LPD`, `RPD` | px | pupil diameter per eye, in *camera pixels*, comparable within a session only, since it changes with head distance |

```csv
timestamp,BPOGX,BPOGY,BPOGV,LPD,RPD
1690535469479,0.58224,0.37114,1,16.61074,15.31291
1690535469485,0.57659,0.38235,1,16.46606,15.16937
```

Measured over the full recording (230,974 samples, 25.8 min): 149.3 Hz; `BPOGX` mean
0.489 (sd 0.142); `BPOGY` mean 0.385 (sd 0.294); `LPD` mean 17.41 px (sd 2.19); `RPD`
mean 16.99 px (sd 2.25).

**0..1 is where the screen is, not where the data is.** 13% of *valid* `BPOGY` values
fall outside 0..1, as far as -1.38 and +2.38, because the participant looks past the
monitor and the tracker keeps extrapolating. Code that does `int(BPOGY * screen_height)`
will throw. Better to find that out now.

**Put `validity_mask` first when you replay this.** The tracker does not blank anything
during a blink. It *freezes*. Of the 118 blinks in this file, 115 of the 116
multi-sample ones have every column identical throughout, and 116 of the 118 onsets
repeat the preceding valid row exactly. A blink is 131 ms (median) of perfectly
plausible, perfectly unchanging numbers, and only `BPOGV` says otherwise:

```yaml
simulate:
  - stage: validity_mask     # BPOGV=0 -> BPOGX/BPOGY/LPD/RPD become real gaps
```

### Logitech C505e: webcam, 720p, 30 fps

One `frame` per sample, base64-encoded JPEG. Needs the video extra:
`pip install thalamus[video]`.

## Using your own

Any CSV, TSV, or JSONL file works. One row per sample, one column per channel:

```csv
timestamp,Fp1,Fp2,Cz
1690535469479,12.3,-4.1,0.8
1690535469483,11.9,NA,1.1
```

A column named `timestamp` (or `time`, `ts`, `unix_ts`, `epoch_ms`) is taken as the
sample's time in UTC milliseconds, and is what lets Thalamus align this recording
against the others. Without one, set `rate:` on the device and samples are stamped as
they are emitted.

`NA` (or an empty cell, or `null`, or `NaN`) is a *gap*, and stays distinguishable
from a real `0` all the way to the client.

To teach Thalamus about hardware it does not know, add a profile. It is one dataclass,
see [`thalamus/devices/profiles.py`](../thalamus/devices/profiles.py).

## Public datasets

The paper's "try before you buy" scenario (§3.2) replays public EEG datasets to decide
what hardware to buy. Point `ReplayDevice` at them once they are in CSV form, and use
`channels:` to pretend a 62-channel recording is a 14-channel headset:

- **DREAMER**: 14-channel EEG + ECG, emotion. Katsigiannis & Ramzan, 2017.
- **MAHNOB-HCI**: EEG, eye tracking, video. Soleymani et al., 2011.
- **SEED**: 62-channel EEG, emotion. Zheng & Lu, 2015.
