# Recordings

You do not need any of this to run Thalamus. `thalamus demo` generates its signals
from nothing, and [`examples/data/`](../examples/data) holds small generated CSVs that
[`examples/study.yaml`](../examples/study.yaml) replays out of the box. Regenerate them
at any time with:

```shell
thalamus make-data --out examples/data
```

## The real recordings from the paper

The EEG, eye-tracking, and webcam recordings used in the paper's figures are here:

https://drive.google.com/drive/folders/1bSI1wsgmD8lBhbxTuVIhsLqM6ISn3P4s

Download them into this directory and point a device at them:

```yaml
devices:
  - id: eeg
    type: replay
    path: ../pre_recorded_files/eeg.csv
```

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

`NA` — or an empty cell, or `null`, or `NaN` — is a *gap*, and stays distinguishable
from a real `0` all the way to the client.

## Public datasets

The paper's "try before you buy" scenario (§3.2) replays public EEG datasets to decide
what hardware to buy. Point `ReplayDevice` at them once they are in CSV form, and use
`channels:` to pretend a 62-channel recording is a 14-channel headset:

- **DREAMER** — 14-channel EEG + ECG, emotion. Katsigiannis & Ramzan, 2017.
- **MAHNOB-HCI** — EEG, eye tracking, video. Soleymani et al., 2011.
- **SEED** — 62-channel EEG, emotion. Zheng & Lu, 2015.
