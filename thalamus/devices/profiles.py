"""The hardware from the paper, described once.

A profile is what a device *actually emits*: its real column names, its real
sampling rate, the units, and the quirks. It exists because the alternative — which
is what this toolkit shipped with — is inventing plausible-looking column names like
``gaze_x`` and ``ch_Fp1``, and then discovering on the day of the study that the
Gazepoint writes ``BPOGX`` and the Unicorn writes ``EEG1``, and that every line of
analysis code written against the simulation has to be rewritten against the file.

That is the failure the paper is trying to prevent, so the simulation had better
not cause it. With a profile, the synthetic device and the real recording produce
the *same columns*, and code written against one runs unchanged against the other::

    - id: eeg
      type: synthetic            # no hardware in the room
      profile: unicorn_hybrid_black

    - id: eeg
      type: replay               # the real recording, same columns
      path: unicorn.csv
      profile: unicorn_hybrid_black

The three profiles here are the three devices used in the paper. The channel names,
rates, and value ranges are taken from the recordings themselves (see
``pre_recorded_files/README.md``), not from a datasheet.

Adding your own is a dataclass::

    register_profile(DeviceProfile(
        key="my_tracker", vendor="...", model="...", modality="eye",
        rate=60, channels=(Channel("x", unit="px"), ...),
    ))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "Channel",
    "DeviceProfile",
    "available_profiles",
    "get_profile",
    "register_profile",
]


@dataclass(frozen=True)
class Channel:
    """One column of a device's output.

    ``signal`` is the recipe a :class:`~thalamus.devices.synthetic.SyntheticDevice`
    uses to fake this channel. It is tuned to the range and shape of the real
    recording — an EEG channel that swings +-30 uV, a pupil diameter that hovers
    around 16 px — so that a plot of the simulation and a plot of the recording have
    the same axes. It is *not* a model of the underlying physiology, and nothing here
    should be mistaken for one.
    """

    name: str
    unit: str = ""
    about: str = ""
    signal: Optional[Dict[str, Any]] = None
    #: Decimal places to round to when writing this channel to a file. ``0`` means
    #: the channel is an integer (a flag, a counter) and is written as one.
    digits: int = 4


@dataclass(frozen=True)
class DeviceProfile:
    """A real device: what it is, how fast it runs, and what it puts in each column."""

    key: str
    vendor: str
    model: str
    modality: str
    rate: float
    channels: Tuple[Channel, ...]

    #: The channel carrying the device's own "is this sample any good?" flag, if it
    #: has one. Both devices in the paper do, and it is the single most important
    #: thing a profile knows: see :class:`~thalamus.processing.missing.ValidityMaskStage`.
    validity_flag: Optional[str] = None
    #: What that flag reads when the sample *is* good.
    validity_ok: Any = 1
    #: The channels the flag vouches for. Empty means "every channel except the flag".
    validity_covers: Tuple[str, ...] = ()

    #: The imperfections this hardware has in the field, as a ``simulate:`` pipeline.
    #: A study config that names a profile and gives no ``simulate:`` of its own gets
    #: these — and ``thalamus run`` prints them on startup, so it is never a surprise.
    simulate: Tuple[Dict[str, Any], ...] = ()

    notes: str = ""
    aliases: Tuple[str, ...] = field(default=())

    # -- what the device emits -------------------------------------------------

    @property
    def channel_names(self) -> List[str]:
        return [c.name for c in self.channels]

    def channel(self, name: str) -> Optional[Channel]:
        for c in self.channels:
            if c.name == name:
                return c
        return None

    def signals(self) -> Dict[str, Dict[str, Any]]:
        """The synthetic recipe for every channel, for :class:`SyntheticDevice`."""
        return {
            c.name: dict(c.signal, digits=c.digits) for c in self.channels if c.signal is not None
        }

    def covered_by_flag(self) -> List[str]:
        """Which channels the validity flag speaks for."""
        if not self.validity_flag:
            return []
        if self.validity_covers:
            return list(self.validity_covers)
        return [c.name for c in self.channels if c.name != self.validity_flag]

    # -- checking a recording against the profile -------------------------------

    def missing_from(self, columns: Sequence[str]) -> List[str]:
        """Channels this device should have produced that the file does not contain.

        Worth saying out loud rather than discovering downstream: a GP3 export with no
        ``BPOGV`` column cannot tell you which samples were blinks, and no amount of
        processing will recover that.
        """
        present = set(columns)
        return [c.name for c in self.channels if c.name not in present]

    def describe(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "profile": self.key,
            "vendor": self.vendor,
            "model": self.model,
            "modality": self.modality,
            "declared_rate": self.rate,
            "channels": self.channel_names,
            "units": {c.name: c.unit for c in self.channels if c.unit},
        }
        if self.validity_flag:
            info["validity_flag"] = self.validity_flag
        return info


# --------------------------------------------------------------------------- #
# the registry
# --------------------------------------------------------------------------- #

_PROFILES: Dict[str, DeviceProfile] = {}


def register_profile(profile: DeviceProfile) -> DeviceProfile:
    """Add a profile. Yours are as first-class as the built-in ones."""
    _PROFILES[profile.key] = profile
    for alias in profile.aliases:
        _PROFILES[alias] = profile
    return profile


def get_profile(key: str) -> DeviceProfile:
    try:
        return _PROFILES[key]
    except KeyError:
        raise ValueError(
            f"unknown device profile {key!r}; available: {', '.join(available_profiles())}"
        ) from None


def available_profiles() -> List[str]:
    """The canonical keys, without the aliases."""
    return sorted({p.key for p in _PROFILES.values()})


def all_profiles() -> List[DeviceProfile]:
    return [_PROFILES[k] for k in available_profiles()]


# --------------------------------------------------------------------------- #
# Gazepoint GP3 HD — eye tracker
# --------------------------------------------------------------------------- #
#
# Every number below is measured off the paper's own 26-minute recording
# (230,974 samples). Where the datasheet and the recording disagree, the recording
# wins.
#
#   rate       149.3 Hz measured; intervals 1-24 ms, median 7. Genuinely jittery,
#              which is why replaying the file's own timestamps (no `rate:`) is more
#              honest than imposing a perfect 150 Hz.
#   BPOGX      mean 0.489, sd 0.142
#   BPOGY      mean 0.385, sd 0.294
#   LPD        mean 17.41, sd 2.19   (1st-99th percentile 13.1-22.7)
#   RPD        mean 16.99, sd 2.25
#   BPOGV=0    1.28% of samples, in 118 runs; median 19.5 samples (131 ms)
#
# The gaze columns are the Gazepoint API's "best point of gaze" — the fused binocular
# estimate — in *normalized screen coordinates*: 0..1 across the display, origin
# top-left. Not pixels.
#
# But 0..1 is where the screen is, NOT where the data is. In the recording, 13% of
# valid BPOGY values fall outside 0..1, ranging as far as -1.38 and +2.38, because the
# participant looks past the monitor and the tracker keeps extrapolating. So the
# generators below are deliberately NOT clamped to the screen: a simulation that only
# ever emits 0..1 will happily pass code that indexes a screen-sized array with
# `int(BPOGY * height)`, and that code will throw on the very first real recording.

register_profile(
    DeviceProfile(
        key="gp3",
        aliases=("gazepoint_gp3", "gp3_hd"),
        vendor="Gazepoint",
        model="GP3 HD",
        modality="eye",
        rate=150,
        validity_flag="BPOGV",
        validity_ok=1,
        validity_covers=("BPOGX", "BPOGY", "LPD", "RPD"),
        channels=(
            Channel(
                "BPOGX",
                unit="norm",
                about="gaze x: 0 = left edge of screen, 1 = right. Can fall outside 0..1",
                digits=5,
                # Mean-reverting, with sd = step / sqrt(2 * pull) ~= 0.142, to match.
                signal={
                    "kind": "random_walk",
                    "step": 0.009,
                    "start": 0.489,
                    "pull": 0.002,
                    "minimum": -1.0,  # far outside the screen: a backstop, not a screen edge
                    "maximum": 2.0,
                },
            ),
            Channel(
                "BPOGY",
                unit="norm",
                about="gaze y: 0 = top edge of screen, 1 = bottom. Often outside 0..1",
                digits=5,
                signal={
                    "kind": "random_walk",
                    "step": 0.0186,
                    "start": 0.385,
                    "pull": 0.002,
                    "minimum": -1.5,
                    "maximum": 2.5,
                },
            ),
            Channel(
                "BPOGV",
                unit="flag",
                about="1 when the gaze point is valid; 0 during blinks and tracking loss",
                digits=0,
                signal={"kind": "constant", "value": 1},
            ),
            Channel(
                "LPD",
                unit="px",
                about="left pupil diameter, in camera pixels (not mm)",
                digits=5,
                # Pupil dilates and constricts slowly: a weak pull gives a wandering
                # baseline (sd ~2.2 px) rather than a tidy sinusoid.
                signal={
                    "kind": "random_walk",
                    "step": 0.069,
                    "start": 17.41,
                    "pull": 0.0005,
                    "minimum": 5.0,
                    "maximum": 32.0,
                    "noise": 0.1,
                },
            ),
            Channel(
                "RPD",
                unit="px",
                about="right pupil diameter, in camera pixels (not mm)",
                digits=5,
                signal={
                    "kind": "random_walk",
                    "step": 0.071,
                    "start": 16.99,
                    "pull": 0.0005,
                    "minimum": 5.0,
                    "maximum": 32.0,
                    "noise": 0.1,
                },
            ),
        ),
        # Blinks, reproduced from the recording rather than imagined.
        #
        # `mode: hold` is the whole point. The GP3 does not blank anything during a
        # blink -- it FREEZES. In the recording, 115 of the 116 multi-sample invalid
        # runs have every column identical throughout, and 116 of the 118 onsets repeat
        # the last valid sample exactly. So a blink arrives as ~131 ms of entirely
        # plausible, entirely unchanging numbers, with a 0 in BPOGV that nothing forces
        # you to read. Simulating it as a nice obvious gap would be simulating a kinder
        # device than the one you own.
        #
        # One blink per ~1957 samples (p = 0.0005), median 19.5 samples long.
        simulate=(
            {
                "stage": "missing_inject",
                "mode": "hold",
                "probability": 0.0005,
                "burst": [6, 34],
                "channels": ["BPOGX", "BPOGY", "LPD", "RPD"],
                "flag": "BPOGV",
                "seed": 7,
            },
        ),
        notes=(
            "Gaze is normalized to the screen (0..1), not in pixels -- and it leaves "
            "that range often: 13% of valid BPOGY values in the paper's recording are "
            "outside 0..1. Pupil diameter is in camera pixels, comparable within a "
            "session only. BPOGV=0 marks a blink, and during one the tracker FREEZES "
            "the last valid gaze and pupil rather than blanking them, so those rows "
            "look like real data. Always run validity_mask."
        ),
    )
)


# --------------------------------------------------------------------------- #
# g.tec Unicorn Hybrid Black — 8-channel EEG
# --------------------------------------------------------------------------- #
#
# Measured off the paper's 26-minute recording (382,988 samples):
#
#   rate       exactly 4 ms between every pair of samples. Not a single interval
#              differs. 250.00 Hz, rock steady.
#   loss       zero. Counter runs 206206 -> 589193 with no jump, and there are exactly
#              that many rows. ValidationIndicator is 1 for every sample.
#   EEG        sd ~15 uV, mean ~0, excursions to +-450 uV (blinks, movement, cable)
#   IMU        AccelY sits at 0.95 g -- gravity, the head is upright
#   battery    93.333 -> 86.667 over 26 min. Only TWO distinct values in the whole
#              file: the gauge reports in fifteenths (100/15 = 6.667).
#
# Note what is NOT here: this device did not drop a single packet. An earlier version
# of this profile gave it Bluetooth dropouts by default, which was a failure mode
# invented from a datasheet rather than observed in the data. It has been removed. If
# your link is worse than this one's, add a `dropout` stage yourself -- and the Counter
# column is how you will prove it.
#
# Seventeen columns, and only the first eight are EEG. The rest are what makes this
# device interesting to simulate: a 6-axis IMU (head movement, the biggest source of
# EEG artefact you will meet), a battery that runs down, a sample counter, and a
# validity flag. Code that assumes "every column is a channel of brain data" breaks
# on this device, and it is better to find that out here.

_EEG_SITES = ("Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8")

register_profile(
    DeviceProfile(
        key="unicorn_hybrid_black",
        aliases=("unicorn", "unicorn_black"),
        vendor="g.tec",
        model="Unicorn Hybrid Black",
        modality="eeg",
        rate=250,
        validity_flag="ValidationIndicator",
        validity_ok=1,
        validity_covers=tuple(f"EEG{i}" for i in range(1, 9)),
        channels=(
            *(
                Channel(
                    f"EEG{i}",
                    unit="uV",
                    about=f"scalp potential at {site} (10-20 system)",
                    digits=6,
                    # Each channel gets its own alpha phase, so the eight traces are
                    # correlated in character but not identical -- as a real cap's are.
                    signal={
                        "kind": "eeg",
                        "amplitude": 18.0,
                        "alpha_freq": 10.0,
                        "phase": i * 0.4,
                    },
                )
                for i, site in enumerate(_EEG_SITES, start=1)
            ),
            # IMU parameters below are set so the simulated mean and sd match the
            # recording's, channel by channel.
            Channel(
                "AccelerometerX",
                unit="g",
                about="head acceleration, x",
                digits=3,
                signal={
                    "kind": "sine",
                    "freq": 0.07,
                    "amplitude": 0.008,
                    "offset": 0.007,
                    "noise": 0.006,
                },
            ),
            Channel(
                "AccelerometerY",
                unit="g",
                about="head acceleration, y (~1 g: this is gravity, the head is upright)",
                digits=3,
                signal={
                    "kind": "sine",
                    "freq": 0.05,
                    "amplitude": 0.005,
                    "offset": 0.952,
                    "noise": 0.004,
                },
            ),
            Channel(
                "AccelerometerZ",
                unit="g",
                about="head acceleration, z",
                digits=3,
                signal={
                    "kind": "sine",
                    "freq": 0.06,
                    "amplitude": 0.02,
                    "offset": -0.245,
                    "noise": 0.015,
                },
            ),
            Channel(
                "GyroscopeX",
                unit="deg/s",
                about="head rotation rate, x",
                digits=3,
                signal={
                    "kind": "sine",
                    "freq": 0.09,
                    "amplitude": 1.4,
                    "offset": 0.347,
                    "noise": 1.2,
                },
            ),
            Channel(
                "GyroscopeY",
                unit="deg/s",
                about="head rotation rate, y",
                digits=3,
                signal={
                    "kind": "sine",
                    "freq": 0.06,
                    "amplitude": 0.9,
                    "offset": 0.695,
                    "phase": 1.1,
                    "noise": 0.75,
                },
            ),
            Channel(
                "GyroscopeZ",
                unit="deg/s",
                about="head rotation rate, z",
                digits=3,
                signal={
                    "kind": "sine",
                    "freq": 0.08,
                    "amplitude": 0.55,
                    "offset": 0.843,
                    "phase": 2.2,
                    "noise": 0.47,
                },
            ),
            Channel(
                "BatteryLevel",
                unit="%",
                about="charge remaining; a step function, not a slope (reported in 1/15ths)",
                digits=3,
                # The recording drains 6.667% over 26 minutes -- but in ONE step, because
                # the gauge only reports fifteenths. `quantize` is what reproduces that:
                # code that waits for the battery to *change* and code that fits a
                # *slope* to it behave very differently, and only a stepped simulation
                # tells them apart.
                signal={
                    "kind": "ramp",
                    "start": 93.333,
                    "per_second": -0.00435,
                    "quantize": 6.6667,
                    "minimum": 0.0,
                },
            ),
            Channel(
                "Counter",
                unit="",
                about="sample counter; a jump means the link dropped packets",
                digits=0,
                signal={"kind": "counter", "start": 206206},
            ),
            Channel(
                "ValidationIndicator",
                unit="flag",
                about="1 when the sample is valid",
                digits=0,
                signal={"kind": "constant", "value": 1},
            ),
        ),
        # Nothing. In 26 minutes this device dropped no packets and flagged no invalid
        # samples, so the profile simulates neither. Add a `dropout` stage if your link
        # is worse than this one's -- the Counter column is how you will see it.
        simulate=(),
        notes=(
            "8 EEG channels (Fz C3 Cz C4 Pz PO7 Oz PO8) plus a 6-axis IMU, battery, "
            "sample counter, and validity flag -- 17 columns, of which only 8 are brain. "
            "Rock steady in the paper's recording: 250.00 Hz, not one dropped packet. "
            "Counter increments by 1 per sample; a gap in it would be dropped packets, "
            "not a dropped value."
        ),
    )
)


# --------------------------------------------------------------------------- #
# Logitech C505e — webcam
# --------------------------------------------------------------------------- #

register_profile(
    DeviceProfile(
        key="c505e",
        aliases=("logitech_c505e", "webcam"),
        vendor="Logitech",
        model="C505e",
        modality="video",
        rate=30,
        channels=(Channel("frame", unit="jpeg/base64", about="one encoded frame per sample"),),
        notes=(
            "720p at 30 fps, fixed focus. Frames go on the wire base64-encoded, which "
            "is ~4 KB-40 KB per sample -- three orders of magnitude more than an EEG "
            "sample, and the reason the Core drops frames for a slow client rather than "
            "blocking. Needs the [video] extra: pip install thalamus[video]"
        ),
    )
)
