"""The hardware profiles, checked against the real recordings.

The value of a profile is a promise: *code written against the simulation runs
unchanged against the recording*. That promise is only worth something if it is
enforced, so the header of a real export from each device is pinned here, and the
profile is checked against it. If someone renames a channel to something tidier,
these tests fail — which is the point, because the hardware will not rename it back.
"""

from __future__ import annotations

import logging
import statistics

import pytest

from thalamus.devices import ReplayDevice, SyntheticDevice, available_profiles, get_profile
from thalamus.processing import build_pipeline
from thalamus.protocol import MISSING, Sample

# The header line of a real recording from each device, verbatim. Sample rows are in
# pre_recorded_files/README.md; the full recordings are the ones used in the paper.
REAL_HEADERS = {
    "gp3": "timestamp,BPOGX,BPOGY,BPOGV,LPD,RPD",
    "unicorn_hybrid_black": (
        "timestamp,EEG1,EEG2,EEG3,EEG4,EEG5,EEG6,EEG7,EEG8,"
        "AccelerometerX,AccelerometerY,AccelerometerZ,"
        "GyroscopeX,GyroscopeY,GyroscopeZ,"
        "BatteryLevel,Counter,ValidationIndicator"
    ),
}


class TestAgainstTheRealRecordings:
    @pytest.mark.parametrize("key", sorted(REAL_HEADERS))
    def test_the_profile_has_exactly_the_columns_the_hardware_writes(self, key):
        expected = REAL_HEADERS[key].split(",")[1:]  # drop 'timestamp'
        assert get_profile(key).channel_names == expected

    @pytest.mark.parametrize("key", sorted(REAL_HEADERS))
    def test_a_synthetic_device_emits_those_same_columns(self, key):
        # The promise, tested directly: swap the recording for the simulation and the
        # client cannot tell.
        device = SyntheticDevice("d", profile=key, seed=1)
        sample = next(iter(device.samples()))
        assert list(sample) == REAL_HEADERS[key].split(",")[1:]

    def test_the_rates_are_the_ones_the_recordings_actually_run_at(self):
        # Measured off the recordings: the Unicorn is a rock-steady 4 ms, the GP3
        # arrives every 6-7 ms.
        assert get_profile("unicorn_hybrid_black").rate == 250
        assert get_profile("gp3").rate == 150

    def test_a_recording_missing_the_validity_column_is_reported(self, caplog):
        # A GP3 export without BPOGV cannot tell you which samples were blinks, and no
        # processing can recover it. Better to hear about it in the first second.
        profile = get_profile("gp3")
        assert profile.missing_from(["BPOGX", "BPOGY", "LPD", "RPD"]) == ["BPOGV"]
        assert profile.missing_from(REAL_HEADERS["gp3"].split(",")) == []


class TestSyntheticValuesLookLikeTheRealOnes:
    """Not physiology — but the ranges have to be right, or every plot axis is wrong
    and every filter is tuned against a signal that does not exist."""

    def test_gaze_leaves_the_screen_sometimes(self):
        # It is tempting to clamp gaze to 0..1, since that is where the screen is. The
        # recording says otherwise: 13% of *valid* BPOGY values fall outside 0..1,
        # reaching -1.38 and +2.38, because the participant looks past the monitor and
        # the tracker keeps extrapolating.
        #
        # A simulation that never leaves 0..1 will happily pass code that does
        # `int(BPOGY * screen_height)` — and that code throws on the first real file.
        # So this asserts the *unsafe* behaviour, on purpose.
        device = SyntheticDevice("eye", profile="gp3", duration_s=300, seed=4)
        y = [s["BPOGY"] for s in device.samples()]

        off_screen = sum(1 for v in y if v < 0.0 or v > 1.0)
        assert off_screen > 0, "gaze never left the screen; the real GP3's does, often"

    def test_gaze_does_not_pin_to_an_edge(self):
        # ...but it must not run away either. A free random walk drifts until it hits
        # whatever bound exists and sticks there, simulating a participant who stares at
        # one spot for five minutes. The walk is mean-reverting; this guards that.
        device = SyntheticDevice("eye", profile="gp3", duration_s=300, seed=4)
        x = [s["BPOGX"] for s in device.samples()]
        assert 0.2 < sum(x) / len(x) < 0.8, "gaze drifted away from the screen centre"

    def test_pupil_diameters_are_in_the_range_the_gp3_reports(self):
        # Real: mean 17.4 px, sd 2.2, 1st-99th percentile 13.1-22.7. An earlier version
        # of this profile had a sd of 0.25 px — a pupil that never dilates, which is a
        # strange thing to hand to someone building a pupillometry study.
        device = SyntheticDevice("eye", profile="gp3", duration_s=300, seed=4)
        pupils = [s["LPD"] for s in device.samples()]

        mean = sum(pupils) / len(pupils)
        sd = statistics.pstdev(pupils)
        assert 14.0 < mean < 21.0, f"mean pupil {mean:.1f} px is not what a GP3 reports"
        assert 1.0 < sd < 4.0, f"pupil sd {sd:.2f} px: real is 2.2"

    def test_eeg_is_in_microvolts_not_volts(self):
        device = SyntheticDevice("eeg", profile="unicorn_hybrid_black", duration_s=4, seed=1)
        values = [s["EEG1"] for s in device.samples()]
        assert 5.0 < max(values) < 100.0, "an EEG channel should swing tens of uV"

    def test_the_accelerometer_feels_gravity(self):
        # AccelerometerY sits at ~1 g in the recording: the head is upright and the
        # device is measuring the planet. A simulation centred on 0 would be wrong in a
        # way that silently breaks any head-movement artefact detection.
        device = SyntheticDevice("eeg", profile="unicorn_hybrid_black", duration_s=2, seed=1)
        y = [s["AccelerometerY"] for s in device.samples()]
        assert 0.9 < sum(y) / len(y) < 1.0

    def test_the_counter_counts_every_sample(self):
        # It is the one channel that can prove a packet was lost, so it had better be
        # exact: consecutive samples must differ by exactly 1.
        device = SyntheticDevice("eeg", profile="unicorn_hybrid_black", duration_s=1, seed=1)
        counters = [s["Counter"] for s in device.samples()]

        assert len(counters) == 250
        assert all(isinstance(c, int) for c in counters), "a counter is an integer"
        assert all(b - a == 1 for a, b in zip(counters, counters[1:]))

    def test_flags_are_integers_on_the_wire(self):
        # The real file has `1`, not `1.0`. A client comparing to 1 should just work.
        eye = next(iter(SyntheticDevice("eye", profile="gp3", seed=1).samples()))
        assert eye["BPOGV"] == 1
        assert isinstance(eye["BPOGV"], int)

    def test_the_battery_runs_down_in_steps_not_a_slope(self):
        # The gauge reports in fifteenths, so the paper's 26-minute recording contains
        # exactly TWO distinct battery values (93.333 and 86.667) — a step, not a ramp.
        # Code that waits for a change and code that fits a slope are different code.
        device = SyntheticDevice("eeg", profile="unicorn_hybrid_black", duration_s=1560, seed=1)
        levels = [s["BatteryLevel"] for s in device.samples()]

        assert levels[0] > levels[-1], "the battery should discharge, not charge"
        distinct = sorted(set(levels))
        assert len(distinct) <= 3, f"battery should step, not slide: {len(distinct)} values"
        assert distinct[0] == pytest.approx(86.667, abs=0.01)

    def test_the_unicorn_simulates_no_packet_loss(self):
        # It dropped nothing in 26 minutes. An earlier version of this profile gave it
        # Bluetooth dropouts by default — a failure mode taken from a datasheet rather
        # than from the data. Do not put words in the hardware's mouth.
        assert get_profile("unicorn_hybrid_black").simulate == ()


class TestReplayingARealRecording:
    """Rows lifted verbatim from the recordings, replayed through the device."""

    GP3_ROWS = (
        "timestamp,BPOGX,BPOGY,BPOGV,LPD,RPD\n"
        "1690535469479,0.58224,0.37114,1,16.61074,15.31291\n"
        "1690535469485,0.57659,0.38235,1,16.46606,15.16937\n"
        "1690535469491,0.59446,0.36541,1,15.84853,15.91231\n"
    )

    def _replay(self, tmp_path, text, **kwargs):
        path = tmp_path / "rec.csv"
        path.write_text(text, encoding="utf-8")
        device = ReplayDevice("eye", path, **kwargs)
        return list(device.samples())

    def test_the_real_columns_survive_the_round_trip(self, tmp_path):
        rows = self._replay(tmp_path, self.GP3_ROWS, profile="gp3")

        assert len(rows) == 3
        assert rows[0]["timestamp"] == 1690535469479
        assert rows[0]["BPOGX"] == 0.58224
        assert rows[0]["LPD"] == 16.61074

    def test_an_integer_column_is_not_widened_to_a_float(self, tmp_path):
        # The file says BPOGV=1. A client that sees 1 from the live tracker and 1.0
        # from the replay of the same tracker has been handed a bug.
        rows = self._replay(tmp_path, self.GP3_ROWS, profile="gp3")
        assert rows[0]["BPOGV"] == 1
        assert isinstance(rows[0]["BPOGV"], int)

    def test_a_recording_without_the_validity_column_warns(self, tmp_path, caplog):
        stripped = "timestamp,BPOGX,BPOGY,LPD,RPD\n1690535469479,0.58224,0.37114,16.61,15.31\n"
        with caplog.at_level(logging.WARNING):
            self._replay(tmp_path, stripped, profile="gp3")

        assert "BPOGV" in caplog.text
        assert "valid" in caplog.text

    def test_a_recording_that_matches_the_profile_says_nothing(self, tmp_path, caplog):
        with caplog.at_level(logging.WARNING):
            self._replay(tmp_path, self.GP3_ROWS, profile="gp3")
        assert caplog.text == ""


class TestBlinksLookLikeRealBlinks:
    """The GP3's blink, reproduced.

    Measured over the paper's 26-minute recording (230,974 samples, 118 blinks):

    * 1.28% of samples have ``BPOGV=0``;
    * a blink runs a median of 19.5 samples (131 ms);
    * **115 of the 116 multi-sample blinks have every column identical throughout**;
    * **116 of the 118 onsets repeat the preceding valid row exactly**.

    In other words the tracker does not blank anything — it freezes. Those rows carry
    entirely plausible numbers, and only ``BPOGV`` says otherwise. Reproducing that is
    the difference between a dry run that prepares you and one that flatters you.
    """

    def _simulate(self, seconds=600, seed=5):
        profile = get_profile("gp3")
        device = SyntheticDevice("eye", profile="gp3", duration_s=seconds, seed=seed)
        pipeline = build_pipeline([dict(s) for s in profile.simulate])
        return [
            s.data
            for i, reading in enumerate(device.samples())
            for s in pipeline.process(Sample("eye", 1_690_535_469_479 + i * 7, dict(reading)))
        ]

    def _runs(self, rows):
        runs, current = [], []
        for row in rows:
            if row["BPOGV"] == 0:
                current.append(row)
            elif current:
                runs.append(current)
                current = []
        return runs

    def test_a_blink_freezes_rather_than_blanks(self):
        # The property. Every row of a blink must be identical to the others: nothing
        # in the data columns betrays it, which is why BPOGV has to be read.
        rows = self._simulate()
        runs = [r for r in self._runs(rows) if len(r) > 1]
        assert runs, "no blinks were simulated at all"

        for run in runs:
            assert all(r["LPD"] == run[0]["LPD"] for r in run), "a blink must not vary"
            assert all(r["BPOGX"] == run[0]["BPOGX"] for r in run)
            assert all(r["LPD"] is not MISSING for r in run), "the GP3 does not blank"

    def test_a_blink_repeats_the_last_valid_reading(self):
        # Not its own fresh value: the one before the gap. An off-by-one here leaves the
        # first sample of every blink carrying a plausible, wrong measurement.
        rows = self._simulate()
        onsets = [
            (rows[i - 1], rows[i])
            for i in range(1, len(rows))
            if rows[i]["BPOGV"] == 0 and rows[i - 1]["BPOGV"] == 1
        ]
        assert onsets

        for before, first in onsets:
            assert first["LPD"] == before["LPD"]
            assert first["BPOGX"] == before["BPOGX"]

    def test_blinks_arrive_at_roughly_the_rate_they_really_do(self):
        rows = self._simulate()
        invalid = [r for r in rows if r["BPOGV"] == 0]
        share = len(invalid) / len(rows)
        assert 0.004 < share < 0.03, f"{share:.1%} of samples invalid; real is 1.28%"

        lengths = [len(r) for r in self._runs(rows)]
        median = statistics.median(lengths)
        assert 10 < median < 32, f"median blink {median} samples; real is 19.5 (131 ms)"

    def test_validity_mask_is_the_only_thing_that_saves_you(self):
        # The payoff. A client that ignores BPOGV averages the frozen blinks into its
        # baseline and cannot tell; one that runs validity_mask gets honest gaps.
        rows = self._simulate()
        mask = build_pipeline([{"stage": "validity_mask", "profile": "gp3"}])

        masked = [
            s.data
            for i, row in enumerate(rows)
            for s in mask.process(Sample("eye", 1_690_535_469_479 + i * 7, dict(row)))
        ]
        blinks = sum(1 for r in rows if r["BPOGV"] == 0)
        gaps = sum(1 for r in masked if r["LPD"] is MISSING)

        assert blinks > 0
        assert gaps == blinks, "every frozen blink must become a real gap"

        # And without the mask, not one of them is detectable as missing.
        assert not any(r["LPD"] is MISSING for r in rows)


class TestTheRegistry:
    def test_the_papers_three_devices_are_all_there(self):
        assert set(available_profiles()) >= {"gp3", "unicorn_hybrid_black", "c505e"}

    def test_aliases_resolve_to_the_same_profile(self):
        assert get_profile("unicorn") is get_profile("unicorn_hybrid_black")
        assert get_profile("gazepoint_gp3") is get_profile("gp3")

    def test_an_unknown_profile_says_what_is_available(self):
        with pytest.raises(ValueError, match="gp3"):
            get_profile("no_such_device")

    def test_an_explicit_rate_overrides_the_profile(self):
        # Borrow a Unicorn's channels, run them at 500 Hz, see what breaks. That is a
        # legitimate thing to want (paper §3.3), so the profile must not be a cage.
        device = SyntheticDevice("eeg", profile="unicorn_hybrid_black", rate=500, seed=1)
        assert device.rate == 500
        assert "EEG8" in next(iter(device.samples()))
