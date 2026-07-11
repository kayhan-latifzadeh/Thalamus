"""The hardware profiles, checked against the real recordings.

The value of a profile is a promise: *code written against the simulation runs
unchanged against the recording*. That promise is only worth something if it is
enforced, so the header of a real export from each device is pinned here, and the
profile is checked against it. If someone renames a channel to something tidier,
these tests fail — which is the point, because the hardware will not rename it back.
"""

from __future__ import annotations

import logging

import pytest

from thalamus.devices import ReplayDevice, SyntheticDevice, available_profiles, get_profile

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

    def test_gaze_stays_on_the_screen_and_does_not_pin_to_an_edge(self):
        # A free random walk drifts into a clamp and stays there, simulating a
        # participant who stares at the top of the monitor for a minute. The GP3
        # profile's walk is mean-reverting; this is what guards that.
        device = SyntheticDevice("eye", profile="gp3", rate=150, duration_s=60, seed=4)
        x = [s["BPOGX"] for s in device.samples()]
        y = [s["BPOGY"] for s in device.samples()]

        assert all(0.0 < v < 1.0 for v in x + y), "gaze wandered off the screen"
        pinned = sum(1 for v in x + y if v <= 0.001 or v >= 0.999)
        assert pinned == 0, f"{pinned} samples pinned to a screen edge"

    def test_pupil_diameters_are_in_the_range_the_gp3_reports(self):
        device = SyntheticDevice("eye", profile="gp3", rate=150, duration_s=10, seed=4)
        pupils = [s["LPD"] for s in device.samples()]
        assert min(pupils) > 14.0
        assert max(pupils) < 18.0

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

    def test_the_battery_runs_down(self):
        device = SyntheticDevice("eeg", profile="unicorn_hybrid_black", duration_s=60, seed=1)
        levels = [s["BatteryLevel"] for s in device.samples()]
        assert levels[0] > levels[-1], "the battery should discharge, not charge"


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
