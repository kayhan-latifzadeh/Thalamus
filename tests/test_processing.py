"""The stages of paper §2.4 — the features the original release documented but never shipped."""

from __future__ import annotations

import pytest

from thalamus.processing import (
    ConstantNoiseStage,
    DelayStage,
    DropoutStage,
    ExponentialStage,
    GaussianNoiseStage,
    KalmanStage,
    MissingFillStage,
    MissingInjectStage,
    MovingAverageStage,
    Pipeline,
    PipelineSpecError,
    SavitzkyGolayStage,
    ValidityMaskStage,
    build_pipeline,
    build_stage,
    pipeline_key,
)
from thalamus.protocol import MISSING, Sample


def run(stage, values, channel="x", start=1000, step=10):
    """Push ``values`` through ``stage`` and return the channel's output values."""
    out = []
    for index, value in enumerate(values):
        out.extend(stage.apply(Sample("d", start + index * step, {channel: value})))
    out.extend(stage.flush())
    return [s.data[channel] for s in out]


class TestFilters:
    def test_moving_average_smooths(self):
        assert run(MovingAverageStage(window=3), [1, 1, 1, 10, 1]) == pytest.approx([1, 1, 1, 4, 4])

    def test_exponential_converges_towards_a_step(self):
        out = run(ExponentialStage(alpha=0.5), [0, 10, 10, 10, 10])
        assert out[0] == 0
        assert out == sorted(out)  # monotonically approaching
        assert out[-1] == pytest.approx(10, abs=1.0)

    def test_kalman_rejects_noise_around_a_constant(self):
        noisy = [10.0, 10.6, 9.4, 10.5, 9.5, 10.4, 9.6] * 6
        filtered = run(KalmanStage(process_noise=1e-5, measurement_noise=0.5), noisy)
        # Whatever else it does, it must reduce the variance of the estimate.
        assert _spread(filtered[10:]) < _spread(noisy[10:]) / 4
        assert filtered[-1] == pytest.approx(10.0, abs=0.3)

    def test_kalman_still_tracks_a_real_change(self):
        # A filter that only smooths is a filter that has stopped listening.
        out = run(KalmanStage(process_noise=0.05, measurement_noise=0.1), [0.0] * 20 + [5.0] * 40)
        assert out[-1] == pytest.approx(5.0, abs=0.3)

    @pytest.mark.parametrize("mode", ["causal", "centered"])
    def test_savgol_reproduces_a_polynomial_exactly(self, mode):
        # The defining property: a Savitzky-Golay filter of order p fits a degree-p
        # polynomial perfectly, so data that *is* such a polynomial must come back
        # unchanged. This also pins down the coefficient ordering — get it backwards
        # and this test fails, where a smoothness test would not.
        quadratic = [2.0 * n * n - 3.0 * n + 7.0 for n in range(30)]
        stage = SavitzkyGolayStage(window=7, polyorder=3, mode=mode)

        emitted = []
        for n, value in enumerate(quadratic):
            emitted.extend(stage.apply(Sample("d", 1000 + n, {"x": value})))
        emitted.extend(stage.flush())

        for sample in emitted:
            expected = quadratic[sample.timestamp - 1000]
            assert sample.data["x"] == pytest.approx(expected, abs=1e-6)

    def test_centered_savgol_lags_by_half_a_window_and_says_so_in_its_timestamps(self):
        stage = SavitzkyGolayStage(window=5, polyorder=2, mode="centered")
        # Nothing comes out until the window is full...
        for n in range(4):
            assert list(stage.apply(Sample("d", 1000 + n, {"x": float(n)}))) == []
        # ...and then what comes out is the *middle* of the window, not the newest.
        [emitted] = stage.apply(Sample("d", 1004, {"x": 4.0}))
        assert emitted.timestamp == 1002

    def test_causal_savgol_never_lags(self):
        stage = SavitzkyGolayStage(window=11, polyorder=3, mode="causal")
        for n in range(20):
            [emitted] = stage.apply(Sample("d", 1000 + n, {"x": float(n)}))
            assert emitted.timestamp == 1000 + n  # one out per one in, same instant

    def test_a_filter_does_not_smooth_across_a_gap(self):
        # Interpolating over missing data would invent readings that were never taken.
        stage = SavitzkyGolayStage(window=5, polyorder=2)
        out = []
        for n, value in enumerate([1.0, 2.0, MISSING, 4.0, 5.0, 6.0, 7.0]):
            out.extend(stage.apply(Sample("d", 1000 + n, {"x": value})))
        assert out[2].data["x"] is MISSING

    def test_savgol_rejects_impossible_parameters(self):
        with pytest.raises(ValueError, match="polyorder"):
            SavitzkyGolayStage(window=5, polyorder=5)
        with pytest.raises(ValueError, match="odd"):
            SavitzkyGolayStage(window=6, polyorder=2, mode="centered")


class TestNoise:
    def test_gaussian_noise_perturbs_around_the_signal(self):
        out = run(GaussianNoiseStage(sigma=1.0, seed=42), [10.0] * 400)
        assert _mean(out) == pytest.approx(10.0, abs=0.2)
        assert _spread(out) == pytest.approx(1.0, abs=0.15)

    def test_the_same_seed_gives_the_same_noise(self):
        # Not a nicety: a study whose noise cannot be reproduced is a study whose
        # results cannot be reproduced.
        first = run(GaussianNoiseStage(sigma=2.0, seed=7), [0.0] * 50)
        second = run(GaussianNoiseStage(sigma=2.0, seed=7), [0.0] * 50)
        assert first == second

    def test_different_seeds_give_different_noise(self):
        assert run(GaussianNoiseStage(sigma=2.0, seed=1), [0.0] * 20) != run(
            GaussianNoiseStage(sigma=2.0, seed=2), [0.0] * 20
        )

    def test_relative_noise_scales_with_the_signal(self):
        small = _spread(run(GaussianNoiseStage(sigma=0.1, relative=True, seed=3), [1.0] * 300))
        large = _spread(run(GaussianNoiseStage(sigma=0.1, relative=True, seed=3), [100.0] * 300))
        assert large == pytest.approx(small * 100, rel=0.2)

    def test_constant_noise_is_a_fixed_offset(self):
        assert run(ConstantNoiseStage(offset=5.0), [1.0, 2.0, 3.0]) == [6.0, 7.0, 8.0]

    def test_drift_grows_over_time(self):
        out = run(ConstantNoiseStage(offset=0.0, drift=0.5), [0.0] * 4)
        assert out == [0.0, 0.5, 1.0, 1.5]

    def test_noise_is_never_added_to_a_gap(self):
        stage = GaussianNoiseStage(sigma=1.0, seed=1)
        [out] = stage.apply(Sample("d", 1, {"x": MISSING}))
        assert out.data["x"] is MISSING

    def test_noise_only_touches_the_channels_it_was_given(self):
        stage = GaussianNoiseStage(sigma=1.0, seed=1, channels=["x"])
        [out] = stage.apply(Sample("d", 1, {"x": 0.0, "y": 0.0}))
        assert out.data["x"] != 0.0
        assert out.data["y"] == 0.0


class TestDelay:
    def test_timestamp_mode_shifts_time_without_holding_the_sample(self):
        stage = DelayStage(mode="timestamp", delay_ms=100)
        [out] = stage.apply(Sample("d", 1000, {"x": 1.0}))
        assert out.timestamp == 1100  # arrives at once, but claims to be older

    def test_buffer_mode_holds_samples_back(self):
        # Note this is what the stream looks like *live*: at any instant, the last
        # `samples` readings are still in the buffer and have not been delivered.
        # (`run` flushes at the end, so it would not show the lag — hence the
        # explicit loop.)
        stage = DelayStage(mode="buffer", samples=3)
        held = []
        for n in range(3):
            held.extend(stage.apply(Sample("d", 1000 + n, {"x": float(n)})))
        assert held == []

    def test_buffer_mode_delays_delivery_by_exactly_n_samples(self):
        stage = DelayStage(mode="buffer", samples=2)
        emitted = []
        for n in range(5):
            emitted.extend(stage.apply(Sample("d", 1000 + n, {"x": float(n)})))
        assert [s.data["x"] for s in emitted] == [0.0, 1.0, 2.0]  # 3 out for 5 in

    def test_a_held_stream_is_not_truncated_when_it_ends(self):
        stage = DelayStage(mode="buffer", samples=3)
        for n in range(3):
            stage.apply(Sample("d", 1000 + n, {"x": float(n)}))
        assert [s.data["x"] for s in stage.flush()] == [0.0, 1.0, 2.0]

    def test_jitter_makes_the_delay_vary(self):
        stage = DelayStage(mode="timestamp", delay_ms=100, jitter_ms=50, seed=1)
        shifts = {s.timestamp - 1000 for s in (stage.apply(Sample("d", 1000, {"x": 1.0}))[0],)}
        for _ in range(20):
            [out] = stage.apply(Sample("d", 1000, {"x": 1.0}))
            shifts.add(out.timestamp - 1000)
        assert len(shifts) > 1
        assert all(50 <= s <= 150 for s in shifts)


class TestDropout:
    def test_everything_survives_at_probability_zero(self):
        assert len(run(DropoutStage(probability=0.0, seed=1), [1.0] * 50)) == 50

    def test_nothing_survives_at_probability_one(self):
        assert run(DropoutStage(probability=1.0, seed=1), [1.0] * 50) == []

    def test_losses_arrive_in_bursts(self):
        # A real connection stall drops a run of samples, not one at a time.
        stage = DropoutStage(probability=1.0, burst=5, seed=1)
        assert run(stage, [1.0] * 10) == []
        assert len(run(DropoutStage(probability=0.05, burst=5, seed=3), [1.0] * 500)) < 500


class TestMissingValues:
    def test_injected_gaps_blank_the_channel(self):
        out = run(MissingInjectStage(probability=1.0, seed=1), [1.0, 2.0, 3.0])
        assert all(v is MISSING for v in out)

    def test_gaps_come_in_bursts_of_the_requested_length(self):
        # A blink blanks a *run* of consecutive samples. Independent coin flips per
        # sample would produce a salt-and-pepper pattern that no real eye tracker
        # ever produces, and that is far easier for downstream code to survive.
        stage = MissingInjectStage(probability=0.02, burst=[5, 5], seed=11)
        out = run(stage, [1.0] * 2000)

        runs, current = [], 0
        for value in out:
            if value is MISSING:
                current += 1
            elif current:
                runs.append(current)
                current = 0

        assert len(runs) > 5, "expected several gaps at p=0.02 over 2000 samples"
        assert set(runs) == {5}, (
            f"every gap should be exactly 5 samples long, got {sorted(set(runs))}"
        )

    def test_a_gap_can_be_restricted_to_one_channel(self):
        # A blink blanks the pupil, but the tracker still reports a gaze position.
        stage = MissingInjectStage(probability=1.0, channels=["pupil"], seed=1)
        [out] = stage.apply(Sample("eye", 1, {"pupil": 3.5, "gaze_x": 960.0}))
        assert out.data["pupil"] is MISSING
        assert out.data["gaze_x"] == 960.0

    def test_a_blink_drops_the_validity_flag_rather_than_blanking_it(self):
        # What a real GP3 does: it keeps emitting rows through a blink, with BPOGV=0.
        # If the flag were blanked along with everything else it would carry no
        # information, and a client could not distinguish a blink from a dead tracker.
        stage = MissingInjectStage(probability=1.0, flag="BPOGV", seed=1)
        [out] = stage.apply(Sample("eye", 1, {"BPOGX": 0.5, "LPD": 16.1, "BPOGV": 1}))

        assert out.data["BPOGV"] == 0, "the flag reports the failure, it does not vanish"
        assert out.data["BPOGX"] is MISSING
        assert out.data["LPD"] is MISSING

    def test_hold_mode_freezes_the_last_valid_reading(self):
        # What the real GP3 does. It does not blank the pupil during a blink -- it
        # repeats the last one it believed, so the rows look like measurements.
        stage = MissingInjectStage(probability=0.0, mode="hold", flag="BPOGV", seed=1)

        [good] = stage.apply(Sample("eye", 1, {"LPD": 16.1, "BPOGV": 1}))
        assert good.data["LPD"] == 16.1

        stage.probability = 1.0  # the next sample begins a blink
        [blink] = stage.apply(Sample("eye", 2, {"LPD": 99.9, "BPOGV": 1}))

        assert blink.data["LPD"] == 16.1, "the blink must repeat the last valid reading"
        assert blink.data["LPD"] is not MISSING, "the GP3 does not blank; it freezes"
        assert blink.data["BPOGV"] == 0, "...only the flag gives it away"

    def test_hold_mode_keeps_the_whole_burst_constant(self):
        stage = MissingInjectStage(probability=0.0, burst=4, mode="hold", flag="BPOGV", seed=1)
        stage.apply(Sample("eye", 0, {"LPD": 16.1, "BPOGV": 1}))

        stage.probability = 1.0
        out = [
            s
            for n in range(4)
            for s in stage.apply(Sample("eye", n + 1, {"LPD": 20.0 + n, "BPOGV": 1}))
        ]

        assert [s.data["LPD"] for s in out] == [16.1, 16.1, 16.1, 16.1]
        assert all(s.data["BPOGV"] == 0 for s in out)

    def test_an_unknown_mode_is_refused(self):
        with pytest.raises(ValueError, match="mode"):
            MissingInjectStage(mode="pretend")

    def test_zero_fill_is_figure_2_of_the_paper(self):
        out = run(MissingFillStage(strategy="zero"), [1.0, MISSING, MISSING, 4.0])
        assert out == [1.0, 0.0, 0.0, 4.0]

    def test_hold_carries_the_last_valid_reading_forward(self):
        out = run(MissingFillStage(strategy="hold"), [1.0, MISSING, MISSING, 4.0])
        assert out == [1.0, 1.0, 1.0, 4.0]

    def test_hold_before_any_valid_reading_falls_back_to_the_default(self):
        out = run(MissingFillStage(strategy="hold", value=-1), [MISSING, 2.0])
        assert out == [-1, 2.0]

    def test_fill_drop_discards_the_whole_sample(self):
        stage = MissingFillStage(strategy="drop")
        emitted = []
        for n, value in enumerate([1.0, MISSING, 3.0]):
            emitted.extend(stage.apply(Sample("d", 1000 + n, {"x": value})))
        assert [s.timestamp for s in emitted] == [1000, 1002]  # the stream is now irregular

    def test_a_sentinel_value_stays_detectable_downstream(self):
        assert run(MissingFillStage(strategy="value", value=-1), [MISSING]) == [-1]


class TestValidityMask:
    """A device's own verdict on its samples, honoured (the GP3's BPOGV, the Unicorn's
    ValidationIndicator)."""

    def test_an_invalid_sample_is_blanked(self):
        stage = ValidityMaskStage(flag="BPOGV")
        [out] = stage.apply(Sample("eye", 1, {"BPOGX": 0.5, "LPD": 16.1, "BPOGV": 0}))
        assert out.data["BPOGX"] is MISSING
        assert out.data["LPD"] is MISSING
        assert out.data["BPOGV"] == 0

    def test_a_valid_sample_is_untouched(self):
        stage = ValidityMaskStage(flag="BPOGV")
        [out] = stage.apply(Sample("eye", 1, {"BPOGX": 0.5, "BPOGV": 1}))
        assert out.data["BPOGX"] == 0.5

    def test_stale_values_behind_a_zero_flag_do_not_survive(self):
        # The failure this stage exists to prevent. Real hardware does not blank the
        # data columns when it loses tracking -- it leaves numbers there, and they are
        # meaningless. Nothing downstream can tell, so it must be caught here.
        stage = ValidityMaskStage(flag="BPOGV")
        [out] = stage.apply(Sample("eye", 1, {"BPOGX": 0.0, "LPD": 0.0, "BPOGV": 0}))
        assert out.data["LPD"] is MISSING, "a 0.0 behind BPOGV=0 is not a 0 mm pupil"

    def test_a_csv_string_flag_still_reads_as_valid(self):
        # Replay hands us whatever the file had. "1" and 1 and 1.0 all mean valid.
        stage = ValidityMaskStage(flag="BPOGV")
        [out] = stage.apply(Sample("eye", 1, {"BPOGX": 0.5, "BPOGV": "1"}))
        assert out.data["BPOGX"] == 0.5

    def test_the_profile_supplies_the_flag_and_what_it_covers(self):
        stage = ValidityMaskStage(profile="gp3")
        assert stage.flag == "BPOGV"

        [out] = stage.apply(
            Sample("eye", 1, {"BPOGX": 0.5, "BPOGY": 0.4, "LPD": 16.1, "RPD": 15.8, "BPOGV": 0})
        )
        assert all(out.data[c] is MISSING for c in ("BPOGX", "BPOGY", "LPD", "RPD"))

    def test_a_unicorn_flag_does_not_blank_the_battery(self):
        # The Unicorn's ValidationIndicator vouches for the 8 EEG channels. It says
        # nothing about the battery level or the sample counter, and blanking those
        # would destroy the very evidence you use to diagnose the problem.
        stage = ValidityMaskStage(profile="unicorn_hybrid_black")
        [out] = stage.apply(
            Sample(
                "eeg",
                1,
                {"EEG1": 12.0, "BatteryLevel": 93.3, "Counter": 7, "ValidationIndicator": 0},
            )
        )
        assert out.data["EEG1"] is MISSING
        assert out.data["BatteryLevel"] == 93.3
        assert out.data["Counter"] == 7

    def test_drop_discards_the_whole_sample(self):
        stage = ValidityMaskStage(flag="BPOGV", drop=True)
        assert stage.apply(Sample("eye", 1, {"BPOGX": 0.5, "BPOGV": 0})) == []
        assert len(list(stage.apply(Sample("eye", 2, {"BPOGX": 0.5, "BPOGV": 1})))) == 1

    def test_a_device_with_no_flag_column_is_passed_through(self):
        # Not every device reports validity. Absence of a flag is not a failed sample.
        stage = ValidityMaskStage(flag="BPOGV")
        [out] = stage.apply(Sample("mouse", 1, {"x": 3.0}))
        assert out.data["x"] == 3.0

    def test_it_refuses_to_be_built_without_a_flag(self):
        with pytest.raises(ValueError, match="flag"):
            ValidityMaskStage()

    def test_the_round_trip_a_study_actually_runs(self):
        # Inject blinks the way the hardware produces them, then mask them the way a
        # client consumes them. These are the two halves of the same story and they
        # have to agree: what the simulated GP3 emits, the GP3 profile must recognise.
        blink = MissingInjectStage(
            probability=1.0, burst=3, flag="BPOGV", channels=["BPOGX", "LPD"], seed=1
        )
        mask = ValidityMaskStage(profile="gp3")

        emitted = blink.apply(Sample("eye", 1, {"BPOGX": 0.5, "LPD": 16.1, "BPOGV": 1}))
        [out] = [s for sample in emitted for s in mask.apply(sample)]

        assert out.data["BPOGV"] == 0
        assert out.data["LPD"] is MISSING


class TestPipeline:
    def test_stages_run_in_order(self):
        # Fill the gaps, *then* smooth: smoothing across a gap is meaningless, so the
        # order is not a detail.
        pipeline = Pipeline([MissingFillStage(strategy="zero"), ConstantNoiseStage(offset=10)])
        [out] = pipeline.process(Sample("d", 1, {"x": MISSING}))
        assert out.data["x"] == 10.0

    def test_a_stage_that_drops_a_sample_ends_the_chain(self):
        pipeline = Pipeline([DropoutStage(probability=1.0, seed=1), ConstantNoiseStage(offset=1)])
        assert pipeline.process(Sample("d", 1, {"x": 0.0})) == []

    def test_an_empty_pipeline_is_the_identity(self):
        sample = Sample("d", 1, {"x": 1.0})
        assert Pipeline().process(sample) == [sample]

    def test_flush_drains_buffered_stages_through_the_rest_of_the_chain(self):
        pipeline = Pipeline([DelayStage(mode="buffer", samples=2), ConstantNoiseStage(offset=100)])
        for n in range(2):
            assert pipeline.process(Sample("d", 1000 + n, {"x": float(n)})) == []
        # The held samples must still be offset by the downstream stage on the way out.
        assert [s.data["x"] for s in pipeline.flush()] == [100.0, 101.0]


class TestBuildFromSpec:
    def test_builds_a_stage_from_a_config_dict(self):
        stage = build_stage({"stage": "gaussian_noise", "sigma": 2.0, "seed": 1})
        assert isinstance(stage, GaussianNoiseStage)
        assert stage.sigma == 2.0

    def test_an_unknown_stage_names_the_ones_that_exist(self):
        with pytest.raises(PipelineSpecError, match="available stages"):
            build_stage({"stage": "magic"})

    def test_a_bad_parameter_is_rejected_at_build_time(self):
        with pytest.raises(PipelineSpecError):
            build_stage({"stage": "gaussian_noise", "sigma": "loud"})
        with pytest.raises(PipelineSpecError, match="bad parameters"):
            build_stage({"stage": "gaussian_noise", "nonsense": 1})

    def test_a_stage_entry_must_name_a_stage(self):
        with pytest.raises(PipelineSpecError, match="'stage' key"):
            build_pipeline([{"sigma": 1.0}])

    def test_pipeline_key_ignores_how_the_spec_was_written(self):
        # Two clients asking for the same processing must share one pipeline, however
        # they happened to order their keys.
        a = [{"stage": "savgol", "window": 11, "polyorder": 3}]
        b = [{"polyorder": 3, "stage": "savgol", "window": 11}]
        assert pipeline_key(a) == pipeline_key(b)

    def test_pipeline_key_distinguishes_different_processing(self):
        assert pipeline_key([{"stage": "savgol", "window": 11}]) != pipeline_key(
            [{"stage": "savgol", "window": 21}]
        )
        assert pipeline_key(None) == pipeline_key([])


def _mean(values):
    return sum(values) / len(values)


def _spread(values):
    mean = _mean(values)
    return (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
