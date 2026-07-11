"""Cross-device alignment (paper §2.4.3, Figure 4)."""

from __future__ import annotations

from thalamus.protocol import Sample
from thalamus.sync import Synchronizer


def s(device, timestamp, value=0.0):
    return Sample(device, timestamp, {"v": value})


class TestAlignment:
    def test_pairs_each_reference_sample_with_the_nearest_other_sample(self):
        sync = Synchronizer(["eeg", "eye"], reference="eeg", tolerance_ms=20)

        frames = []
        frames += sync.push(s("eeg", 1000, 1.0))
        frames += sync.push(s("eye", 1003, 10.0))  # nearest to 1000
        frames += sync.push(s("eeg", 1004, 2.0))
        frames += sync.push(s("eye", 1013, 20.0))  # settles the 1004 frame

        assert [f.timestamp for f in frames] == [1000, 1004]
        assert frames[0].streams["eye"].data["v"] == 10.0
        assert frames[1].streams["eye"].data["v"] == 10.0  # 1003 is nearer to 1004 than 1013

    def test_a_frame_waits_until_the_nearest_sample_is_actually_known(self):
        # The heart of it: at the moment the reference sample arrives, a closer
        # sample from the other stream may still be in flight. Emitting immediately
        # would silently pair it with the wrong one.
        sync = Synchronizer(["eeg", "eye"], reference="eeg", tolerance_ms=50)

        assert sync.push(s("eye", 990, 1.0)) == []
        assert sync.push(s("eeg", 1000)) == []  # 990 is close, but 1005 might be closer

        frames = sync.push(s("eye", 1002, 2.0))  # now the other stream is past 1000
        assert len(frames) == 1
        assert frames[0].streams["eye"].data["v"] == 2.0  # and 1002 was indeed closer

    def test_a_slow_stream_is_upsampled_by_repetition(self):
        # 250 Hz reference against a 150 Hz stream: consecutive reference samples
        # legitimately share the same nearest neighbour, so it must not be consumed
        # by the first frame that uses it.
        sync = Synchronizer(["fast", "slow"], reference="fast", tolerance_ms=20)

        frames = []
        for n in range(5):
            frames += sync.push(s("fast", 1000 + n * 4))
        frames += sync.push(s("slow", 1002, 7.0))
        frames += sync.push(s("slow", 1030, 9.0))

        attached = [f.streams["slow"].data["v"] for f in frames if f.streams["slow"]]
        assert attached.count(7.0) >= 2  # the same slow sample serves several frames

    def test_a_stream_with_nothing_near_enough_contributes_nothing(self):
        # An honest None, not an interpolated guess.
        sync = Synchronizer(["eeg", "eye"], reference="eeg", tolerance_ms=5)

        sync.push(s("eye", 900))
        frames = sync.push(s("eeg", 1000)) + sync.push(s("eye", 1100))

        assert len(frames) == 1
        assert frames[0].streams["eye"] is None
        assert not frames[0].complete

    def test_out_of_order_arrivals_are_put_back_in_order(self):
        sync = Synchronizer(["eeg", "eye"], reference="eeg", tolerance_ms=20)
        sync.push(s("eye", 1010, 2.0))
        sync.push(s("eye", 1001, 1.0))  # arrived late, belongs earlier
        frames = sync.push(s("eeg", 1000)) + sync.push(s("eye", 1020))
        assert frames[0].streams["eye"].data["v"] == 1.0


class TestStalledDevices:
    def test_a_dead_device_does_not_hold_frames_forever(self):
        # Without this, one unplugged sensor silently stops the whole synchronizer —
        # a poor failure mode for a toolkit whose job is to surface exactly that.
        sync = Synchronizer(["eeg", "eye"], reference="eeg", timeout_ms=500, tolerance_ms=20)

        assert sync.push(s("eeg", 1000), now_ms=0.0) == []
        assert sync.tick(now_ms=100.0) == []  # not yet

        frames = sync.tick(now_ms=600.0)  # the eye tracker is clearly not coming back
        assert len(frames) == 1
        assert frames[0].streams["eye"] is None

    def test_flush_releases_everything_pending(self):
        sync = Synchronizer(["eeg", "eye"], reference="eeg")
        for n in range(3):
            sync.push(s("eeg", 1000 + n))
        assert len(sync.flush()) == 3

    def test_three_streams_align_together(self):
        sync = Synchronizer(["eeg", "eye", "ecg"], reference="eeg", tolerance_ms=20)

        sync.push(s("eye", 998, 1.0))
        sync.push(s("ecg", 1001, 2.0))
        frames = sync.push(s("eeg", 1000, 3.0))
        assert frames == []  # eye has not yet moved past 1000

        frames = sync.push(s("eye", 1005, 4.0))
        assert len(frames) == 1
        assert frames[0].complete
        assert frames[0].streams["ecg"].data["v"] == 2.0

    def test_the_wire_form_says_which_streams_were_missing(self):
        sync = Synchronizer(["eeg", "eye"], reference="eeg", tolerance_ms=1)
        sync.push(s("eeg", 1000))
        frame = sync.push(s("eye", 2000))[0].to_wire()

        assert frame["type"] == "frame"
        assert frame["complete"] is False
        assert frame["streams"]["eye"] is None
        assert frame["streams"]["eeg"]["timestamp"] == 1000
