"""The wire format, and the framing bug that used to corrupt subscriptions."""

from __future__ import annotations

import json

import pytest

from thalamus.protocol import (
    MISSING,
    LineDecoder,
    ProtocolError,
    Sample,
    encode,
    is_control,
    now_ms,
)


class TestLineDecoder:
    """TCP hands you a byte stream, not messages. This is where that is dealt with."""

    def test_reassembles_a_message_split_across_chunks(self):
        # The original server did one recv(1024) and assumed it held a whole message.
        # A subscription to enough devices to exceed the buffer was silently truncated
        # and then failed to parse, so the client received nothing, forever.
        decoder = LineDecoder()
        payload = encode({"subscribe": [f"device_{i:03d}" for i in range(200)]})
        assert len(payload) > 1024

        received = []
        for start in range(0, len(payload), 100):
            received.extend(decoder.feed(payload[start : start + 100]))

        assert len(received) == 1
        assert len(received[0]["subscribe"]) == 200

    def test_splits_several_messages_from_one_chunk(self):
        decoder = LineDecoder()
        chunk = encode({"a": 1}) + encode({"a": 2}) + encode({"a": 3})
        assert [m["a"] for m in decoder.feed(chunk)] == [1, 2, 3]

    def test_holds_a_partial_line_until_the_rest_arrives(self):
        decoder = LineDecoder()
        assert list(decoder.feed(b'{"device_id": "eeg", "Fp1"')) == []
        assert [m["Fp1"] for m in decoder.feed(b": 1.5}\n")] == [1.5]

    def test_ignores_blank_lines(self):
        decoder = LineDecoder()
        assert len(list(decoder.feed(b'\n\n{"a":1}\n\n'))) == 1

    def test_a_malformed_line_raises_but_leaves_the_decoder_usable(self):
        decoder = LineDecoder()
        with pytest.raises(ProtocolError):
            list(decoder.feed(b"not json\n"))
        assert [m["a"] for m in decoder.feed(b'{"a":1}\n')] == [1]

    def test_a_line_that_never_ends_does_not_grow_without_bound(self):
        decoder = LineDecoder(max_line_bytes=1000)
        with pytest.raises(ProtocolError, match="exceeded"):
            list(decoder.feed(b"x" * 1001))


class TestSample:
    def test_channels_are_everything_that_is_not_metadata(self):
        sample = Sample.from_wire({"device_id": "eeg", "timestamp": 100, "Fp1": 1.0, "Fp2": 2.0})
        assert sample.device_id == "eeg"
        assert sample.timestamp == 100
        assert sample.data == {"Fp1": 1.0, "Fp2": 2.0}

    @pytest.mark.parametrize("token", ["NA", "N/A", "", "  ", "nan", "null", None, float("nan")])
    def test_every_way_a_device_says_nothing_becomes_MISSING(self, token):
        sample = Sample.from_wire({"device_id": "eye", "timestamp": 1, "pupil": token})
        assert sample.data["pupil"] is MISSING

    def test_a_real_zero_is_not_missing(self):
        # The distinction the whole missing-value design rests on.
        sample = Sample.from_wire({"device_id": "eye", "timestamp": 1, "pupil": 0})
        assert sample.data["pupil"] == 0
        assert sample.data["pupil"] is not MISSING

    def test_a_sample_without_a_device_id_is_rejected(self):
        with pytest.raises(ProtocolError, match="device_id"):
            Sample.from_wire({"timestamp": 1, "Fp1": 1.0})

    def test_a_missing_timestamp_falls_back_to_arrival(self):
        sample = Sample.from_wire({"device_id": "eeg", "Fp1": 1.0}, default_timestamp=999)
        assert sample.timestamp == 999

    def test_a_non_numeric_timestamp_is_rejected(self):
        with pytest.raises(ProtocolError, match="timestamp"):
            Sample.from_wire({"device_id": "eeg", "timestamp": "yesterday"})

    def test_float_timestamps_are_truncated_to_ms(self):
        assert (
            Sample.from_wire({"device_id": "d", "timestamp": 1690535469479.7}).timestamp
            == 1690535469479
        )

    def test_select_keeps_only_the_named_channels(self):
        sample = Sample("eeg", 1, {"Fp1": 1.0, "Fp2": 2.0, "Cz": 3.0})
        assert sample.select(["Fp1", "Cz"]).data == {"Fp1": 1.0, "Cz": 3.0}
        assert sample.select(None).data == sample.data

    def test_round_trips_through_the_wire(self):
        original = Sample("eeg", 100, {"Fp1": 1.5, "Fp2": MISSING})
        decoded = Sample.from_wire(json.loads(encode(original).decode()))
        assert decoded.timestamp == 100
        assert decoded.data["Fp1"] == 1.5
        assert decoded.data["Fp2"] is MISSING


class TestEncode:
    def test_missing_becomes_json_null_so_a_gap_is_distinguishable_from_a_zero(self):
        line = encode(Sample("eye", 1, {"pupil": MISSING}))
        assert json.loads(line)["pupil"] is None

    def test_nan_does_not_produce_invalid_json(self):
        # json.dumps happily emits bare NaN, which is not JSON and which every
        # non-Python client rejects. A NaN from a sensor must not be able to break
        # every other client in the study.
        line = encode({"device_id": "d", "x": float("nan"), "y": float("inf")})
        assert b"NaN" not in line and b"Infinity" not in line
        assert json.loads(line) == {"device_id": "d", "x": None, "y": None}

    def test_every_message_ends_with_a_newline(self):
        assert encode({"a": 1}).endswith(b"\n")


def test_control_messages_are_distinguishable_from_data():
    assert is_control({"type": "subscribe"})
    assert not is_control({"device_id": "eeg", "Fp1": 1.0})


def test_now_ms_is_unix_milliseconds():
    assert 1_600_000_000_000 < now_ms() < 3_000_000_000_000
