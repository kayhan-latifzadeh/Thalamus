"""Replay a recording as if it were a live device (paper §2.2).

This is the "simulated device" of the paper: a public dataset injected through a
controller that parses it and formats it for the wire. Point it at a CSV or JSONL
file and it becomes an EEG cap, an eye tracker, or anything else, at the right
rate and with the right timestamps — no hardware, no purchase, no participant.

The file is streamed lazily, row by row. The original implementation loaded the
whole recording into a pandas DataFrame up front, which is both a heavyweight
dependency and a poor fit: a one-hour 62-channel EEG recording at 250 Hz is a
gigabyte of floats that you are going to emit one row at a time anyway.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

from .base import Reading, RecordingDevice

logger = logging.getLogger(__name__)

#: Column names that are taken to hold the sample's time, in order of preference.
TIMESTAMP_COLUMNS = ("timestamp", "time", "ts", "unix_ts", "utc", "epoch_ms")


class ReplayDevice(RecordingDevice):
    """Stream a CSV or JSONL recording into Thalamus.

    ``path``
        The recording. ``.csv``/``.tsv`` are read with the header as channel names;
        ``.jsonl``/``.ndjson`` with one JSON object per line.

    ``rate``
        Emit at this many Hz, ignoring whatever timing the file implies. Leave it
        unset and the file's own timestamp column drives the pace, which replays the
        recording's real jitter and gaps rather than an idealized version of them.

    ``timestamp_column``
        Which column holds the time. Auto-detected from :data:`TIMESTAMP_COLUMNS` if
        not given; if there is no such column, samples are stamped on emission and
        you should set ``rate``.

    ``channels``
        Keep only these columns. Handy for pretending a 62-channel SEED recording is
        a 14-channel DREAMER headset — which is precisely the "try before you buy"
        question of §3.2, answered without buying either device.

    ``loop``
        Restart at the top when the file runs out, with timestamps continuing
        forward rather than jumping back. Use it when you need a stream that outlasts
        the recording you have.
    """

    def __init__(
        self,
        device_id: str,
        path: Union[str, Path],
        *,
        timestamp_column: Optional[str] = None,
        channels: Optional[Sequence[str]] = None,
        loop: bool = False,
        max_samples: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(device_id, **kwargs)
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"[{device_id}] no such recording: {self.path}. "
                f"Sample recordings are listed in pre_recorded_files/README.md, or "
                f"generate one with: thalamus make-data"
            )

        self.timestamp_column = timestamp_column
        self.channels = list(channels) if channels else None
        self.loop = loop
        self.max_samples = max_samples
        self._detected_columns: List[str] = []

    def describe(self) -> Dict[str, Any]:
        info = super().describe()
        info.setdefault("source", str(self.path))
        if self.channels:
            info.setdefault("channels", list(self.channels))
        elif self._detected_columns:
            info.setdefault("channels", list(self._detected_columns))
        return info

    def samples(self) -> Iterator[Reading]:
        emitted = 0
        # When looping, keep pushing timestamps forward by one full pass so the
        # stream stays monotonic. A stream whose clock jumps backwards would break
        # every synchronizer downstream, which is not the failure mode we are here
        # to simulate.
        loop_offset = 0
        span = 0

        while True:
            first_ts: Optional[int] = None
            last_ts: Optional[int] = None

            for row in self._rows():
                reading = self._to_reading(row)

                timestamp = reading.get("timestamp")
                if timestamp is not None:
                    if first_ts is None:
                        first_ts = timestamp
                    last_ts = timestamp
                    reading["timestamp"] = timestamp + loop_offset

                yield reading

                emitted += 1
                if self.max_samples is not None and emitted >= self.max_samples:
                    return

            if not self.loop:
                return

            if first_ts is not None and last_ts is not None:
                # One inter-sample interval past the last one, so the seam between
                # passes has the same spacing as everything else.
                interval = (last_ts - first_ts) / max(1, emitted - 1) if emitted > 1 else 0
                span = (last_ts - first_ts) + int(interval)
                loop_offset += span
            logger.info("[%s] end of %s; looping", self.device_id, self.path.name)

    # -- reading the file ------------------------------------------------------

    def _rows(self) -> Iterator[Dict[str, Any]]:
        suffix = self.path.suffix.lower()
        if suffix in (".jsonl", ".ndjson"):
            yield from self._json_rows()
        else:
            yield from self._csv_rows()

    def _csv_rows(self) -> Iterator[Dict[str, Any]]:
        delimiter = "\t" if self.path.suffix.lower() == ".tsv" else ","
        with self.path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            if reader.fieldnames:
                self._detected_columns = [
                    f for f in reader.fieldnames if f != self._timestamp_key(reader.fieldnames)
                ]
            yield from reader

    def _json_rows(self) -> Iterator[Dict[str, Any]]:
        with self.path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except ValueError as exc:
                    logger.warning(
                        "[%s] %s:%d is not valid JSON: %s",
                        self.device_id,
                        self.path.name,
                        number,
                        exc,
                    )

    def _timestamp_key(self, columns: Sequence[str]) -> Optional[str]:
        if self.timestamp_column:
            return self.timestamp_column if self.timestamp_column in columns else None
        lowered = {c.lower(): c for c in columns}
        for candidate in TIMESTAMP_COLUMNS:
            if candidate in lowered:
                return lowered[candidate]
        return None

    def _to_reading(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Turn one file row into a wire reading: numbers parsed, channels selected."""
        timestamp_key = self._timestamp_key(list(row))
        reading: Dict[str, Any] = {}

        for key, value in row.items():
            if key is None or key == "device_id":
                continue
            if key == timestamp_key:
                parsed = _to_number(value)
                if parsed is not None:
                    reading["timestamp"] = int(parsed)
                continue
            if self.channels is not None and key not in self.channels:
                continue
            # Leave unparseable values as-is: "NA" and friends are recognised as
            # gaps by the protocol layer, and a genuinely non-numeric channel (a
            # label, a marker) should survive intact.
            number = _to_number(value)
            reading[key] = value if number is None else number

        return reading


def _to_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    try:
        return float(value.strip())
    except ValueError:
        return None
