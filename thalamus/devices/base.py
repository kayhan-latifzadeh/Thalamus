"""The recording-device SDK.

The socket loop — connect, serialize, ``sendall``, wait, handle ``BrokenPipeError`` —
is written once, here, rather than in every device. A device supplies
:meth:`RecordingDevice.samples`,
a generator of readings, and gets connection handling, reconnection, hello
announcement, UTC timestamping, and drift-free pacing for free. A new device is
about ten lines::

    class MyDevice(RecordingDevice):
        def samples(self):
            while True:
                yield {"temperature": read_sensor()}

    MyDevice("thermometer", rate=10).run()
"""

from __future__ import annotations

import contextlib
import json
import logging
import socket
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Union

from ..protocol import DEFAULT_DEVICE_PORT, Sample, encode, now_ms

logger = logging.getLogger(__name__)

Reading = Union[Dict[str, Any], Sample]


class Pacer:
    """Emits at a target rate without accumulating drift.

    The obvious approach, ``time.sleep(1 / rate)`` between samples, is incorrect: the
    sleep excludes the time spent producing and sending the sample, so a device
    declared at 250 Hz emits at 250 Hz *minus its own overhead*, and the error
    compounds. Over a ten-minute run a nominally 250 Hz stream can fall thousands of
    samples short, and a study synchronizing it against a 150 Hz stream, whose
    overhead differs, observes the two drifting apart with no evident cause.

    So we schedule against an absolute origin instead: sample *n* is due at
    ``origin + offset(n)``, and we sleep until then. Overhead is absorbed rather
    than accumulated, and the emitted rate is the requested one.
    """

    def __init__(self, *, warn_after_ms: float = 250.0) -> None:
        self.origin: Optional[float] = None
        self.warn_after_ms = warn_after_ms
        self._warned = False
        self.max_lag_ms = 0.0

    def wait_until(self, offset_ms: float) -> None:
        """Block until ``offset_ms`` after the first call."""
        if self.origin is None:
            self.origin = time.monotonic()

        due = self.origin + offset_ms / 1000.0
        remaining = due - time.monotonic()

        if remaining > 0:
            time.sleep(remaining)
            return

        # Behind schedule: emit immediately rather than sleeping a negative amount,
        # but say so, because a device that cannot keep up is a finding in itself —
        # it is device stress-testing reporting a result.
        lag_ms = -remaining * 1000.0
        self.max_lag_ms = max(self.max_lag_ms, lag_ms)
        if lag_ms > self.warn_after_ms and not self._warned:
            self._warned = True
            logger.warning(
                "falling behind schedule by %.0f ms: this machine cannot sustain the "
                "requested rate. Lower the rate, or reduce per-sample work.",
                lag_ms,
            )

    def reset(self) -> None:
        self.origin = None
        self._warned = False


class RecordingDevice(ABC):
    """Base class for anything that streams samples into Thalamus.

    Subclasses implement :meth:`samples`. Everything else — the socket, the
    reconnection, the timestamps, the pacing — is handled here.

    Pacing is driven by whichever of these the subclass provides:

    * ``rate`` (Hz): sample *n* is emitted at ``n / rate`` seconds after the start.
      Use this for a device with a fixed sampling frequency.
    * a ``timestamp`` on each yielded reading: the stream is replayed honouring the
      *original* intervals between those timestamps, divided by ``speed``. Use this
      to replay a recording with its real jitter intact — which is the more faithful
      simulation, since no real device is perfectly periodic.

    If neither is given, samples go out as fast as the generator produces them.
    """

    def __init__(
        self,
        device_id: str,
        *,
        host: str = "localhost",
        port: int = DEFAULT_DEVICE_PORT,
        rate: Optional[float] = None,
        speed: float = 1.0,
        rebase_timestamps: bool = True,
        reconnect: bool = True,
        reconnect_delay_s: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if rate is not None and rate <= 0:
            raise ValueError(f"rate must be > 0, got {rate}")
        if speed <= 0:
            raise ValueError(f"speed must be > 0, got {speed}")

        self.device_id = device_id
        self.host = host
        self.port = port
        self.rate = rate
        self.speed = speed
        self.rebase_timestamps = rebase_timestamps
        self.reconnect = reconnect
        self.reconnect_delay_s = reconnect_delay_s
        self.metadata = dict(metadata or {})

        self.sent = 0
        self._stop = threading.Event()
        self._pacer = Pacer()
        self._socket: Optional[socket.socket] = None

    # -- what a subclass provides ---------------------------------------------

    @abstractmethod
    def samples(self) -> Iterator[Reading]:
        """Yield readings, either as ``{"channel": value}`` dicts or as Samples.

        A dict may carry a ``timestamp`` (UTC ms); if it does not, one is stamped
        at emission. Return (or raise ``StopIteration``) to end the stream.
        """

    def describe(self) -> Dict[str, Any]:
        """Metadata announced to the Core on connect. Override to add channels etc."""
        info: Dict[str, Any] = dict(self.metadata)
        if self.rate is not None:
            info.setdefault("sample_rate", self.rate)
        return info

    # -- the loop, written once ------------------------------------------------

    def run(self) -> None:
        """Stream until the source is exhausted or :meth:`stop` is called."""
        while not self._stop.is_set():
            try:
                self._stream_once()
                return  # the source ended cleanly; nothing to reconnect for
            except (ConnectionError, OSError) as exc:
                if self._stop.is_set():
                    # We asked it to stop, and the socket broke because of that. A
                    # requested shutdown is not a failure, and must not be raised out
                    # of the device thread as one.
                    logger.info("[%s] stopped after %d samples", self.device_id, self.sent)
                    return
                if not self.reconnect:
                    logger.error("[%s] connection failed: %s", self.device_id, exc)
                    raise
                logger.warning(
                    "[%s] connection lost (%s); retrying in %.1fs",
                    self.device_id,
                    exc,
                    self.reconnect_delay_s,
                )
                # Reset the pacer: after a gap, resuming the old schedule would try
                # to "catch up" by dumping every sample missed during the outage.
                self._pacer.reset()
                self._stop.wait(self.reconnect_delay_s)

    def _stream_once(self) -> None:
        with socket.create_connection((self.host, self.port), timeout=10) as sock:
            self._socket = sock
            sock.settimeout(None)
            logger.info("[%s] connected to %s:%d", self.device_id, self.host, self.port)

            self._send(sock, {"type": "hello", "device_id": self.device_id, **self.describe()})

            origin_timestamp: Optional[int] = None
            wall_origin = now_ms()

            for index, reading in enumerate(self.samples()):
                if self._stop.is_set():
                    break

                # Whether the *source* supplied a timestamp decides everything below,
                # so establish it before from_wire() papers over the difference by
                # stamping one in.
                recorded = self._has_own_timestamp(reading)
                sample = self._as_sample(reading)

                if index == 0:
                    origin_timestamp = sample.timestamp

                # Pacing: an explicit rate wins; otherwise a recording is replayed on
                # its own timing (which preserves its real jitter — no actual device
                # is perfectly periodic, and code that assumes it is should fail here
                # rather than in the study); otherwise the source sets the pace.
                if self.rate is not None:
                    self._pacer.wait_until(index * 1000.0 / self.rate)
                elif recorded:
                    self._pacer.wait_until((sample.timestamp - origin_timestamp) / self.speed)

                # Timestamps: a recorded one is the ground truth and is kept, but
                # rebased onto now. A file recorded in 2023 must not stream 2023
                # timestamps — the Core aligns devices against each other on UTC, so a
                # replay has to be moved onto the live clock with its internal
                # intervals intact. A reading with no timestamp of its own was already
                # stamped at emission and is left alone.
                if recorded and self.rebase_timestamps:
                    sample.timestamp = wall_origin + int(
                        (sample.timestamp - origin_timestamp) / self.speed
                    )

                sample.device_id = self.device_id
                self._send(sock, sample)
                self.sent += 1

            logger.info("[%s] source exhausted after %d samples", self.device_id, self.sent)

    @staticmethod
    def _has_own_timestamp(reading: Reading) -> bool:
        if isinstance(reading, Sample):
            return True
        return isinstance(reading, dict) and reading.get("timestamp") is not None

    def _as_sample(self, reading: Reading) -> Sample:
        if isinstance(reading, Sample):
            return reading
        if not isinstance(reading, dict):
            raise TypeError(
                f"[{self.device_id}] samples() must yield a dict or a Sample, "
                f"got {type(reading).__name__}"
            )
        return Sample.from_wire({"device_id": self.device_id, **reading})

    def _send(self, sock: socket.socket, message: Any) -> None:
        sock.sendall(encode(message))

    def stop(self) -> None:
        """Ask the stream to end. Safe to call from another thread."""
        self._stop.set()
        if self._socket is not None:
            with contextlib.suppress(OSError):
                self._socket.shutdown(socket.SHUT_RDWR)

    def run_in_thread(self) -> threading.Thread:
        """Run in a daemon thread, so one process can host several devices."""
        thread = threading.Thread(target=self.run, name=f"device-{self.device_id}", daemon=True)
        thread.start()
        return thread

    def send_event(self, label: str, **fields: Any) -> None:
        """Mark an instant on the shared timeline (stimulus onset, trial start).

        Events are broadcast to every client, which makes them the common reference
        for slicing all the streams at once — the ``EVENT [Type A]`` markers in
        one timeline.
        """
        if self._socket is None:
            raise RuntimeError(f"[{self.device_id}] cannot send an event before connecting")
        self._send(
            self._socket,
            {
                "type": "event",
                "device_id": self.device_id,
                "label": label,
                "timestamp": now_ms(),
                **fields,
            },
        )


class CallableDevice(RecordingDevice):
    """Wraps a plain generator function, for when a class is more ceremony than needed.

    CallableDevice("mouse", lambda: ({"x": p.x, "y": p.y} for p in track()), rate=60).run()
    """

    def __init__(self, device_id: str, source, **kwargs: Any) -> None:
        super().__init__(device_id, **kwargs)
        self._source = source

    def samples(self) -> Iterator[Reading]:
        return iter(self._source())


def read_channels(path: str, limit: int = 1) -> List[str]:
    """Peek at a CSV or JSONL file and report its channel names, without loading it."""
    import csv

    if path.endswith(".jsonl") or path.endswith(".ndjson"):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    return [k for k in json.loads(line) if k not in ("device_id", "timestamp")]
        return []

    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        del limit
        return [h for h in header if h not in ("device_id", "timestamp")]
