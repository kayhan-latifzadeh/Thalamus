"""The client SDK.

A client is anything that wants to *receive* signals. It does not have to use this
class — the whole point of JSON-over-TCP is that a MATLAB script or a browser can
subscribe just as well — but if you are in Python, this saves you writing the
socket handling again::

    with ThalamusClient() as client:
        client.subscribe("eeg", pipeline=[{"stage": "savgol", "window": 11}])
        for sample in client.stream():
            print(sample.timestamp, sample.data)

A client may also *send* samples, which makes it a recording device as well as a
consumer — the client that also feeds back its own mouse trace or its classifier's
output, for the other clients to see. That is Recording Device #5 in Figure 1.
"""

from __future__ import annotations

import contextlib
import logging
import socket
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

from .protocol import (
    DEFAULT_CLIENT_PORT,
    LineDecoder,
    ProtocolError,
    Sample,
    encode,
    is_control,
    now_ms,
)

logger = logging.getLogger(__name__)

Message = Union[Sample, Dict[str, Any]]


class ThalamusClient:
    """A blocking client connection to Thalamus Core.

    Blocking on purpose: a researcher plotting a signal or feeding a classifier
    wants a ``for`` loop, not an event loop. The Core is async so that it can serve
    many clients at once; a client only has itself to serve.
    """

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = DEFAULT_CLIENT_PORT,
        timeout: Optional[float] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self._socket: Optional[socket.socket] = None
        self._decoder = LineDecoder()
        self._inbox: List[Dict[str, Any]] = []
        self.welcome: Optional[Dict[str, Any]] = None

    # -- connection ------------------------------------------------------------

    def connect(self) -> ThalamusClient:
        self._socket = socket.create_connection((self.host, self.port), timeout=self.timeout)
        logger.info("connected to Thalamus at %s:%d", self.host, self.port)
        # The Core greets every client with the devices it knows about and the
        # stages it can run, so a client never has to guess what is on offer.
        self.welcome = self._await_control("welcome")
        return self

    def close(self) -> None:
        if self._socket is not None:
            with contextlib.suppress(OSError):
                self._socket.shutdown(socket.SHUT_RDWR)
            self._socket.close()
            self._socket = None

    def __enter__(self) -> ThalamusClient:
        return self.connect()

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # -- requests --------------------------------------------------------------

    def subscribe(
        self,
        device_id: str,
        *,
        channels: Optional[Sequence[str]] = None,
        pipeline: Optional[Sequence[Dict[str, Any]]] = None,
        latency_ms: float = 0.0,
    ) -> None:
        """Ask for one device's stream, optionally processed and channel-filtered.

        ``pipeline`` is run *in the Core*, once, and shared with any other client
        that asked for the same thing — so filtering in ten clients costs what
        filtering in one does. ``latency_ms`` delays delivery to this client only,
        which is how you simulate a slow link to one participant without touching
        anyone else's stream.
        """
        entry: Dict[str, Any] = {"device_id": device_id}
        if channels:
            entry["channels"] = list(channels)
        if pipeline:
            entry["pipeline"] = list(pipeline)
        if latency_ms:
            entry["latency_ms"] = latency_ms

        self._send({"type": "subscribe", "devices": [entry]})
        self._await_control("subscribed")

    def subscribe_synced(
        self,
        devices: Sequence[str],
        *,
        reference: Optional[str] = None,
        tolerance_ms: float = 20.0,
        timeout_ms: float = 500.0,
    ) -> None:
        """Ask for several devices aligned to each other, as time-stamped frames.

        Instead of samples you now receive ``{"type": "frame", ...}`` messages, each
        holding one reading per device taken at (nearly) the same instant. Iterate
        them with :meth:`frames`.
        """
        self._send(
            {
                "type": "subscribe",
                "sync": {
                    "devices": list(devices),
                    "reference": reference,
                    "tolerance_ms": tolerance_ms,
                    "timeout_ms": timeout_ms,
                },
            }
        )
        self._await_control("subscribed")

    def unsubscribe(self, *devices: str) -> int:
        self._send({"type": "unsubscribe", "devices": list(devices) or None})
        return int(self._await_control("unsubscribed").get("removed", 0))

    def devices(self) -> List[Dict[str, Any]]:
        """Ask the Core what is connected right now, and at what measured rate."""
        self._send({"type": "list_devices"})
        return self._await_control("devices").get("devices", [])

    def send_event(self, label: str, **fields: Any) -> None:
        """Broadcast a marker (stimulus onset, trial start) to every client."""
        self._send({"type": "event", "label": label, "timestamp": now_ms(), **fields})

    def send_sample(
        self, device_id: str, data: Dict[str, Any], timestamp: Optional[int] = None
    ) -> None:
        """Push a sample *into* Thalamus, acting as a recording device.

        The mouse trace of one participant, shared with the others; a classifier's
        live output, fed back for everyone to consume. The Core routes it exactly as
        it routes a real device's stream.
        """
        self._send({"device_id": device_id, "timestamp": timestamp or now_ms(), **data})

    # -- receiving -------------------------------------------------------------

    def messages(self) -> Iterator[Message]:
        """Every message: data as :class:`~thalamus.protocol.Sample`, the rest as dicts.

        Control messages matter more than they look. ``device_disconnected`` is how
        you find out a sensor died mid-run, which is exactly the corner case the
        study should be prepared for.
        """
        while True:
            for obj in self._drain():
                yield self._interpret(obj)

            chunk = self._recv()
            if not chunk:
                return
            try:
                for obj in self._decoder.feed(chunk):
                    yield self._interpret(obj)
            except ProtocolError as exc:
                logger.warning("dropping a malformed message: %s", exc)

    def stream(self) -> Iterator[Sample]:
        """Only the data samples. The common case."""
        for message in self.messages():
            if isinstance(message, Sample):
                yield message

    def frames(self) -> Iterator[Dict[str, Any]]:
        """Only the synchronized frames, for a client that called :meth:`subscribe_synced`."""
        for message in self.messages():
            if isinstance(message, dict) and message.get("type") == "frame":
                yield message

    def __iter__(self) -> Iterator[Message]:
        return self.messages()

    # -- internals -------------------------------------------------------------

    def _interpret(self, obj: Dict[str, Any]) -> Message:
        if is_control(obj):
            return obj
        try:
            return Sample.from_wire(obj)
        except ProtocolError:
            return obj

    def _send(self, obj: Dict[str, Any]) -> None:
        if self._socket is None:
            raise RuntimeError("not connected: call connect(), or use `with ThalamusClient() as c`")
        self._socket.sendall(encode(obj))

    def _recv(self) -> bytes:
        if self._socket is None:
            raise RuntimeError("not connected")
        return self._socket.recv(65536)

    def _drain(self) -> List[Dict[str, Any]]:
        """Messages that arrived while we were waiting for a control reply."""
        pending, self._inbox = self._inbox, []
        return pending

    def _await_control(self, kind: str, *, limit: int = 100_000) -> Dict[str, Any]:
        """Block until a control message of type ``kind`` arrives.

        Data samples that arrive in the meantime are set aside rather than dropped,
        so subscribing to a second device does not silently lose a handful of the
        first device's samples.
        """
        for _ in range(limit):
            chunk = self._recv()
            if not chunk:
                raise ConnectionError("Thalamus closed the connection")

            for obj in self._decoder.feed(chunk):
                if obj.get("type") == kind:
                    return obj
                if obj.get("type") == "error":
                    raise ThalamusError(obj.get("message", "the Core rejected the request"))
                self._inbox.append(obj)

        raise ConnectionError(f"no {kind!r} reply from Thalamus")


class ThalamusError(RuntimeError):
    """The Core rejected a request — a bad pipeline spec, an unknown message type."""
