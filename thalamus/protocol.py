"""Wire protocol for Thalamus.

Every message on the wire is a single JSON object followed by a newline
("JSON Lines"). This keeps the protocol language-agnostic: any client that can
open a TCP socket and split on ``\\n`` can talk to Thalamus, which is the whole
point of the design (Latifzadeh & Leiva, UMAP '25, §2.3).

Two kinds of messages travel over a connection:

*Data samples* carry sensor readings. On the wire they are flat objects::

    {"device_id": "eeg", "timestamp": 1690535469479, "Fp1": 12.3, "Fp2": -4.1}

Every key other than the reserved ones is a channel. This is the format the
original Thalamus release used, and it is preserved exactly so that existing
devices and clients keep working.

*Control messages* carry everything else and are distinguished by a ``type``
field (``hello``, ``subscribe``, ``devices``, ``event``, ``error``, ...). A data
sample never has a ``type`` field, so the two are unambiguous.

Timestamps are integer milliseconds since the Unix epoch, UTC. A device that
omits one gets stamped on arrival at the Core, but it should not rely on that:
cross-device synchronization is only as good as the timestamps the devices
supply.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

DEFAULT_DEVICE_PORT = 9000
DEFAULT_CLIENT_PORT = 9001

#: Keys that carry metadata rather than a channel reading.
RESERVED_KEYS = frozenset({"device_id", "timestamp", "type", "seq"})

#: Values a device may use to say "no reading here". Eye trackers write "NA"
#: when the participant looks off-screen; other devices use nulls or NaNs.
MISSING_TOKENS = frozenset({"NA", "N/A", "na", "nan", "NaN", "", "-", "null", "None"})


class _Missing:
    """Sentinel for a channel that has no valid reading in this sample.

    Distinct from ``0`` and from ``None``: a device that genuinely measured zero
    must not be confused with one that measured nothing. Stages test for it with
    ``value is MISSING``.
    """

    _instance: Optional[_Missing] = None

    def __new__(cls) -> _Missing:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "MISSING"

    def __bool__(self) -> bool:
        return False


MISSING = _Missing()


def now_ms() -> int:
    """Current UTC time in milliseconds since the Unix epoch."""
    return int(time.time() * 1000)


def normalize_value(value: Any) -> Any:
    """Map a raw wire value onto either a number, a string, or :data:`MISSING`."""
    if value is None:
        return MISSING
    if isinstance(value, str):
        return MISSING if value.strip() in MISSING_TOKENS else value
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return MISSING
    return value


@dataclass
class Sample:
    """One reading from one device at one point in time.

    ``data`` maps channel name to value. A value is a number, a string, or
    :data:`MISSING`.
    """

    device_id: str
    timestamp: int
    data: Dict[str, Any] = field(default_factory=dict)

    @property
    def channels(self) -> List[str]:
        return list(self.data)

    def copy(self) -> Sample:
        """A deep-enough copy: stages mutate ``data``, never the values in it."""
        return Sample(self.device_id, self.timestamp, dict(self.data))

    def select(self, channels: Optional[List[str]]) -> Sample:
        """Restrict to ``channels``. ``None`` means "keep everything"."""
        if channels is None:
            return self
        keep = set(channels)
        return Sample(
            self.device_id,
            self.timestamp,
            {k: v for k, v in self.data.items() if k in keep},
        )

    @classmethod
    def from_wire(cls, obj: Dict[str, Any], *, default_timestamp: Optional[int] = None) -> Sample:
        """Parse a flat wire object into a Sample.

        Raises :class:`ProtocolError` if there is no ``device_id``; falls back to
        ``default_timestamp`` (normally arrival time) if the device sent none.
        """
        device_id = obj.get("device_id")
        if not device_id or not isinstance(device_id, str):
            raise ProtocolError("sample is missing a string 'device_id'")

        raw_ts = obj.get("timestamp")
        if raw_ts is None:
            timestamp = default_timestamp if default_timestamp is not None else now_ms()
        else:
            try:
                timestamp = int(float(raw_ts))
            except (TypeError, ValueError) as exc:
                raise ProtocolError(f"'timestamp' is not a number: {raw_ts!r}") from exc

        data = {k: normalize_value(v) for k, v in obj.items() if k not in RESERVED_KEYS}
        return cls(device_id=device_id, timestamp=timestamp, data=data)

    def to_wire(self, *, missing_marker: Any = None) -> Dict[str, Any]:
        """Render back to the flat wire form.

        :data:`MISSING` becomes ``missing_marker`` (JSON ``null`` by default, so
        that a client can tell a gap from a real zero).
        """
        obj: Dict[str, Any] = {"device_id": self.device_id, "timestamp": self.timestamp}
        for key, value in self.data.items():
            obj[key] = missing_marker if value is MISSING else value
        return obj


class ProtocolError(ValueError):
    """A message could not be understood."""


def encode(obj: Any) -> bytes:
    """Serialize one message as a JSON line, including the trailing newline.

    Guarantees the output is *valid* JSON, which takes a little care: Python's
    ``json`` cheerfully emits bare ``NaN`` and ``Infinity`` tokens, which are not
    JSON and which every non-Python client rejects. A single NaN out of a filter
    would otherwise break every MATLAB, JavaScript, and Java client in the study.
    So we forbid them, and on the rare occasion one appears we scrub it to
    ``null`` — the same thing a gap serializes to, which is what a NaN means.
    """
    if isinstance(obj, Sample):
        obj = obj.to_wire()
    try:
        text = json.dumps(obj, separators=(",", ":"), default=_json_default, allow_nan=False)
    except ValueError:
        # allow_nan=False raised: there is a NaN or an Inf in here somewhere. This is
        # the cold path — pay for the walk only when it is actually needed.
        text = json.dumps(_scrub(obj), separators=(",", ":"), default=_json_default)
    return (text + "\n").encode("utf-8")


def _scrub(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_scrub(v) for v in value]
    return value


def _json_default(value: Any) -> Any:
    if value is MISSING:
        return None
    # numpy scalars and anything else with a scalar view; keeps stages free to
    # return numpy types without every one of them having to cast back.
    item = getattr(value, "item", None)
    if callable(item):
        scalar = item()
        if isinstance(scalar, float) and (math.isnan(scalar) or math.isinf(scalar)):
            return None
        return scalar
    raise TypeError(f"cannot serialize {type(value).__name__} to JSON")


class LineDecoder:
    """Reassembles newline-delimited JSON from arbitrary TCP chunks.

    TCP gives you a byte stream, not messages: a single ``recv`` can return half
    a line, or three lines and a bit. The original implementation assumed one
    ``recv`` per message, which silently corrupted any subscription longer than
    the read buffer. Feed every chunk through here instead.
    """

    def __init__(self, max_line_bytes: int = 16 * 1024 * 1024) -> None:
        self._buffer = bytearray()
        self._max_line_bytes = max_line_bytes

    def feed(self, chunk: bytes) -> Iterator[Dict[str, Any]]:
        """Yield every complete JSON object contained in ``chunk``.

        Malformed lines raise :class:`ProtocolError`; the decoder stays usable,
        so a caller may log and continue with the next line.
        """
        self._buffer.extend(chunk)
        if len(self._buffer) > self._max_line_bytes:
            self._buffer.clear()
            raise ProtocolError(
                f"line exceeded {self._max_line_bytes} bytes without a newline; buffer dropped"
            )

        while True:
            index = self._buffer.find(b"\n")
            if index < 0:
                return
            line = bytes(self._buffer[:index])
            del self._buffer[: index + 1]
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line.decode("utf-8"))
            except (ValueError, UnicodeDecodeError) as exc:
                raise ProtocolError(f"malformed JSON line: {exc}") from exc
            if not isinstance(obj, dict):
                raise ProtocolError(f"expected a JSON object, got {type(obj).__name__}")
            yield obj

    def pending(self) -> bytes:
        """Bytes received but not yet terminated by a newline.

        Only needed for one thing: clients written against the pre-1.0 Thalamus
        sent their subscription with no trailing newline and then never spoke
        again, so a strict line decoder waits for a newline that never comes. The
        client handler peeks here to stay compatible with them. New code should
        always terminate its lines.
        """
        return bytes(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()


def is_control(obj: Dict[str, Any]) -> bool:
    """True if ``obj`` is a control message rather than a data sample."""
    return "type" in obj
