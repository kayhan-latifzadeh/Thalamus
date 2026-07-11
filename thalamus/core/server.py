"""Thalamus Core: the TCP server (paper §2.1).

Two listeners — one for devices, one for clients — over plain TCP with JSON-line
framing, exactly as the paper specifies, so that "any client that can open a
socket connection can communicate with Thalamus, regardless of the operating
system or programming languages used" stays true.

The implementation is asyncio rather than a thread per connection. The reason is
not fashion: the hub's job is to fan one sample out to many subscribers, which
means every connection touches shared routing state. With threads that state needs
locking on the hot path — and the original implementation's ``clients`` dict was
in fact mutated from several threads with no lock at all. On an event loop the
routing table is only ever touched by one task at a time, so the races are gone by
construction rather than by care.

Slow clients are the other thing an event loop gets right. Each client has a
bounded queue; if it cannot keep up, *its* queue drops its oldest samples and the
drop is counted and logged. A slow client degrades only itself, and can never
stall the device stream or any other client — which in the original code it could,
since a blocking ``sendall`` to a wedged client held up the device's whole read
loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Any, Dict, List, Optional, Set

from ..processing import PipelineSpecError, available_stages
from ..protocol import (
    DEFAULT_CLIENT_PORT,
    DEFAULT_DEVICE_PORT,
    LineDecoder,
    ProtocolError,
    Sample,
    encode,
    is_control,
    now_ms,
)
from .router import Router

logger = logging.getLogger(__name__)

#: How many messages may pile up for a client before its oldest are dropped.
#: 2000 samples is ~8 s of 250 Hz EEG: long enough to ride out a GC pause or a
#: slow plot redraw, short enough that a truly stuck client is noticed quickly.
DEFAULT_QUEUE_SIZE = 2000

#: How often synchronizers are nudged so a dead device cannot hold frames forever.
TICK_INTERVAL_S = 0.1


class ClientConnection:
    """One connected client, and the queue feeding it.

    Implements the :class:`~thalamus.core.router.Sink` protocol.
    """

    def __init__(
        self,
        client_id: str,
        writer: asyncio.StreamWriter,
        *,
        queue_size: int = DEFAULT_QUEUE_SIZE,
    ) -> None:
        self.id = client_id
        self.writer = writer
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=queue_size)
        self.dropped = 0
        self.sent = 0
        self._closed = False
        self._loop = asyncio.get_running_loop()

    def send(self, message: Any) -> None:
        """Queue one message. Never blocks; drops the oldest if the client lags."""
        if self._closed:
            return
        try:
            self.queue.put_nowait(message)
        except asyncio.QueueFull:
            # Drop the *oldest*: for a live signal, the newest sample is the one
            # that matters, and a client that is behind is better served by fresh
            # data with a gap than by stale data with no gap.
            with contextlib.suppress(asyncio.QueueEmpty):
                self.queue.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                self.queue.put_nowait(message)

            self.dropped += 1
            if self.dropped == 1 or self.dropped % 1000 == 0:
                logger.warning(
                    "client %s cannot keep up: %d sample(s) dropped. "
                    "Consider subscribing to fewer channels, or adding a filter "
                    "stage to the subscription so the Core downsamples for you.",
                    self.id,
                    self.dropped,
                )

    def send_later(self, delay_ms: float, message: Any) -> None:
        """Queue one message after ``delay_ms``. Order is preserved across calls."""
        if self._closed:
            return
        self._loop.call_later(delay_ms / 1000.0, self.send, message)

    async def pump(self) -> None:
        """Drain the queue to the socket until the connection closes."""
        try:
            while True:
                message = await self.queue.get()
                self.writer.write(encode(message))
                await self.writer.drain()
                self.sent += 1
        except (ConnectionError, asyncio.CancelledError):
            raise
        except Exception:
            logger.exception("client %s: write loop failed", self.id)

    def close(self) -> None:
        self._closed = True


class ThalamusCore:
    """The hub. Start it, and it accepts devices on one port and clients on another.

    Usable three ways, which between them cover every entry point in the toolkit::

        await core.serve_forever()          # the `thalamus serve` CLI
        async with core:  ...               # embedded in a study run
        core.run()                          # blocking, from a plain script
    """

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        device_port: int = DEFAULT_DEVICE_PORT,
        client_port: int = DEFAULT_CLIENT_PORT,
        queue_size: int = DEFAULT_QUEUE_SIZE,
    ) -> None:
        self.host = host
        self.device_port = device_port
        self.client_port = client_port
        self.queue_size = queue_size

        self.router = Router()
        self._servers: List[asyncio.AbstractServer] = []
        self._tasks: Set[asyncio.Task] = set()
        self._client_seq = 0
        self._ready = asyncio.Event()

    # -- lifecycle -------------------------------------------------------------

    async def start(self) -> None:
        """Bind both ports and begin accepting. Returns once listening."""
        device_server = await asyncio.start_server(
            self._tracked(self._handle_device), self.host, self.device_port, reuse_address=True
        )
        client_server = await asyncio.start_server(
            self._tracked(self._handle_client), self.host, self.client_port, reuse_address=True
        )
        self._servers = [device_server, client_server]

        # Port 0 means "any free port", which the tests rely on; read back what we
        # actually got so callers can connect to it.
        self.device_port = device_server.sockets[0].getsockname()[1]
        self.client_port = client_server.sockets[0].getsockname()[1]

        self._spawn(self._tick_loop())
        self._ready.set()
        logger.info(
            "Thalamus Core listening: devices on %s:%d, clients on %s:%d",
            self.host,
            self.device_port,
            self.host,
            self.client_port,
        )

    async def stop(self) -> None:
        """Close both ports and cancel every connection task."""
        for server in self._servers:
            server.close()
        for server in self._servers:
            with contextlib.suppress(Exception):
                await server.wait_closed()
        self._servers.clear()

        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self._ready.clear()
        logger.info("Thalamus Core stopped")

    async def serve_forever(self) -> None:
        await self.start()
        try:
            await asyncio.Event().wait()  # until cancelled
        finally:
            await self.stop()

    async def __aenter__(self) -> ThalamusCore:
        await self.start()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.stop()

    def run(self) -> None:
        """Blocking entry point, for scripts that are not already async."""
        try:
            asyncio.run(self.serve_forever())
        except KeyboardInterrupt:
            logger.info("interrupted")

    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def _tracked(self, handler):
        """Wrap a connection handler so :meth:`stop` can actually cancel it.

        ``asyncio.start_server`` spawns each connection callback in a task it owns and
        does not tell us about, so without this the handlers are invisible to ``stop``:
        the event loop shuts down, the tasks are garbage-collected mid-await, and
        asyncio prints "Task was destroyed but it is pending!". Worse than the noise,
        their ``finally`` blocks — the ones that mark a device disconnected and tear its
        subscriptions down — are not guaranteed to run.

        Registering the task the moment it starts puts it back under ``stop``'s control,
        so a shutdown cancels every live connection and waits for its cleanup.
        """

        async def wrapper(reader, writer):
            task = asyncio.current_task()
            if task is not None:
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
            await handler(reader, writer)

        return wrapper

    async def _tick_loop(self) -> None:
        while True:
            await asyncio.sleep(TICK_INTERVAL_S)
            try:
                self.router.tick()
            except Exception:
                logger.exception("tick failed")

    # -- devices ---------------------------------------------------------------

    async def _handle_device(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        peer = _peer_name(writer)
        decoder = LineDecoder()
        device_ids: Set[str] = set()
        logger.info("device connection from %s", peer)

        try:
            while True:
                chunk = await reader.read(65536)
                if not chunk:
                    break

                try:
                    messages = list(decoder.feed(chunk))
                except ProtocolError as exc:
                    logger.warning("device %s sent a bad line: %s", peer, exc)
                    continue

                arrival = now_ms()
                for obj in messages:
                    self._handle_device_message(obj, device_ids, arrival, peer)
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("device %s: read loop failed", peer)
        finally:
            for device_id in device_ids:
                self.router.device_disconnected(device_id)
            _close(writer)
            logger.info("device connection closed: %s (%s)", peer, ", ".join(device_ids) or "none")

    def _handle_device_message(
        self, obj: Dict[str, Any], device_ids: Set[str], arrival: int, peer: str
    ) -> None:
        if is_control(obj):
            kind = obj.get("type")
            device_id = obj.get("device_id")

            if kind == "hello" and device_id:
                device_ids.add(device_id)
                metadata = {k: v for k, v in obj.items() if k not in ("type", "device_id")}
                self.router.device_connected(device_id, metadata)
            elif kind == "event":
                # A marker (stimulus onset, trial start). It belongs to no single
                # signal, so every client gets it — that is what makes it usable as
                # a common time reference for slicing all of them.
                self.router.broadcast({**obj, "timestamp": obj.get("timestamp", arrival)})
            else:
                logger.debug("device %s sent an unhandled control message: %s", peer, kind)
            return

        try:
            sample = Sample.from_wire(obj, default_timestamp=arrival)
        except ProtocolError as exc:
            logger.warning("device %s sent an unusable sample: %s", peer, exc)
            return

        # A device that never says hello is fine; the first sample announces it.
        if sample.device_id not in device_ids:
            device_ids.add(sample.device_id)
            self.router.device_connected(sample.device_id)

        self.router.route(sample)

    # -- clients ---------------------------------------------------------------

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self._client_seq += 1
        client = ClientConnection(
            f"client-{self._client_seq}@{_peer_name(writer)}", writer, queue_size=self.queue_size
        )
        decoder = LineDecoder()
        self.router.add_sink(client)
        pump = self._spawn(client.pump())

        # Tell the client what it can ask for, rather than making it guess. This is
        # §2.3's "the Core will provide the device with a list of available
        # signals": you cannot subscribe to a channel whose name you do not know.
        client.send(
            {
                "type": "welcome",
                "client_id": client.id,
                "devices": [info.to_wire() for info in self.router.devices.values()],
                "stages": available_stages(),
            }
        )
        logger.info("client connected: %s", client.id)

        try:
            while True:
                chunk = await reader.read(65536)
                if not chunk:
                    break

                try:
                    messages = list(decoder.feed(chunk))
                except ProtocolError as exc:
                    client.send({"type": "error", "message": str(exc)})
                    continue

                legacy = _legacy_subscription(decoder)
                if legacy is not None:
                    messages.append(legacy)

                for obj in messages:
                    self._handle_client_message(client, obj)
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("client %s: read loop failed", client.id)
        finally:
            client.close()
            pump.cancel()
            self.router.remove_sink(client)
            _close(writer)
            logger.info(
                "client disconnected: %s (%d sent, %d dropped)",
                client.id,
                client.sent,
                client.dropped,
            )

    def _handle_client_message(self, client: ClientConnection, obj: Dict[str, Any]) -> None:
        try:
            # The pre-1.0 form: {"subscribe": ["eeg", "eye"]}. Still supported.
            if "subscribe" in obj and "type" not in obj:
                obj = {"type": "subscribe", "devices": obj["subscribe"]}

            if not is_control(obj):
                # No type and no 'subscribe' key: it must be data. A client that
                # sends samples *is* a recording device — Figure 1's client #1,
                # which doubles as Recording Device #5.
                sample = Sample.from_wire(obj, default_timestamp=now_ms())
                info = self.router.device_info(sample.device_id)
                if not info.connected:
                    self.router.device_connected(sample.device_id, {"source": "client"})
                self.router.route(sample)
                return

            handler = {
                "subscribe": self._on_subscribe,
                "unsubscribe": self._on_unsubscribe,
                "list_devices": self._on_list_devices,
                "event": self._on_event,
                "ping": self._on_ping,
            }.get(obj.get("type"))

            if handler is None:
                client.send(
                    {"type": "error", "message": f"unknown message type {obj.get('type')!r}"}
                )
                return

            handler(client, obj)

        except (PipelineSpecError, ProtocolError, ValueError) as exc:
            logger.info("client %s sent a rejected request: %s", client.id, exc)
            client.send({"type": "error", "message": str(exc)})

    def _on_subscribe(self, client: ClientConnection, obj: Dict[str, Any]) -> None:
        subscribed: List[str] = []

        for entry in obj.get("devices") or ():
            # Accept both "eeg" and {"device_id": "eeg", "channels": [...], ...}.
            spec = {"device_id": entry} if isinstance(entry, str) else dict(entry)
            device_id = spec.pop("device_id", None)
            if not device_id:
                raise ProtocolError(f"subscription entry has no device_id: {entry!r}")

            self.router.subscribe(
                client,
                device_id,
                channels=spec.pop("channels", None),
                pipeline=spec.pop("pipeline", None),
                latency_ms=float(spec.pop("latency_ms", 0.0)),
            )
            subscribed.append(device_id)

        sync = obj.get("sync")
        if sync:
            devices = sync.get("devices") or []
            self.router.subscribe_sync(
                client,
                devices,
                reference=sync.get("reference"),
                tolerance_ms=float(sync.get("tolerance_ms", 20.0)),
                timeout_ms=float(sync.get("timeout_ms", 500.0)),
                latency_ms=float(sync.get("latency_ms", 0.0)),
            )
            subscribed.extend(f"sync({d})" for d in devices)

        if not subscribed:
            raise ProtocolError("a subscription must name at least one device")

        client.send({"type": "subscribed", "devices": subscribed})

    def _on_unsubscribe(self, client: ClientConnection, obj: Dict[str, Any]) -> None:
        devices = obj.get("devices")
        if devices:
            removed = sum(self.router.unsubscribe(client, d) for d in devices)
        else:
            removed = self.router.unsubscribe(client)
        client.send({"type": "unsubscribed", "removed": removed})

    def _on_list_devices(self, client: ClientConnection, _obj: Dict[str, Any]) -> None:
        client.send({"type": "devices", **self.router.snapshot()})

    def _on_event(self, client: ClientConnection, obj: Dict[str, Any]) -> None:
        del client
        self.router.broadcast({**obj, "timestamp": obj.get("timestamp", now_ms())})

    def _on_ping(self, client: ClientConnection, _obj: Dict[str, Any]) -> None:
        client.send({"type": "pong", "timestamp": now_ms()})


def _legacy_subscription(decoder: LineDecoder) -> Optional[Dict[str, Any]]:
    """Recover a subscription that arrived without a trailing newline.

    Clients written against the original Thalamus did exactly this, and then went
    quiet — so a strict line decoder would wait forever for a newline that is never
    coming, and the client would silently receive nothing. We accept the
    unterminated buffer if, and only if, it parses as a complete legacy
    subscription object. Anything else is left in the buffer for the next chunk.
    """
    pending = decoder.pending().strip()
    if not pending:
        return None
    try:
        obj = json.loads(pending)
    except ValueError:
        return None  # a genuinely partial line: wait for the rest
    if isinstance(obj, dict) and "subscribe" in obj and "type" not in obj:
        decoder.clear()
        return obj
    return None


def _peer_name(writer: asyncio.StreamWriter) -> str:
    peer = writer.get_extra_info("peername")
    if isinstance(peer, tuple) and len(peer) >= 2:
        return f"{peer[0]}:{peer[1]}"
    return str(peer)


def _close(writer: asyncio.StreamWriter) -> None:
    with contextlib.suppress(Exception):
        writer.close()
