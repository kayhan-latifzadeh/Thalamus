"""End-to-end, over real TCP sockets on real ports.

Everything else in the suite tests a layer in isolation. These tests start an actual
Core, connect actual devices and clients to it, and assert on what comes out the far
end — because the bugs that matter in a networked toolkit live in the seams, and a
mock socket has no seams.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import socket

import pytest

from thalamus import SyntheticDevice, ThalamusClient, ThalamusCore
from thalamus.protocol import MISSING, encode


@pytest.fixture
async def core():
    """A Core on two free ports, torn down afterwards."""
    core = ThalamusCore(host="127.0.0.1", device_port=0, client_port=0)
    await core.start()
    yield core
    await core.stop()


async def read_messages(reader, count, timeout=5.0):
    """Read exactly ``count`` JSON-line messages, or fail the test."""
    messages = []
    buffer = b""
    while len(messages) < count:
        chunk = await asyncio.wait_for(reader.read(65536), timeout=timeout)
        if not chunk:
            break
        buffer += chunk
        while b"\n" in buffer and len(messages) < count:
            line, buffer = buffer.split(b"\n", 1)
            if line.strip():
                messages.append(json.loads(line))
    return messages


async def connect_client(core):
    reader, writer = await asyncio.open_connection("127.0.0.1", core.client_port)
    [welcome] = await read_messages(reader, 1)
    assert welcome["type"] == "welcome"
    return reader, writer, welcome


async def connect_device(core):
    return await asyncio.open_connection("127.0.0.1", core.device_port)


class TestEndToEnd:
    async def test_a_sample_gets_from_a_device_to_a_subscribed_client(self, core):
        reader, writer, _ = await connect_client(core)
        writer.write(encode({"type": "subscribe", "devices": ["eeg"]}))
        await writer.drain()
        [ack] = await read_messages(reader, 1)
        assert ack["type"] == "subscribed"

        _, device = await connect_device(core)
        device.write(encode({"device_id": "eeg", "timestamp": 1000, "Fp1": 1.5}))
        await device.drain()

        messages = await read_messages(reader, 2)  # device_connected, then the sample
        sample = next(m for m in messages if "Fp1" in m)
        assert sample == {"device_id": "eeg", "timestamp": 1000, "Fp1": 1.5}

        writer.close()
        device.close()

    async def test_the_core_processes_once_and_fans_out_to_every_client(self, core):
        # Two clients, same pipeline: both must get the identical filtered stream.
        clients = []
        for _ in range(2):
            reader, writer, _ = await connect_client(core)
            writer.write(
                encode(
                    {
                        "type": "subscribe",
                        "devices": [
                            {
                                "device_id": "eeg",
                                "pipeline": [{"stage": "constant_noise", "offset": 100}],
                            }
                        ],
                    }
                )
            )
            await writer.drain()
            await read_messages(reader, 1)
            clients.append((reader, writer))

        _, device = await connect_device(core)
        for n in range(3):
            device.write(encode({"device_id": "eeg", "timestamp": 1000 + n, "x": float(n)}))
        await device.drain()

        for reader, writer in clients:
            messages = await read_messages(reader, 4)
            values = [m["x"] for m in messages if "x" in m]
            assert values == [100.0, 101.0, 102.0]
            writer.close()

        assert len(core.router._pipelines["eeg"]) == 1  # one pipeline, not two
        device.close()

    async def test_a_client_can_act_as_a_recording_device(self, core):
        # Figure 1: client #1 is also Recording Device #5.
        producer_reader, producer, _ = await connect_client(core)
        consumer_reader, consumer, _ = await connect_client(core)

        consumer.write(encode({"type": "subscribe", "devices": ["mouse"]}))
        await consumer.drain()
        await read_messages(consumer_reader, 1)

        producer.write(encode({"device_id": "mouse", "timestamp": 1000, "x": 42}))
        await producer.drain()

        messages = await read_messages(consumer_reader, 2)
        assert any(m.get("x") == 42 for m in messages)

        producer.close()
        consumer.close()

    async def test_clients_are_told_when_a_device_dies(self, core):
        # The corner case the paper explicitly asks you to prepare for.
        reader, writer, _ = await connect_client(core)
        writer.write(encode({"type": "subscribe", "devices": ["eeg"]}))
        await writer.drain()
        await read_messages(reader, 1)

        _, device = await connect_device(core)
        device.write(encode({"device_id": "eeg", "timestamp": 1, "x": 1.0}))
        await device.drain()
        await read_messages(reader, 2)

        device.close()
        [notice] = await read_messages(reader, 1)
        assert notice == {"type": "device_disconnected", "device_id": "eeg"}
        writer.close()

    async def test_a_gap_survives_the_round_trip_as_null_not_zero(self, core):
        reader, writer, _ = await connect_client(core)
        writer.write(encode({"type": "subscribe", "devices": ["eye"]}))
        await writer.drain()
        await read_messages(reader, 1)

        _, device = await connect_device(core)
        device.write(encode({"device_id": "eye", "timestamp": 1, "pupil": "NA"}))
        await device.drain()

        messages = await read_messages(reader, 2)
        sample = next(m for m in messages if "pupil" in m)
        assert sample["pupil"] is None  # a gap, distinguishable from a real 0

        writer.close()
        device.close()

    async def test_a_bad_pipeline_gets_an_error_not_silence(self, core):
        reader, writer, _ = await connect_client(core)
        writer.write(
            encode(
                {
                    "type": "subscribe",
                    "devices": [{"device_id": "eeg", "pipeline": [{"stage": "nonsense"}]}],
                }
            )
        )
        await writer.drain()

        [error] = await read_messages(reader, 1)
        assert error["type"] == "error"
        assert "nonsense" in error["message"]
        writer.close()

    async def test_list_devices_reports_what_is_connected(self, core):
        _, device = await connect_device(core)
        device.write(encode({"type": "hello", "device_id": "eeg", "sample_rate": 250}))
        device.write(encode({"device_id": "eeg", "timestamp": 1, "Fp1": 1.0}))
        await device.drain()
        await asyncio.sleep(0.05)

        reader, writer, _ = await connect_client(core)
        writer.write(encode({"type": "list_devices"}))
        await writer.drain()

        messages = await read_messages(reader, 1)
        [info] = messages[0]["devices"]
        assert info["device_id"] == "eeg"
        assert info["declared_rate"] == 250
        assert info["channels"] == ["Fp1"]

        writer.close()
        device.close()


class TestShutdown:
    async def test_stop_cancels_live_connections(self):
        # asyncio.start_server spawns each connection handler in a task it owns and does
        # not hand back. Unless the Core registers them itself they are invisible to
        # stop(): the loop tears down, they are collected mid-await ("Task was destroyed
        # but it is pending!"), and their finally blocks -- which mark devices
        # disconnected and drop subscriptions -- are not guaranteed to run.
        core = ThalamusCore(host="127.0.0.1", device_port=0, client_port=0)
        await core.start()

        _, device_writer = await connect_device(core)
        client_reader, client_writer, _ = await connect_client(core)
        device_writer.write(encode({"type": "hello", "device_id": "eeg", "rate": 250}))
        await device_writer.drain()
        await asyncio.sleep(0.1)

        # Both handlers are now live, and the Core knows about both.
        assert len(core._tasks) >= 2, "connection handlers are not being tracked"

        await core.stop()

        assert not core._tasks, "a connection task outlived stop()"
        for writer in (device_writer, client_writer):
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def test_stop_is_safe_with_nothing_connected(self):
        core = ThalamusCore(host="127.0.0.1", device_port=0, client_port=0)
        await core.start()
        await core.stop()
        assert not core._tasks


class TestBackwardsCompatibility:
    async def test_the_pre_1_0_client_still_works(self, core):
        # The original client sent {"subscribe": [...]} with *no trailing newline*
        # and then went quiet. A strict line decoder would wait forever for a newline
        # that is never coming, and the client would silently receive nothing.
        reader, writer, _ = await connect_client(core)
        writer.write(json.dumps({"subscribe": ["eeg"]}).encode("utf-8"))  # no "\n"
        await writer.drain()
        await read_messages(reader, 1)  # the subscription was understood

        _, device = await connect_device(core)
        device.write(encode({"device_id": "eeg", "timestamp": 1, "Fp1": 9.9}))
        await device.drain()

        messages = await read_messages(reader, 2)
        assert any(m.get("Fp1") == 9.9 for m in messages)

        writer.close()
        device.close()

    async def test_a_device_that_never_says_hello_still_registers(self, core):
        # The pre-1.0 devices just started writing samples. They still can.
        _, device = await connect_device(core)
        device.write(encode({"device_id": "legacy", "timestamp": 1, "x": 1.0}))
        await device.drain()
        await asyncio.sleep(0.05)

        assert "legacy" in core.router.devices
        assert core.router.devices["legacy"].connected
        device.close()


class TestBackpressure:
    async def test_a_slow_client_drops_its_own_samples_and_stalls_nobody(self, core):
        # The original code called sendall() to each client from inside the *device's*
        # read loop, so one wedged client blocked the device and, with it, every other
        # client in the study. Here a client that stops reading can only hurt itself.
        core.queue_size = 20

        # A client that subscribes and then never reads a single byte.
        stuck = socket.create_connection(("127.0.0.1", core.client_port))
        stuck.sendall(encode({"type": "subscribe", "devices": ["eeg"]}))
        await asyncio.sleep(0.05)

        healthy_reader, healthy, _ = await connect_client(core)
        healthy.write(encode({"type": "subscribe", "devices": ["eeg"]}))
        await healthy.drain()
        await read_messages(healthy_reader, 1)

        # Each sample carries a few KB of payload, the way a webcam frame does. Tiny
        # samples would simply vanish into the OS socket buffer and the wedged client
        # would never actually fall behind — there would be nothing to test.
        blob = "x" * 4000

        _, device = await connect_device(core)
        for n in range(400):
            device.write(encode({"device_id": "eeg", "timestamp": n, "x": float(n), "frame": blob}))
            if n % 10 == 0:
                await device.drain()
                await asyncio.sleep(0.002)  # let the event loop actually serve people

        # The healthy client keeps receiving, in order, despite the wedged one.
        messages = await read_messages(healthy_reader, 100)
        values = [m["x"] for m in messages if "x" in m]
        assert len(values) > 50
        assert values == sorted(values)

        # ...and the wedged client is the one paying for being wedged.
        wedged = [c for c in core.router._sinks.values() if c.dropped > 0]
        assert wedged, "the client that stopped reading should have dropped samples"

        stuck.close()
        healthy.close()
        device.close()


class TestDeviceSDK:
    async def test_a_synthetic_device_streams_at_the_rate_it_was_given(self, core):
        reader, writer, _ = await connect_client(core)
        writer.write(encode({"type": "subscribe", "devices": ["synth"]}))
        await writer.drain()
        await read_messages(reader, 1)

        device = SyntheticDevice(
            "synth",
            {"x": {"kind": "sine", "freq": 1.0}},
            rate=100,
            host="127.0.0.1",
            port=core.device_port,
        )
        thread = device.run_in_thread()

        messages = await read_messages(reader, 30)
        samples = [m for m in messages if "x" in m]
        assert len(samples) >= 20

        # Timestamps must advance by ~10 ms at 100 Hz, and never go backwards.
        stamps = [m["timestamp"] for m in samples]
        assert stamps == sorted(stamps)
        gaps = [b - a for a, b in zip(stamps, stamps[1:])]
        assert 5 <= sum(gaps) / len(gaps) <= 15

        device.stop()
        thread.join(timeout=2)
        writer.close()

    async def test_the_client_sdk_talks_to_the_core(self, core):
        # The blocking client is what a researcher actually writes, so it gets an
        # end-to-end test of its own, driven from a thread.
        device = SyntheticDevice(
            "eye",
            {"pupil": {"kind": "sine", "freq": 0.5, "offset": 3.5}},
            rate=100,
            host="127.0.0.1",
            port=core.device_port,
        )
        device.run_in_thread()

        def use_client(port):
            with ThalamusClient(host="127.0.0.1", port=port, timeout=5) as client:
                client.subscribe("eye", pipeline=[{"stage": "missing_fill", "strategy": "zero"}])
                collected = []
                for sample in client.stream():
                    collected.append(sample)
                    if len(collected) >= 10:
                        return collected
            return collected

        samples = await asyncio.get_running_loop().run_in_executor(
            None, use_client, core.client_port
        )
        device.stop()

        assert len(samples) == 10
        assert all(s.device_id == "eye" for s in samples)
        assert all(s.data["pupil"] is not MISSING for s in samples)
