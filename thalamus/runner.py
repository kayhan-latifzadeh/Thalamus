"""Run a whole simulated study in one process.

The Core is async; devices are blocking socket clients. Rather than force one
model on the other, each device runs on its own thread and talks to the Core over
a real TCP connection — the same connection a device on another machine would use.
That is deliberate: the in-process path and the across-the-network path are the
*same* path, so a study that works on your laptop works when you move the eye
tracker to the machine it is actually plugged into. Nothing is special-cased for
being local, and so nothing can work locally and break remotely.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

from .config import StudyConfig
from .core import ThalamusCore
from .devices import RecordingDevice

logger = logging.getLogger(__name__)


class StudyRunner:
    """Starts the Core and every device described by a :class:`StudyConfig`."""

    def __init__(self, config: StudyConfig) -> None:
        self.config = config
        self.core = ThalamusCore(
            host=config.host,
            device_port=config.device_port,
            client_port=config.client_port,
        )
        self.devices: List[RecordingDevice] = []

    async def start(self) -> None:
        await self.core.start()

        # Register each device's imperfections with the Core *before* the device can
        # connect, so that not a single sample slips through unsimulated.
        for spec in self.config.devices:
            if spec.simulate:
                self.core.router.set_ingest_pipeline(spec.id, spec.simulate)

        for spec in self.config.devices:
            device = spec.build(host="127.0.0.1", port=self.core.device_port)
            self.devices.append(device)
            device.run_in_thread()
            logger.info(
                "started device %s (%s)%s",
                spec.id,
                spec.type,
                f", simulating: {', '.join(s['stage'] for s in spec.simulate)}"
                if spec.simulate
                else "",
            )

    async def stop(self) -> None:
        for device in self.devices:
            device.stop()
        self.devices.clear()
        await self.core.stop()

    async def run_forever(self, duration_s: Optional[float] = None) -> None:
        await self.start()
        try:
            if duration_s is None:
                await asyncio.Event().wait()
            else:
                await asyncio.sleep(duration_s)
        finally:
            await self.stop()
