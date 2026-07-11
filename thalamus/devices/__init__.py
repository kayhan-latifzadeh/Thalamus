"""Recording devices, real and simulated.

:class:`~thalamus.devices.base.RecordingDevice` is the one thing to subclass to
integrate a *real* device: implement :meth:`samples` and the socket handling is
done for you. The rest are simulated devices ready to use:

* :class:`~thalamus.devices.replay.ReplayDevice` — stream a CSV/JSONL recording.
* :class:`~thalamus.devices.synthetic.SyntheticDevice` — generate signals from nothing.
* :class:`~thalamus.devices.video.VideoDevice` — stream a video file as frames.

:class:`VideoDevice` is imported lazily, so the package works without OpenCV.
"""

from typing import Any

from .base import CallableDevice, Pacer, RecordingDevice, read_channels  # noqa: F401
from .profiles import (  # noqa: F401
    Channel,
    DeviceProfile,
    available_profiles,
    get_profile,
    register_profile,
)
from .replay import ReplayDevice  # noqa: F401
from .synthetic import SyntheticDevice  # noqa: F401

__all__ = [
    "CallableDevice",
    "Channel",
    "DeviceProfile",
    "Pacer",
    "RecordingDevice",
    "ReplayDevice",
    "SyntheticDevice",
    "VideoDevice",
    "available_profiles",
    "get_profile",
    "read_channels",
    "register_profile",
]


def __getattr__(name: str) -> Any:
    # Keep OpenCV out of the import path for the 95% of users who never stream video.
    if name == "VideoDevice":
        from .video import VideoDevice

        return VideoDevice
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
