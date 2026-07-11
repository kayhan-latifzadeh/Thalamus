"""Thalamus — a user simulation toolkit for prototyping multimodal sensing studies.

Named after the part of the brain that relays sensory information: Thalamus Core is
the hub that every device streams into and every client subscribes to.

    Latifzadeh, K. & Leiva, L.A. (2025). Thalamus: A User Simulation Toolkit for
    Prototyping Multimodal Sensing Studies. UMAP Adjunct '25.
    https://doi.org/10.1145/3708319.3733687

The shortest useful program::

    from thalamus import SyntheticDevice, ThalamusClient

    SyntheticDevice("eye", {"pupil": {"kind": "sine", "freq": 0.2}}, rate=150).run_in_thread()

    with ThalamusClient() as client:
        client.subscribe("eye", pipeline=[{"stage": "gaussian_noise", "sigma": 0.1}])
        for sample in client.stream():
            print(sample.timestamp, sample.data)

...with ``thalamus serve`` running somewhere. Or skip all of it and run
``thalamus demo``.
"""

from .client import ThalamusClient, ThalamusError
from .config import ConfigError, StudyConfig
from .core import Router, ThalamusCore
from .devices import CallableDevice, RecordingDevice, ReplayDevice, SyntheticDevice
from .processing import Pipeline, PipelineSpecError, Stage, available_stages, register
from .protocol import MISSING, Sample
from .runner import StudyRunner
from .sync import Frame, Synchronizer

__version__ = "1.0.0"

__all__ = [
    "__version__",
    # protocol
    "Sample",
    "MISSING",
    # core
    "ThalamusCore",
    "Router",
    "StudyRunner",
    "StudyConfig",
    "ConfigError",
    # devices
    "RecordingDevice",
    "ReplayDevice",
    "SyntheticDevice",
    "CallableDevice",
    # clients
    "ThalamusClient",
    "ThalamusError",
    # processing
    "Stage",
    "Pipeline",
    "PipelineSpecError",
    "available_stages",
    "register",
    # sync
    "Synchronizer",
    "Frame",
]
