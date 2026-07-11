"""Describe a whole simulated study in one file.

The point of this module is that a researcher should not have to write Python to
run a dry-run of their study. A YAML file names the devices, where their data comes
from, and what should go wrong with them; ``thalamus run study.yaml`` starts the
Core and every device in one process::

    core:
      device_port: 9000
      client_port: 9001

    devices:
      - id: eeg
        type: replay
        path: examples/data/eeg.csv
        rate: 250
        simulate:                     # this device's own imperfections
          - stage: constant_noise     # a drifting electrode
            offset: 0.5
            drift: 0.001

      - id: eye_tracker
        type: synthetic
        rate: 150
        signals:
          pupil:  {kind: sine, freq: 0.2, amplitude: 0.4, offset: 3.5}
          gaze_x: {kind: random_walk, step: 6, start: 960, minimum: 0, maximum: 1920}
        simulate:
          - stage: missing_inject     # blinks: 100-400 ms of nothing, at 150 Hz
            probability: 0.004
            burst: [15, 60]
            seed: 7

The file is validated eagerly and completely: every device, every stage, every
parameter is checked *before* anything starts listening, and the errors name the
device and the line. A study config that is going to fail should fail in the first
second, not forty minutes into a dry run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .devices import (
    DeviceProfile,
    RecordingDevice,
    ReplayDevice,
    SyntheticDevice,
    get_profile,
    register_profile,
)
from .processing import PipelineSpecError, build_pipeline
from .protocol import DEFAULT_CLIENT_PORT, DEFAULT_DEVICE_PORT

logger = logging.getLogger(__name__)

DEVICE_TYPES = ("replay", "synthetic", "video")


class ConfigError(ValueError):
    """The study configuration is not usable. The message says where and why."""


@dataclass
class DeviceConfig:
    id: str
    type: str
    simulate: List[Dict[str, Any]] = field(default_factory=list)
    options: Dict[str, Any] = field(default_factory=dict)

    def build(self, *, host: str, port: int) -> RecordingDevice:
        """Construct the device. Raises :class:`ConfigError` with the device named."""
        options = dict(self.options, host=host, port=port)
        try:
            if self.type == "replay":
                return ReplayDevice(self.id, **options)
            if self.type == "synthetic":
                return SyntheticDevice(self.id, **options)
            if self.type == "video":
                from .devices.video import VideoDevice

                # VideoDevice has one channel and no signals to fake, so a profile can
                # only tell it a frame rate. Take that and drop the rest.
                profile = options.pop("profile", None)
                if profile:
                    options.setdefault("rate", get_profile(profile).rate)
                return VideoDevice(self.id, **options)
        except (TypeError, ValueError, FileNotFoundError, ImportError) as exc:
            raise ConfigError(f"device {self.id!r}: {exc}") from exc

        raise ConfigError(
            f"device {self.id!r}: unknown type {self.type!r}; "
            f"expected one of {', '.join(DEVICE_TYPES)}"
        )


@dataclass
class StudyConfig:
    host: str = "0.0.0.0"
    device_port: int = DEFAULT_DEVICE_PORT
    client_port: int = DEFAULT_CLIENT_PORT
    devices: List[DeviceConfig] = field(default_factory=list)
    path: Optional[Path] = None

    @classmethod
    def load(cls, path: Union[str, Path]) -> StudyConfig:
        """Read and validate a study file (YAML or JSON)."""
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"no such config file: {path}")

        text = path.read_text(encoding="utf-8")
        try:
            raw = _parse(text, path)
        except ValueError as exc:
            raise ConfigError(f"{path}: {exc}") from exc

        config = cls.from_dict(raw)
        config.path = path

        # Relative paths in a study file are relative to *the study file*, not to
        # whatever directory the command happened to be run from. Anything else
        # makes a config that only works from one place.
        for device in config.devices:
            source = device.options.get("path")
            if source and not Path(source).is_absolute():
                device.options["path"] = str((path.parent / source).resolve())

        return config

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> StudyConfig:
        if not isinstance(raw, dict):
            raise ConfigError(f"the config must be a mapping, got {type(raw).__name__}")

        core = raw.get("core") or {}
        if not isinstance(core, dict):
            raise ConfigError("'core' must be a mapping")

        unknown = set(raw) - {"core", "devices", "profiles"}
        if unknown:
            raise ConfigError(
                f"unknown top-level key(s): {', '.join(sorted(unknown))}. "
                f"Expected 'core', 'devices', and/or 'profiles'."
            )

        # Your device, described in your study file. The three profiles Thalamus ships
        # with are examples, not a supported-hardware list: anything defined here is
        # exactly as first-class, and `profile:` on a device below can name it.
        custom = raw.get("profiles") or {}
        if not isinstance(custom, dict):
            raise ConfigError("'profiles' must be a mapping of name -> profile")
        for name, spec in custom.items():
            try:
                register_profile(DeviceProfile.from_dict(str(name), spec))
            except (ValueError, TypeError) as exc:
                raise ConfigError(f"profiles: {exc}") from exc

        entries = raw.get("devices") or []
        if not isinstance(entries, list):
            raise ConfigError("'devices' must be a list")

        devices: List[DeviceConfig] = []
        seen: set = set()

        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ConfigError(f"devices[{index}] must be a mapping")

            options = dict(entry)
            device_id = options.pop("id", None)
            device_type = options.pop("type", None)
            simulate = options.pop("simulate", None) or []

            if not device_id:
                raise ConfigError(f"devices[{index}] has no 'id'")
            if device_id in seen:
                raise ConfigError(f"duplicate device id {device_id!r}")
            seen.add(device_id)
            if not device_type:
                raise ConfigError(f"device {device_id!r} has no 'type'")
            if device_type not in DEVICE_TYPES:
                raise ConfigError(
                    f"device {device_id!r}: unknown type {device_type!r}; "
                    f"expected one of {', '.join(DEVICE_TYPES)}"
                )

            # A device that names real hardware and says nothing about what goes wrong
            # with it gets what actually goes wrong with it: the GP3 blinks, the Unicorn
            # drops Bluetooth packets. Writing `simulate: []` turns that off, and
            # `thalamus run` prints the stages it ended up with on every start, so the
            # default is a shortcut rather than a secret.
            profile_name = options.get("profile")
            if profile_name and "simulate" not in entry:
                try:
                    simulate = [dict(s) for s in get_profile(profile_name).simulate]
                except ValueError as exc:
                    raise ConfigError(f"device {device_id!r}: {exc}") from exc

            # Build the pipeline now and throw it away. It costs nothing and it
            # turns "your seed is a string, forty minutes in" into "your seed is a
            # string, before anything started".
            try:
                build_pipeline(simulate)
            except PipelineSpecError as exc:
                raise ConfigError(f"device {device_id!r}: simulate: {exc}") from exc

            devices.append(
                DeviceConfig(
                    id=device_id, type=device_type, simulate=list(simulate), options=options
                )
            )

        return cls(
            host=core.get("host", "0.0.0.0"),
            device_port=int(core.get("device_port", DEFAULT_DEVICE_PORT)),
            client_port=int(core.get("client_port", DEFAULT_CLIENT_PORT)),
            devices=devices,
        )


def _parse(text: str, path: Path) -> Dict[str, Any]:
    if path.suffix.lower() == ".json":
        import json

        return json.loads(text)

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - PyYAML is a declared dependency
        raise ValueError(
            "reading a YAML config needs PyYAML (pip install pyyaml), "
            "or write the config as .json instead"
        ) from exc

    return yaml.safe_load(text) or {}
