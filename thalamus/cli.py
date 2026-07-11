"""The ``thalamus`` command.

Everything the toolkit does is reachable without writing any Python:

    thalamus demo                      # a whole simulated study, no data files needed
    thalamus run study.yaml            # your study: Core plus every device in it
    thalamus serve                     # just the Core, for devices elsewhere
    thalamus devices                   # what is connected, and at what real rate
    thalamus monitor eeg --filter savgol
    thalamus record eeg eye --out recordings/ --duration 60
    thalamus stages                    # what processing is available
    thalamus make-data --out examples/data
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import csv
import json
import logging
import signal
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import __version__
from .client import ThalamusClient, ThalamusError
from .config import ConfigError, StudyConfig
from .core import ThalamusCore
from .processing import available_stages
from .protocol import DEFAULT_CLIENT_PORT, DEFAULT_DEVICE_PORT, MISSING, Sample
from .runner import StudyRunner

logger = logging.getLogger("thalamus")


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #


def cmd_serve(args: argparse.Namespace) -> int:
    """Just the Core. Devices and clients connect from wherever they live."""
    core = ThalamusCore(host=args.host, device_port=args.device_port, client_port=args.client_port)
    print("Thalamus Core")
    print(f"  devices -> {args.host}:{args.device_port}")
    print(f"  clients -> {args.host}:{args.client_port}")
    print("Ctrl-C to stop.\n")
    _run(core.serve_forever())
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """A whole study: the Core, plus every device the config describes."""
    try:
        config = StudyConfig.load(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    runner = StudyRunner(config)
    print(f"Study: {config.path}")
    for device in config.devices:
        simulated = (
            "  [" + ", ".join(s["stage"] for s in device.simulate) + "]" if device.simulate else ""
        )
        print(f"  {device.id:<20} {device.type}{simulated}")
    print(f"\n  clients -> {config.host}:{config.client_port}")
    print("Ctrl-C to stop.\n")

    _run(runner.run_forever(duration_s=args.duration))
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """A complete three-device study out of thin air — no data files, no hardware.

    This exists so that ``git clone && pip install -e . && thalamus demo`` produces
    a running multimodal study in under a minute. Everything else in the toolkit can
    be learned from a thing that is already working.
    """
    config = StudyConfig.from_dict(
        {
            "core": {"device_port": args.device_port, "client_port": args.client_port},
            "devices": [
                # The two devices from the paper, with the real channel names the
                # hardware writes. The GP3 inherits its real failure mode from the
                # profile: a blink freezes the gaze and pupil and drops BPOGV to 0.
                #
                # The Unicorn's real recording dropped no packets at all, so the profile
                # simulates none — but a flaky link is worth showing, so we ask for one
                # here explicitly. Watch the Counter column jump when it happens.
                {
                    "id": "eeg",
                    "type": "synthetic",
                    "profile": "unicorn_hybrid_black",
                    "seed": 1,
                    "simulate": [{"stage": "dropout", "probability": 0.0008, "burst": 3}],
                },
                {"id": "eye_tracker", "type": "synthetic", "profile": "gp3", "seed": 2},
                # And an ECG, which no profile covers, to show the hand-rolled form.
                {
                    "id": "ecg",
                    "type": "synthetic",
                    "rate": 128,
                    "seed": 3,
                    "signals": {"lead_ii": {"kind": "ecg", "heart_rate": 72}},
                    "simulate": [{"stage": "dropout", "probability": 0.001, "burst": 5, "seed": 9}],
                },
            ],
        }
    )

    runner = StudyRunner(config)
    print("Thalamus demo — three simulated devices, no data files, no hardware.\n")
    print("  eeg          250 Hz   Unicorn Hybrid Black: EEG1-8 + IMU, a lossy link (see Counter)")
    print("  eye_tracker  150 Hz   Gazepoint GP3: BPOGX/Y + pupils, blinks that FREEZE (BPOGV=0)")
    print("  ecg          128 Hz   lead II at 72 bpm, occasional dropped packets\n")
    print(f"  clients -> localhost:{config.client_port}")
    print("\nIn another terminal, try:")
    print(f"  thalamus devices --port {config.client_port}")
    print(f"  thalamus monitor eye_tracker --port {config.client_port}")
    print(f"  thalamus monitor eeg --channels EEG1 --filter savgol --port {config.client_port}")
    print(
        f"  thalamus record eeg eye_tracker --out ./recordings --duration 30"
        f" --port {config.client_port}"
    )
    print("\nCtrl-C to stop.\n")

    _run(runner.run_forever(duration_s=args.duration))
    return 0


def cmd_devices(args: argparse.Namespace) -> int:
    """What is connected, and — importantly — at what rate it is *actually* running."""
    with _client(args) as client:
        devices = client.devices()

    if not devices:
        print("No devices connected.")
        return 0

    print(f"{'DEVICE':<20} {'RATE':>12} {'SAMPLES':>10}  CHANNELS")
    for device in devices:
        declared = device.get("declared_rate")
        effective = device.get("effective_rate")
        rate = _format_rate(declared, effective)
        status = "" if device.get("connected") else "  (disconnected)"
        channels = ", ".join(device.get("channels") or []) or "-"
        if len(channels) > 40:
            channels = channels[:37] + "..."
        print(
            f"{device['device_id']:<20} {rate:>12} {device.get('samples', 0):>10}  "
            f"{channels}{status}"
        )
    return 0


def cmd_monitor(args: argparse.Namespace) -> int:
    """Print a live stream. The quickest way to see whether a thing is working."""
    pipeline = _pipeline_from_args(args)

    with _client(args) as client:
        try:
            if args.sync:
                client.subscribe_synced(args.devices, tolerance_ms=args.tolerance_ms)
            else:
                for device_id in args.devices:
                    client.subscribe(
                        device_id,
                        channels=args.channels,
                        pipeline=pipeline,
                        latency_ms=args.latency_ms,
                    )
        except ThalamusError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        print(f"Subscribed to {', '.join(args.devices)}. Ctrl-C to stop.\n")
        count = 0
        try:
            for message in client.messages():
                line = _format_message(message)
                if line is None:
                    continue
                print(line, flush=True)
                count += 1
                if args.limit and count >= args.limit:
                    break
        except KeyboardInterrupt:
            print(f"\n{count} message(s).")
    return 0


def cmd_record(args: argparse.Namespace) -> int:
    """Write the streams to CSV, one file per device, plus events.

    A dry run you cannot inspect afterwards is only half a dry run: this is what
    lets you check that the data you *would* have collected is the data your
    analysis expects — before a participant is ever in the room.
    """
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    writers: Dict[str, Any] = {}
    handles: Dict[str, Any] = {}
    counts: Dict[str, int] = {}
    events: List[Dict[str, Any]] = []
    deadline = None

    with _client(args) as client:
        for device_id in args.devices:
            client.subscribe(device_id, channels=args.channels, pipeline=_pipeline_from_args(args))

        if args.duration:
            import time

            deadline = time.monotonic() + args.duration

        print(f"Recording {', '.join(args.devices)} to {out}/ — Ctrl-C to stop.")
        try:
            for message in client.messages():
                if isinstance(message, Sample):
                    _write_row(message, out, writers, handles, counts)
                elif message.get("type") == "event":
                    events.append(message)
                elif message.get("type") == "device_disconnected":
                    print(f"  ! {message['device_id']} disconnected")

                if deadline is not None:
                    import time

                    if time.monotonic() >= deadline:
                        break
        except KeyboardInterrupt:
            pass
        finally:
            for handle in handles.values():
                handle.close()
            if events:
                (out / "events.jsonl").write_text(
                    "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
                )

    print("\nWrote:")
    for device_id, count in sorted(counts.items()):
        print(f"  {out / f'{device_id}.csv'}  ({count} samples)")
    if events:
        print(f"  {out / 'events.jsonl'}  ({len(events)} events)")
    return 0


def cmd_stages(_args: argparse.Namespace) -> int:
    """List the processing stages this installation can run."""
    from .processing import base as _base

    print("Available processing stages:\n")
    for name in available_stages():
        cls = _base._REGISTRY[name]
        summary = (cls.__doc__ or "").strip().split("\n")[0]
        print(f"  {name:<18} {summary}")
    print("\nUse them in a study config under 'simulate:', or in a client subscription.")
    print("Details: python -c \"help('thalamus.processing')\"")
    return 0


def cmd_profiles(args: argparse.Namespace) -> int:
    """The hardware Thalamus knows: real channel names, real rates, real quirks.

    Name one on a device and the simulation emits the columns the actual device
    writes — so the analysis code you write today runs unchanged on the recording
    you make next month.
    """
    from .devices.profiles import all_profiles, get_profile

    if not args.profile:
        print(f"{'PROFILE':<22} {'DEVICE':<32} {'RATE':>8}  CHANNELS")
        for profile in all_profiles():
            device = f"{profile.vendor} {profile.model}"
            names = ", ".join(profile.channel_names)
            if len(names) > 34:
                names = names[:31] + "..."
            print(f"{profile.key:<22} {device:<32} {profile.rate:>5g} Hz  {names}")
        print("\nDetails:  thalamus profiles gp3")
        return 0

    try:
        profile = get_profile(args.profile)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"{profile.vendor} {profile.model}  ({profile.modality}, {profile.rate:g} Hz)\n")
    width = max(len(c.name) for c in profile.channels)
    for channel in profile.channels:
        unit = f"[{channel.unit}]" if channel.unit else ""
        print(f"  {channel.name:<{width}}  {unit:<12} {channel.about}")

    if profile.validity_flag:
        covered = ", ".join(profile.covered_by_flag())
        print(
            f"\n  {profile.validity_flag} = {profile.validity_ok} means valid; it vouches for "
            f"{covered}."
        )
        print("  Replaying a real recording? Put `- stage: validity_mask` first.")

    if profile.notes:
        print(f"\n{_wrap(profile.notes, indent='  ')}")

    print("\nUse it:\n")
    print(f"  devices:\n    - id: {profile.modality}\n      type: synthetic")
    print(f"      profile: {profile.key}")
    if profile.simulate:
        stages = ", ".join(s["stage"] for s in profile.simulate)
        print(f"\n  With no `simulate:` of its own it will also simulate: {stages}")
        print("  (that is what this hardware does in the field; `simulate: []` turns it off)")
    return 0


def cmd_make_data(args: argparse.Namespace) -> int:
    """Generate small sample recordings, so replay works without downloading anything.

    These are written with the *real* column names of the devices in the paper, so a
    file from here and a file off the actual hardware are interchangeable.
    """
    from .devices.profiles import get_profile
    from .devices.synthetic import SyntheticDevice

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # (filename, profile, seed). Same devices as the paper: a Unicorn EEG cap and a
    # Gazepoint eye tracker.
    recipes = [
        ("eeg.csv", "unicorn_hybrid_black", 3),
        ("eye-tracking.csv", "gp3", 4),
    ]

    origin = 1690535469479  # an arbitrary but fixed epoch, so files are reproducible
    for filename, profile_key, seed in recipes:
        profile = get_profile(profile_key)
        device = SyntheticDevice(
            "generator", profile=profile_key, duration_s=args.duration, seed=seed
        )
        columns = [c for c in profile.channels if c.signal is not None]

        path = out / filename
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp", *(c.name for c in columns)])
            for index, reading in enumerate(device.samples()):
                timestamp = origin + round(index * 1000 / profile.rate)
                writer.writerow([timestamp, *(reading[c.name] for c in columns)])
        print(
            f"  wrote {path}  ({profile.model}, {profile.rate:g} Hz, "
            f"{args.duration:g}s, {len(columns)} channels)"
        )

    print("\nReplay them with:\n  thalamus run examples/study.yaml")
    return 0


def _wrap(text: str, *, width: int = 76, indent: str = "") -> str:
    import textwrap

    return "\n".join(
        textwrap.wrap(text, width=width, initial_indent=indent, subsequent_indent=indent)
    )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _client(args: argparse.Namespace) -> ThalamusClient:
    try:
        return ThalamusClient(host=args.host, port=args.port).connect()
    except (ConnectionError, OSError) as exc:
        raise SystemExit(
            f"error: cannot reach Thalamus at {args.host}:{args.port} ({exc}).\n"
            f"Is it running? Start one with:  thalamus demo"
        ) from exc


def _pipeline_from_args(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Turn ``--filter savgol --noise 0.5 --delay 100`` into a pipeline spec."""
    pipeline: List[Dict[str, Any]] = []

    # First, before anything numeric touches the data: honour the device's own
    # verdict on which samples are real.
    if getattr(args, "validity", None):
        pipeline.append({"stage": "validity_mask", "profile": args.validity})

    if getattr(args, "filter", None):
        if args.filter == "savgol":
            pipeline.append({"stage": "savgol", "window": 11, "polyorder": 3})
        else:
            pipeline.append({"stage": args.filter})

    if getattr(args, "noise", None):
        pipeline.append({"stage": "gaussian_noise", "sigma": args.noise})

    if getattr(args, "delay_ms", None):
        pipeline.append({"stage": "delay", "mode": "timestamp", "delay_ms": args.delay_ms})

    if getattr(args, "fill_missing", False):
        pipeline.append({"stage": "missing_fill", "strategy": "zero"})

    return pipeline


def _format_rate(declared: Optional[float], effective: Optional[float]) -> str:
    if effective is None:
        return f"{declared:g} Hz" if declared else "-"
    if declared and abs(effective - declared) / declared > 0.05:
        # A device that claims 250 Hz and delivers 190 is the finding, not a detail.
        return f"{effective:g}/{declared:g} Hz!"
    return f"{effective:g} Hz"


def _format_message(message: Any) -> Optional[str]:
    if isinstance(message, Sample):
        values = ", ".join(
            f"{k}={'--' if v is MISSING else (f'{v:.3f}' if isinstance(v, float) else v)}"
            for k, v in list(message.data.items())[:6]
            if not (isinstance(v, str) and len(v) > 24)  # skip base64 video frames
        )
        return f"{message.timestamp}  {message.device_id:<14} {values}"

    kind = message.get("type")
    if kind == "frame":
        parts = []
        for device, sample in message["streams"].items():
            parts.append(f"{device}={'--' if sample is None else 'ok'}")
        mark = " " if message.get("complete") else "!"
        return f"{message['timestamp']} {mark} frame  " + "  ".join(parts)
    if kind in ("device_connected", "device_disconnected"):
        name = message.get("device_id") or message.get("device", {}).get("device_id")
        return f"--- {kind}: {name}"
    if kind == "event":
        return f"--- event: {message.get('label')}"
    if kind == "error":
        return f"!!! error: {message.get('message')}"
    return None


def _write_row(sample: Sample, out: Path, writers, handles, counts) -> None:
    device_id = sample.device_id
    if device_id not in writers:
        handle = (out / f"{device_id}.csv").open("w", newline="", encoding="utf-8")
        writer = csv.writer(handle)
        # The channel set is fixed by the first sample. A device whose channels
        # change mid-stream is a bug worth hearing about, not one to paper over.
        writer.writerow(["timestamp", *sample.data])
        handles[device_id] = handle
        writers[device_id] = (writer, list(sample.data))
        counts[device_id] = 0

    writer, channels = writers[device_id]
    writer.writerow(
        [sample.timestamp]
        + ["NA" if sample.data.get(c, MISSING) is MISSING else sample.data.get(c) for c in channels]
    )
    counts[device_id] += 1


def _run(coro) -> None:
    """Run an async entry point, exiting quietly on Ctrl-C."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    task = loop.create_task(coro)

    def _cancel() -> None:
        task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        # Windows has no add_signal_handler; KeyboardInterrupt still gets us out.
        with contextlib.suppress(NotImplementedError, RuntimeError):
            loop.add_signal_handler(sig, _cancel)

    try:
        loop.run_until_complete(task)
    except (asyncio.CancelledError, KeyboardInterrupt):
        print("\nStopped.")
    finally:
        loop.close()


# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thalamus",
        description="Simulate, capture, and manipulate multimodal sensing data streams.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Start here:  thalamus demo",
    )
    parser.add_argument("--version", action="version", version=f"thalamus {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="log what the Core is doing")
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    def add(name: str, help_text: str, func) -> argparse.ArgumentParser:
        sub = subparsers.add_parser(name, help=help_text, description=func.__doc__)
        sub.set_defaults(func=func)
        return sub

    serve = add("serve", "run the Core on its own", cmd_serve)
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--device-port", type=int, default=DEFAULT_DEVICE_PORT)
    serve.add_argument("--client-port", type=int, default=DEFAULT_CLIENT_PORT)

    run = add("run", "run a study: the Core plus its devices", cmd_run)
    run.add_argument("config", help="a study file (YAML or JSON)")
    run.add_argument("--duration", type=float, help="stop after this many seconds")

    demo = add("demo", "run a three-device study with no data files", cmd_demo)
    demo.add_argument("--device-port", type=int, default=DEFAULT_DEVICE_PORT)
    demo.add_argument("--client-port", type=int, default=DEFAULT_CLIENT_PORT)
    demo.add_argument("--duration", type=float, help="stop after this many seconds")

    devices = add("devices", "list what is connected", cmd_devices)
    _add_client_args(devices)

    monitor = add("monitor", "print a live stream", cmd_monitor)
    monitor.add_argument("devices", nargs="+", help="device id(s) to watch")
    _add_client_args(monitor)
    _add_pipeline_args(monitor)
    monitor.add_argument("--sync", action="store_true", help="align the devices into frames")
    monitor.add_argument("--tolerance-ms", type=float, default=20.0, help="with --sync")
    monitor.add_argument("--latency-ms", type=float, default=0.0, help="delay delivery to me")
    monitor.add_argument("--limit", type=int, help="stop after N messages")

    record = add("record", "write streams to CSV", cmd_record)
    record.add_argument("devices", nargs="+", help="device id(s) to record")
    record.add_argument("--out", default="recordings", help="output directory")
    record.add_argument("--duration", type=float, help="stop after this many seconds")
    _add_client_args(record)
    _add_pipeline_args(record)

    add("stages", "list the available processing stages", cmd_stages)

    profiles = add("profiles", "list the hardware Thalamus knows", cmd_profiles)
    profiles.add_argument("profile", nargs="?", help="show one in full (e.g. gp3)")

    make_data = add("make-data", "generate sample recordings to replay", cmd_make_data)
    make_data.add_argument("--out", default="examples/data", help="output directory")
    make_data.add_argument("--duration", type=float, default=60.0, help="seconds per file")

    return parser


def _add_client_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--host", default="localhost", help="where the Core is")
    sub.add_argument("--port", type=int, default=DEFAULT_CLIENT_PORT, help="its client port")


def _add_pipeline_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--channels", nargs="+", help="only these channels")
    sub.add_argument(
        "--validity",
        metavar="PROFILE",
        help="use this hardware's validity flag (e.g. gp3) to blank invalid samples",
    )
    sub.add_argument(
        "--filter",
        choices=["savgol", "kalman", "moving_average", "exponential"],
        help="filter the stream in the Core before it reaches me",
    )
    sub.add_argument("--noise", type=float, metavar="SIGMA", help="add Gaussian noise")
    sub.add_argument("--delay-ms", type=float, help="shift timestamps by this much")
    sub.add_argument(
        "--fill-missing", action="store_true", help="replace gaps with 0 (paper Fig. 2)"
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        parser.print_help()
        return 0

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
