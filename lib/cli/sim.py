"""MuJoCo sim entrypoint."""

from __future__ import annotations

import argparse
import shutil
import sys

from lib.cli.deploy_runner import run_docker, run_host
from lib.vendor.env import in_docker


def _deploy_argv(args: argparse.Namespace) -> list[str]:
    argv = ["--network", "sim_sync_mujoco"]
    if args.auto_next_node:
        argv.append("--auto-next-node")
    if args.vr_replay:
        argv.extend(["--vr-replay", args.vr_replay])
    elif args.vr:
        argv.append("--vr")
        if args.vr_port != 8012:
            argv.extend(["--vr-port", str(args.vr_port)])
    elif args.wasd:
        argv.append("--wasd")
    if getattr(args, "cpu", False):
        argv.extend(["--device", "cpu"])
    if args.checkpoint:
        argv.extend(["-c", args.checkpoint])
    if args.debug:
        argv.append("--debug")
    if args.view:
        argv.append("--view")
    if args.view_port != 8766:
        argv.extend(["--view-port", str(args.view_port)])
    if args.no_wait:
        argv.append("--no-wait")
    if getattr(args, "fast", False):
        argv.append("--fast")
    if getattr(args, "window", False):
        argv.append("--window")
    if args.extra:
        extra = list(args.extra)
        if extra and extra[0] == "--":
            extra = extra[1:]
        argv.extend(extra)
    return argv


def cmd_sim(args: argparse.Namespace) -> int:
    deploy_argv = _deploy_argv(args)
    use_docker = args.docker or (not args.host and not in_docker("g1_sim"))
    if args.host:
        use_docker = False

    if use_docker and shutil.which("docker") is None:
        print("docker not found; run with --host or install Docker", file=sys.stderr)
        return 1

    try:
        if use_docker:
            return run_docker(deploy_argv, "g1-sim-cpu" if args.cpu else "g1-sim")
        return run_host(deploy_argv, mujoco_gl="egl")
    except KeyboardInterrupt:
        return 130


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    sim = subparsers.add_parser("sim", help="Run MuJoCo sim deploy (deploy)")
    sim.add_argument(
        "--wasd",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keyboard WASD pose control (default: on)",
    )
    sim.add_argument(
        "--vr-replay",
        metavar="JSON",
        help="VR replay trajectory under assets/trajectories/",
    )
    sim.add_argument(
        "--vr",
        action="store_true",
        help="Live VR via Vuer WebXR — prints an https URL to open in the headset browser",
    )
    sim.add_argument(
        "--vr-port",
        type=int,
        default=8012,
        help="HTTPS/WSS port for the Vuer WebXR server (default: 8012)",
    )
    sim.add_argument(
        "--cpu",
        action="store_true",
        help="CPU-only / software-rendering sim (g1-sim-cpu image, no NVIDIA GPU; slow). "
        "Use on macOS/Windows with WSL2 or hosts without an NVIDIA GPU.",
    )
    sim.add_argument(
        "--auto-next-node",
        action="store_true",
        help="Auto-traverse the FSM (damp → stand → interpolate → teleop) instead of "
        "driving it manually (keyboard 0-9 / VR A-B buttons)",
    )
    sim.add_argument("-c", "--checkpoint", help="Checkpoint path under assets/")
    sim.add_argument("--debug", action="store_true")
    sim.add_argument(
        "--view",
        action="store_true",
        help="Live MJPEG preview in browser (default port 8766; see --view-port)",
    )
    sim.add_argument(
        "--view-port",
        type=int,
        default=8766,
        help="HTTP port for --view (default: 8766)",
    )
    sim.add_argument(
        "--no-wait",
        action="store_true",
        help="With --view, start immediately instead of waiting for browser Start / Enter",
    )
    sim.add_argument(
        "--fast",
        action="store_true",
        help="With --view, run the sim as fast as possible (uncap realtime pacing)",
    )
    sim.add_argument(
        "--window",
        action="store_true",
        help="Open the interactive MuJoCo viewer window (Docker: Linux with X11). "
        "On macOS with Docker, use --cpu --view instead.",
    )
    sim.add_argument(
        "--docker",
        action="store_true",
        help="Run inside yact g1-sim container (default when not already in container)",
    )
    sim.add_argument(
        "--host",
        action="store_true",
        help="Run on host Python (skip docker even if available)",
    )
    sim.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra deploy flags (prefix with -- if needed)",
    )
    sim.set_defaults(handler=cmd_sim)
