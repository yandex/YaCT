"""Real-robot deploy entrypoint."""

from __future__ import annotations

import argparse
import shutil
import sys

from lib.cli.deploy_runner import run_docker, stage_robot_image


def _deploy_argv(args: argparse.Namespace) -> list[str]:
    if not args.network:
        raise SystemExit("--network is required for real-robot deploy")
    argv = ["--network", args.network]
    if args.vr_replay:
        argv.extend(["--vr-replay", args.vr_replay])
    elif args.vr:
        argv.append("--vr")
    elif args.wasd:
        argv.append("--wasd")
    if args.checkpoint:
        argv.extend(["-c", args.checkpoint])
    if args.dex3:
        argv.append("--dex3")
    if args.auto_next_node:
        argv.append("--auto-next-node")
    if args.restart_localization:
        argv.append("--restart-localization")
    if args.debug:
        argv.append("--debug")
    if args.extra:
        extra = list(args.extra)
        if extra and extra[0] == "--":
            extra = extra[1:]
        argv.extend(extra)
    return argv


def cmd_deploy(args: argparse.Namespace) -> int:
    deploy_argv = _deploy_argv(args)

    if shutil.which("docker") is None:
        print("docker not found; install Docker to run deploy", file=sys.stderr)
        return 1

    # Host-side: stage the lio image onto the Jetson before the (daemonless)
    # deploy container starts. An explicitly empty Jetson password skips staging.
    stage_robot_image()

    try:
        return run_docker(deploy_argv)
    except KeyboardInterrupt:
        return 130


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    deploy = subparsers.add_parser("deploy", help="Run deploy on a real robot (deploy)")
    deploy.add_argument(
        "-n",
        "--network",
        required=True,
        help="Robot network interface (e.g. enp3s0)",
    )
    deploy.add_argument("--wasd", action="store_true", help="Keyboard WASD pose control")
    deploy.add_argument("--vr", action="store_true", help="Live VR via Vuer WebXR")
    deploy.add_argument(
        "--vr-replay",
        metavar="JSON",
        help="VR replay trajectory under assets/trajectories/",
    )
    deploy.add_argument("-c", "--checkpoint", help="Checkpoint path under assets/")
    deploy.add_argument("--dex3", action="store_true", help="Enable Dex3 hands")
    deploy.add_argument(
        "--auto-next-node",
        action="store_true",
        help="Automatically traverse FSM nodes",
    )
    deploy.add_argument(
        "--restart-localization",
        action="store_true",
        help="Force restart Fast-LIO on the Jetson",
    )
    deploy.add_argument("--debug", action="store_true")
    deploy.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra deploy flags (prefix with -- if needed)",
    )
    deploy.set_defaults(handler=cmd_deploy)
