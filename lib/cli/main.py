"""YaCT command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lib.cli import deploy as deploy_cmd
from lib.cli import docker as docker_cmd
from lib.cli import setup as setup_cmd
from lib.cli import sim as sim_cmd


def _pkg_version() -> str:
    try:
        from importlib.metadata import version

        return version("yact")
    except Exception:
        pass
    pyproject = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"
    if pyproject.is_file():
        for line in pyproject.read_text(encoding="utf-8").splitlines():
            if line.startswith("version"):
                _, _, raw = line.partition("=")
                return raw.strip().strip('"').strip("'")
    return "0.0.0"


def _cmd_version(_args: argparse.Namespace) -> int:
    print(_pkg_version())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yact",
        description="YaCT — Unitree G1 teleoperation deploy and MuJoCo sim",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    version_parser = sub.add_parser("version", help="Print package version")
    version_parser.set_defaults(handler=_cmd_version)

    docker_cmd.add_parser(sub)
    sim_cmd.add_parser(sub)
    deploy_cmd.add_parser(sub)
    setup_cmd.add_parser(sub)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 1
    try:
        return int(handler(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
