"""Environment and asset smoke checks."""

from __future__ import annotations

import argparse
import shutil
import sys

from lib.assets.verify import (
    format_asset_report,
    verify_assets,
)
from lib.cli import docker as docker_cmd
from lib.cli.paths import repo_root
from lib.vendor.env import in_docker


def _check(label: str, ok: bool, detail: str = "", *, required: bool = True) -> bool:
    mark = "ok" if ok else ("skip" if not required else "FAIL")
    suffix = f" ({detail})" if detail else ""
    print(f"  [{mark}] {label}{suffix}")
    return ok if required else True


def _verify_assets_step() -> bool:
    required_issues, missing_optional = verify_assets()
    if missing_optional:
        print(format_asset_report([], missing_optional))

    if required_issues:
        print(format_asset_report(required_issues, []))
        print("  hint: restore the missing files under assets/ or set YACT_ASSETS_DIR")
        return False

    print("  [ok] required assets present")
    return True


def cmd_setup(args: argparse.Namespace) -> int:
    ok = True
    print("YaCT setup")
    print(f"  repo: {repo_root()}")

    ok &= _check("python", sys.version_info >= (3, 10), sys.version.split()[0])
    ok &= _check("docker CLI", shutil.which("docker") is not None)
    ok &= _check(
        "nvidia-smi",
        shutil.which("nvidia-smi") is not None or args.no_gpu,
        "skipped" if args.no_gpu else "",
    )
    _check(
        "in g1_sim container",
        in_docker("g1_sim"),
        required=False,
    )

    if not args.no_assets:
        print("Assets:")
        ok &= _verify_assets_step()

    if not args.no_docker and shutil.which("docker") is not None:
        print("Docker image (g1-sim):")
        rc = docker_cmd.ensure_image(
            "g1-sim",
            force_build=args.build_only,
            skip_pull=args.build_only,
        )
        ok &= rc == 0

    if ok:
        print("\nReady. Try: ./yact sim")
        print("         ./yact docker shell g1-sim   (optional: manual container work)")
        print("         ./yact deploy -n <iface> --wasd   (real robot)")
        return 0

    print("\nFix failures above (see README assets/ section).")
    return 1


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    setup = subparsers.add_parser("setup", help="Check toolchain, assets/, and docker image")
    setup.add_argument(
        "--no-assets",
        action="store_true",
        help="Skip asset verification",
    )
    setup.add_argument(
        "--build-only",
        action="store_true",
        help="Build g1-sim locally (skip registry pull)",
    )
    setup.add_argument(
        "--no-docker",
        action="store_true",
        help="Skip docker image pull/build",
    )
    setup.add_argument(
        "--no-gpu",
        action="store_true",
        help="Do not require nvidia-smi on the host",
    )
    setup.set_defaults(handler=cmd_setup)
