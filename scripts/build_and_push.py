#!/usr/bin/env python3
"""Build and push YaCT Docker images to the registry, multi-arch where supported.

    scripts/build_and_push.py                 # both images
    scripts/build_and_push.py g1-sim-cpu       # one image
    scripts/build_and_push.py lio              # Jetson image

IMAGE defaults to: g1-sim g1-sim-cpu

Arch matrix:
    g1-sim      linux/amd64               CUDA + PyTorch GPU base (no arm64)
    g1-sim-cpu  linux/amd64,linux/arm64   CPU torch + slim base (multi-arch)
    lio         linux/arm64               Jetson Fast-LIO

Each image is tagged "<REGISTRY>/<name>:<version>" and "<REGISTRY>/<name>:latest".

Prerequisites:
    - docker buildx (ships with modern Docker)
    - `docker login ghcr.io` already done (this script never logs in for you)
    - For arm64 cross-build on an amd64 host, register QEMU once:
        docker run --privileged --rm tonistiigi/binfmt --install arm64
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.cli.paths import REGISTRY, load_image_config, registry_image_ref, repo_root

BUILDER = "yact-multiarch"

# Per-image build platforms. g1-sim is amd64-only: its CUDA + PyTorch GPU
# base targets amd64 workstations and is not built for arm64.
PLATFORMS = {
    "g1_sim": "linux/amd64",
    "g1_sim_cpu": "linux/amd64,linux/arm64",
    "lio": "linux/arm64",  # Jetson-only
}


def _run(cmd: list[str]) -> None:
    print("+", shlex.join(cmd))
    subprocess.run(cmd, check=True)


def _ensure_builder() -> None:
    exists = subprocess.run(
        ["docker", "buildx", "inspect", BUILDER],
        capture_output=True,
        text=True,
    ).returncode == 0
    if not exists:
        print(f"Creating buildx builder '{BUILDER}'...")
        # network=host: the buildkit container needs the host's DNS/routing to
        # reach the base-image registries (matches the CLI's `docker build
        # --network=host`); without it, base-image pulls time out.
        _run([
            "docker", "buildx", "create",
            "--name", BUILDER,
            "--driver", "docker-container",
            "--driver-opt", "network=host",
        ])


def _push_image(image: str) -> None:
    cfg = load_image_config(image)
    folder = cfg["_folder"]
    ref = registry_image_ref(cfg)            # ghcr.io/yandex/yact/g1-sim:0.1.0
    repo_tag, _, _version = ref.rpartition(":")
    latest = f"{repo_tag}:latest"
    platforms = PLATFORMS.get(folder, "linux/amd64")
    dockerfile = repo_root() / cfg["dockerfile"]
    context = repo_root() / cfg.get("build_context", ".")

    print(f"==> {image}  ({platforms})  ->  {ref}  (+ :latest)")
    cmd = [
        "docker", "buildx", "build",
        "--builder", BUILDER,
        "--platform", platforms,
        "-f", str(dockerfile),
        "-t", ref,
        "-t", latest,
        "--label", "org.opencontainers.image.source=https://github.com/yandex/YaCT",
        "--push",
        str(context),
    ]
    _run(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and push YaCT Docker images (multi-arch where supported).",
    )
    parser.add_argument(
        "images",
        nargs="*",
        default=["g1-sim", "g1-sim-cpu"],
        help="Images to push (default: g1-sim g1-sim-cpu)",
    )
    args = parser.parse_args()
    images = args.images or ["g1-sim", "g1-sim-cpu"]

    _ensure_builder()
    for image in images:
        _push_image(image)

    print(f"Done. Pushed: {' '.join(images)}  ->  {REGISTRY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
