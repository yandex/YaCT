"""Docker build/run/exec helpers (stdlib + docker CLI only)."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from typing import Any, Sequence

from lib.assets.paths import assets_root
from lib.cli.paths import (
    container_name,
    host_user,
    load_image_config,
    local_image_tag,
    registry_image_ref,
    repo_root,
    user_image_tag,
)
from lib.cli.user_image import build_user_image


def _run(
    cmd: Sequence[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    print("+", shlex.join(cmd))
    return subprocess.run(cmd, check=check, text=True)


def _image_exists_locally(tag: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", tag],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _pull_image(ref: str) -> bool:
    print(f"Pulling {ref}...")
    result = subprocess.run(["docker", "pull", ref], text=True)
    if result.returncode != 0:
        print(f"Pull failed for {ref}", file=sys.stderr)
        return False
    print(f"Pulled {ref}")
    return True


def _build_base_image(cfg: dict[str, Any], *, tag: str | None = None) -> int:
    dockerfile = repo_root() / cfg["dockerfile"]
    context = repo_root() / cfg.get("build_context", ".")
    image_tag = tag or registry_image_ref(cfg)
    cmd = [
        "docker",
        "build",
        "--network=host",
        "-f",
        str(dockerfile),
    ]
    for key, value in (cfg.get("build_args") or {}).items():
        cmd.extend(["--build-arg", f"{key}={value}"])
    cmd.extend(["-t", image_tag, str(context)])
    result = _run(cmd, check=False)
    if result.returncode != 0:
        print(f"Build failed for {image_tag}", file=sys.stderr)
        return result.returncode
    print(f"Built {image_tag}")
    return 0


def ensure_image(
    image_name: str,
    *,
    force_build: bool = False,
    skip_pull: bool = False,
) -> int:
    """Resolve docker image: user tag, then pull/build base + user overlay."""
    cfg = load_image_config(image_name)
    user_tag = user_image_tag(cfg)
    base_tag = registry_image_ref(cfg)
    slug = cfg.get("name", image_name)

    if force_build:
        print(f"Building {slug} (--force)...")
        rc = _build_base_image(cfg)
        if rc != 0:
            return rc
        return build_user_image(base_tag, user_tag)

    if _image_exists_locally(user_tag):
        print(f"Using local image {user_tag}")
        return 0

    if not skip_pull and _pull_image(base_tag):
        return build_user_image(base_tag, user_tag)

    if _image_exists_locally(base_tag):
        print(f"Building user layer for {slug}...")
        return build_user_image(base_tag, user_tag)

    print(f"Building {slug} locally...")
    rc = _build_base_image(cfg)
    if rc != 0:
        return rc
    return build_user_image(base_tag, user_tag)


def _exec_cmd(cfg: dict[str, Any], name: str, command: list[str], *, interactive: bool) -> list[str]:
    run_cfg = cfg.get("run") or {}
    cmd = ["docker", "exec"]
    if interactive:
        cmd.append("-i")
        if sys.stdin.isatty() and sys.stdout.isatty():
            cmd.append("-t")
    cmd.extend(["-u", host_user()])
    if run_cfg.get("workdir"):
        cmd.extend(["-w", str(run_cfg["workdir"])])
    cmd.append(name)
    cmd.extend(command)
    return cmd


def _build_run_args(cfg: dict[str, Any], *, no_gpu: bool) -> list[str]:
    run_cfg = cfg.get("run") or {}
    args = ["docker", "run", "-d", "--name", container_name(cfg.get("name", cfg["_folder"]))]

    if not no_gpu and run_cfg.get("gpus"):
        args.extend(["--gpus", str(run_cfg["gpus"])])
    if run_cfg.get("network"):
        args.extend(["--net", str(run_cfg["network"])])
    if run_cfg.get("shm_size"):
        args.extend(["--shm-size", str(run_cfg["shm_size"])])
    if run_cfg.get("privileged"):
        args.append("--privileged")

    mount_repo = run_cfg.get("mount_repo")
    if mount_repo:
        root = repo_root()
        args.extend(["-v", f"{root}:{mount_repo}"])
        args.extend(["-e", f"YACT_HOST_ROOT={root}"])
        args.extend(["-e", f"YACT_CONTAINER_ROOT={mount_repo}"])
        if os.environ.get("YACT_ASSETS_DIR"):
            # Keep asset resolution identical inside the container.
            args.extend(["--mount", f"type=bind,src={assets_root()},dst={mount_repo}/assets"])
    if run_cfg.get("workdir"):
        args.extend(["-w", str(run_cfg["workdir"])])

    for key, value in (run_cfg.get("env") or {}).items():
        args.extend(["-e", f"{key}={value}"])

    for device in run_cfg.get("devices") or []:
        args.extend(["--device", str(device)])
    for volume in run_cfg.get("volumes") or []:
        args.extend(["-v", str(volume)])

    # X11 socket for the interactive MuJoCo viewer (--window) on Linux hosts.
    # Harmless when unused; macOS reaches XQuartz over TCP instead (no socket).
    if sys.platform.startswith("linux") and os.path.isdir("/tmp/.X11-unix"):
        args.extend(["-v", "/tmp/.X11-unix:/tmp/.X11-unix:rw"])

    args.extend(["--entrypoint", ""])
    args.append(local_image_tag(cfg))
    args.extend(["sleep", "infinity"])
    return args


def cmd_pull(args: argparse.Namespace) -> int:
    cfg = load_image_config(args.image)
    base_tag = registry_image_ref(cfg)
    if not _pull_image(base_tag):
        return 1
    return build_user_image(base_tag, user_image_tag(cfg))


def cmd_stop(args: argparse.Namespace) -> int:
    return cmd_run(argparse.Namespace(
        image=args.image,
        stop=True,
        shell=False,
        no_gpu=False,
    ))


def cmd_build(args: argparse.Namespace) -> int:
    if args.tag:
        cfg = load_image_config(args.image)
        if args.force:
            rc = _build_base_image(cfg, tag=args.tag)
            if rc != 0:
                return rc
            return build_user_image(args.tag, user_image_tag(cfg))
        if _image_exists_locally(args.tag):
            user_tag = user_image_tag(cfg)
            if _image_exists_locally(user_tag):
                print(f"Using local images {args.tag} and {user_tag}")
                return 0
            print(f"Building user layer for {user_tag}...")
            return build_user_image(args.tag, user_tag)
        cfg = dict(cfg)
        cfg["registry_image"] = args.tag
        if not args.no_pull and _pull_image(args.tag):
            return build_user_image(args.tag, user_image_tag(cfg))
        rc = _build_base_image(cfg, tag=args.tag)
        if rc != 0:
            return rc
        return build_user_image(args.tag, user_image_tag(cfg))

    return ensure_image(
        args.image,
        force_build=args.force,
        skip_pull=args.no_pull,
    )


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_image_config(args.image)
    name = container_name(args.image)

    if args.stop:
        if _container_exists(name):
            _run(["docker", "rm", "-f", name], check=False)
            print(f"Removed container {name}")
        else:
            print(f"No container {name}")
        return 0

    if _container_running(name):
        print(f"Container {name} is already running")
        if args.shell:
            return cmd_shell(argparse.Namespace(image=args.image))
        return 0

    if _container_exists(name):
        _run(["docker", "start", name])
        print(f"Started existing container {name}")
        if args.shell:
            return cmd_shell(argparse.Namespace(image=args.image))
        return 0

    if not _image_exists_locally(local_image_tag(cfg)):
        rc = ensure_image(args.image)
        if rc != 0:
            return rc

    if args.shell:
        _run(_build_run_args(cfg, no_gpu=args.no_gpu))
        print(f"Started container {name}")
        return cmd_shell(argparse.Namespace(image=args.image))

    _run(_build_run_args(cfg, no_gpu=args.no_gpu))
    print(f"Started container {name}")
    print(f"  shell:  ./yact docker shell {cfg.get('name', args.image)}")
    print(f"  exec:   ./yact docker exec {cfg.get('name', args.image)} -- <cmd>")
    print(f"  stop:   ./yact docker stop {cfg.get('name', args.image)}")
    return 0


def _container_exists(name: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", name],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def _container_running(name: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", name],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def cmd_shell(args: argparse.Namespace) -> int:
    cfg = load_image_config(args.image)
    name = container_name(args.image)
    if not _container_running(name):
        print(f"Container {name} is not running. Start with: ./yact sim or ./yact docker run {args.image}")
        return 1
    cmd = _exec_cmd(cfg, name, ["/bin/bash"], interactive=True)
    return subprocess.call(cmd)


def cmd_exec(args: argparse.Namespace) -> int:
    cfg = load_image_config(args.image)
    name = container_name(args.image)
    if not _container_running(name):
        print(f"Container {name} is not running. Start with: ./yact sim or ./yact docker run {args.image}")
        return 1
    if not args.command:
        print("Missing command after `--`")
        return 1
    command = list(args.command)
    if command[0] == "--":
        command = command[1:]
    if not command:
        print("Missing command after `--`")
        return 1
    cmd = _exec_cmd(cfg, name, command, interactive=True)
    return subprocess.call(cmd)


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    docker = subparsers.add_parser("docker", help="Build and run YaCT docker images")
    sub = docker.add_subparsers(dest="docker_command", required=True)

    build = sub.add_parser(
        "build",
        help="Ensure image exists (user layer, pull base, or build)",
    )
    build.add_argument("image", help="Image name (g1-sim, lio)")
    build.add_argument("--tag", help="Override base image tag")
    build.add_argument(
        "--force",
        action="store_true",
        help="Rebuild base and user images locally",
    )
    build.add_argument(
        "--no-pull",
        action="store_true",
        help="Skip registry pull; use local base or build",
    )
    build.set_defaults(handler=cmd_build)

    pull = sub.add_parser("pull", help="Pull base image and build user layer")
    pull.add_argument("image", help="Image name (g1-sim, lio)")
    pull.set_defaults(handler=cmd_pull)

    stop = sub.add_parser("stop", help="Remove a long-lived container")
    stop.add_argument("image", help="Image name (g1-sim, lio)")
    stop.set_defaults(handler=cmd_stop)

    run = sub.add_parser("run", help="Start a long-lived container")
    run.add_argument("image", help="Image name (g1-sim, lio)")
    run.add_argument("--stop", action="store_true", help="Remove the container")
    run.add_argument("--shell", action="store_true", help="Open an interactive shell")
    run.add_argument("--no-gpu", action="store_true", help="Do not pass --gpus all")
    run.set_defaults(handler=cmd_run)

    shell = sub.add_parser("shell", help="Open bash in a running container")
    shell.add_argument("image", help="Image name (g1-sim, lio)")
    shell.set_defaults(handler=cmd_shell)

    exe = sub.add_parser("exec", help="Run a command in a running container")
    exe.add_argument("image", help="Image name (g1-sim, lio)")
    exe.add_argument("command", nargs=argparse.REMAINDER, help="Command after --")
    exe.set_defaults(handler=cmd_exec)
