"""Shared host/docker runners for the `sim` and `deploy` CLI commands.

Both commands turn an already-parsed CLI Namespace into a `deploy` argv (each
owns that mapping) and then run it the same two ways: in-process on the host, or
via `docker exec` in the image's container. That run machinery lives here so it
is not duplicated.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

from lib.cli.docker import _container_running, cmd_run
from lib.cli.paths import (
    container_name,
    host_user,
    load_image_config,
    repo_root,
)


def stage_robot_image() -> None:
    """Stage the robot-side lio image onto the Jetson, from the host.

    The deploy container has no Docker daemon, so the pull/save must happen on
    the host (which has Docker, registry access, and paramiko). The SSH transfer
    reuses the deploy SSH machinery. No-op when the Jetson password is empty or the
    image is already on the robot; best-effort (warns, never fails the deploy),
    since not every deploy uses Fast-LIO localization.
    """
    password = os.environ.get("YACT_JETSON_PASSWORD", "123")
    if not password:
        return  # no Jetson creds -> skip; in-container localization reports if needed
    try:
        from lib.deploy.deploy_utils.ssh_utils import (
            ContainerProcess,
            DockerConfig,
            SSHConfig,
            SSHExecutor,
        )
    except Exception as exc:  # paramiko / deps unavailable on this host
        print(f"lio staging skipped ({exc})", file=sys.stderr)
        return

    host = os.environ.get("YACT_JETSON_HOST", "192.168.123.164")
    user = os.environ.get("YACT_JETSON_USER", "unitree")
    # DockerConfig derives both image refs from the lio config. Provide the local
    # build paths as well, for hosts where the registry pull is unavailable.
    lio_cfg = load_image_config("lio")
    cfg = DockerConfig(
        dockerfile=str(repo_root() / lio_cfg["dockerfile"]),
        build_context=str(repo_root() / lio_cfg.get("build_context", ".")),
    )
    print(f"Ensuring '{cfg.image}' is on the robot ({user}@{host})...")
    container = ContainerProcess(
        SSHExecutor(SSHConfig(host=host, user=user, password=password)),
        cfg.container_name,
        cfg,
    )
    if not container.ensure_image_present():
        print(
            "Warning: could not stage the lio image on the robot; "
            "Fast-LIO localization may fail.",
            file=sys.stderr,
        )


def _detect_host_lan_ip() -> str | None:
    """Best-effort primary LAN IPv4 of this host (no packets sent)."""
    import socket

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def run_host(deploy_argv: list[str], *, mujoco_gl: str | None = None) -> int:
    """Run the deploy script in-process on the host."""
    if mujoco_gl is not None:
        os.environ.setdefault("MUJOCO_GL", mujoco_gl)
    from lib.controller.scripts import deploy as deploy_script

    parsed = deploy_script.parse_args(deploy_argv)
    deploy_script.main(parsed)
    return 0


def run_docker(deploy_argv: list[str], image_name: str = "g1-sim") -> int:
    """Run the deploy script via `docker exec` in the image's container."""
    cfg = load_image_config(image_name)
    name = container_name(image_name)

    # --window needs a reachable X display; resolve (and prepare, on macOS) it up
    # front and fail fast with guidance rather than starting the sim only for the
    # in-container viewer to die on "cannot open display".
    x11_display = None
    if "--window" in deploy_argv:
        override = os.environ.get("YACT_X11_DISPLAY")
        if override:
            x11_display = override
        elif sys.platform == "darwin":
            print(
                "error: the interactive MuJoCo window (--window) isn't supported on "
                "macOS with Docker. XQuartz only offers indirect GLX (OpenGL 1.4), but "
                "MuJoCo needs >=1.5, and software GL can't render to a remote X server.\n"
                "  Use the realtime browser stream instead:\n"
                "      ./yact sim --cpu --view      # then open http://localhost:8766\n"
                "  (--window works natively on Linux hosts with a display.)",
                file=sys.stderr,
            )
            return 1
        else:
            x11_display = os.environ.get("DISPLAY")
            if not x11_display:
                print(
                    "error: --window needs a display, but DISPLAY is not set.\n"
                    "  Start an X session (or omit --window).",
                    file=sys.stderr,
                )
                return 1

    if not _container_running(name):
        print(f"Starting container {name}...")
        rc = cmd_run(argparse.Namespace(image=image_name, stop=False, shell=False, no_gpu=False))
        if rc != 0:
            return rc

    python = cfg.get("python", "/opt/venv/bin/python")
    run_cfg = cfg.get("run") or {}
    cmd = ["docker", "exec", "-i"]
    if sys.stdin.isatty() and sys.stdout.isatty():
        cmd.append("-t")
    cmd.extend(["-u", host_user()])
    if run_cfg.get("workdir"):
        cmd.extend(["-w", str(run_cfg["workdir"])])
    # Forward explicit settings by name so SSH passwords stay out of argv.
    # Empty values matter: an empty Jetson password disables image staging.
    for var in ("YACT_LOG_STEP_TIME", "YACT_JETSON_HOST", "YACT_JETSON_USER", "YACT_JETSON_PASSWORD"):
        if var in os.environ:
            cmd.extend(["-e", var])
    # VR (--vr) serves a WebXR page the headset opens over the LAN. The address to
    # advertise is the *host's* LAN IP, which the container can't see (it detects
    # the docker/VM IP), so resolve it here and forward it via -e. Honor an
    # explicit YACT_VR_HOST override.
    if "--vr" in deploy_argv:
        vr_host = os.environ.get("YACT_VR_HOST") or _detect_host_lan_ip()
        if vr_host:
            cmd.extend(["-e", f"YACT_VR_HOST={vr_host}"])
        if os.environ.get("YACT_VR_DEBUG"):
            cmd.extend(["-e", f"YACT_VR_DEBUG={os.environ['YACT_VR_DEBUG']}"])
    # --window: forward the display resolved above (XQuartz TCP on macOS with a
    # software-GL hint; the host's DISPLAY on Linux, X socket mounted at create).
    if x11_display is not None:
        cmd.extend(["-e", f"DISPLAY={x11_display}"])
        if sys.platform == "darwin":
            # XQuartz's *indirect* GLX caps at OpenGL 1.4, but MuJoCo needs >=1.5.
            # So render with Mesa's software rasterizer (llvmpipe, GL 4.x) inside
            # the container and blit the result to XQuartz — needs libgl1-mesa-dri
            # in the image (the swrast driver), which the Dockerfiles install.
            cmd.extend(["-e", "LIBGL_ALWAYS_SOFTWARE=1", "-e", "GALLIUM_DRIVER=llvmpipe"])
    cmd.append(name)
    cmd.append(python)
    cmd.extend(["-m", "lib.controller.scripts.deploy"])
    cmd.extend(deploy_argv)
    env = os.environ.copy()
    for key, value in (run_cfg.get("env") or {}).items():
        env[key] = value
    return subprocess.call(cmd, env=env)
