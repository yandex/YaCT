"""Runtime environment helpers for YaCT OSS deploy."""

import os
from pathlib import Path
from typing import Optional


def in_docker(image: Optional[str] = None) -> bool:
    var = os.environ.get("YACT_IN_DOCKER")
    if image is None:
        return bool(var)
    base_var = os.path.basename(str(var))
    return var == image or base_var == image


_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
LOCAL_ROOT = os.environ.get("YACT_ROOT", str(_PACKAGE_ROOT.parent))


def _is_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


def host_path_for(path: str) -> Optional[str]:
    """Map a container repo path to the host checkout when ``YACT_HOST_ROOT`` is set."""
    if _is_url(path):
        return None
    host_root = os.environ.get("YACT_HOST_ROOT")
    container_root = os.environ.get("YACT_CONTAINER_ROOT")
    if not host_root or not container_root:
        return None

    abs_path = os.path.abspath(path)
    container_root = os.path.normpath(container_root)
    if abs_path != container_root and not abs_path.startswith(container_root + os.sep):
        return None

    rel = os.path.relpath(abs_path, container_root)
    host = os.path.normpath(os.path.join(host_root, rel))
    if host == abs_path:
        return None
    return host


def print_path_lines(
    prefix: str,
    path: str,
    *,
    host_label: str = "host",
    note: Optional[str] = None,
) -> None:
    """Print a path and, in docker, the equivalent host checkout path."""
    line = f"{prefix}{path}"
    if note:
        line += f"  ({note})"
    print(line, flush=True)
    if _is_url(path):
        return
    host = host_path_for(path)
    if host:
        pad = " " * len(prefix)
        print(f"{pad}{host_label}: {host}", flush=True)
