"""Repo paths and docker image config loading."""

from __future__ import annotations

import getpass
from pathlib import Path
from typing import Any

import yaml

# Registry where shareable base images are pushed/pulled. Shared by the CLI
# (docker pull/build) and scripts/build_and_push.py.
REGISTRY = "ghcr.io/yandex/yact"

_IMAGE_ALIASES = {
    "g1-sim": "g1_sim",
    "g1_sim": "g1_sim",
    "g1sim": "g1_sim",
    "lio": "lio",
}


def repo_root() -> Path:
    """YaCT repository root (parent of the `lib` package)."""
    return Path(__file__).resolve().parent.parent.parent


def host_user() -> str:
    return getpass.getuser()


def normalize_image_name(name: str) -> str:
    key = name.strip().lower().replace("_", "-")
    if key in _IMAGE_ALIASES:
        return _IMAGE_ALIASES[key]
    folder = key.replace("-", "_")
    if (repo_root() / "docker" / folder / "config.yaml").is_file():
        return folder
    raise ValueError(f"Unknown docker image {name!r}. Try: g1-sim, lio")


def image_config_path(folder: str) -> Path:
    path = repo_root() / "docker" / folder / "config.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing docker config: {path}")
    return path


def load_image_config(name: str) -> dict[str, Any]:
    folder = normalize_image_name(name)
    with image_config_path(folder).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_folder"] = folder
    config["_repo_root"] = str(repo_root())
    return config


def registry_image_ref(cfg: dict[str, Any]) -> str:
    """Shareable base image ref (pull/push / Dockerfile build), in REGISTRY.

    A name that is already fully qualified (contains a registry host, i.e. a
    "/") is returned as-is; bare names are prefixed with REGISTRY.
    """
    if cfg.get("registry_image"):
        name = str(cfg["registry_image"])
    elif cfg.get("image"):
        name = str(cfg["image"])
    else:
        slug = cfg.get("name", cfg["_folder"]).replace("_", "-")
        name = f"yact-{slug}:{cfg['version']}"
    if "/" in name:
        return name
    return f"{REGISTRY}/{name}"


def user_image_tag(cfg: dict[str, Any]) -> str:
    """Per-user image tag used for docker run (host uid/gid overlay)."""
    slug = cfg.get("name", cfg["_folder"]).replace("_", "-")
    return f"{host_user()}/yact-{slug}:{cfg['version']}"


def local_image_tag(cfg: dict[str, Any]) -> str:
    """Image tag for running containers (user overlay)."""
    return user_image_tag(cfg)


def container_name(image_name: str) -> str:
    folder = normalize_image_name(image_name)
    cfg = load_image_config(folder)
    slug = cfg.get("name", folder.replace("_", "-"))
    return f"yact__{host_user()}__{slug}"
