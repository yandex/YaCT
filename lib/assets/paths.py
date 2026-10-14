"""Resolve inference assets from the in-repo assets/ tree."""

import os
from pathlib import Path


def assets_root() -> Path:
    override = os.environ.get("YACT_ASSETS_DIR")
    if override:
        return Path(override).expanduser().resolve()
    # .../YaCT/lib/assets/paths.py -> .../YaCT/assets
    return Path(__file__).resolve().parent.parent.parent / "assets"


def _normalize_relative(relative: str) -> str:
    return relative.replace("\\", "/").lstrip("/")


def resolve_asset(relative: str) -> str:
    """Return absolute path to an asset under assets/; assets must already exist."""
    if os.path.isabs(relative) and os.path.exists(relative):
        return relative
    path = assets_root() / _normalize_relative(relative)
    if not path.exists():
        raise FileNotFoundError(
            f"Asset not found: {relative!r} (looked at {path}). "
            "Ensure assets are present or set YACT_ASSETS_DIR."
        )
    return str(path)


def resolve_checkpoint(checkpoint: str) -> str:
    """Resolve a checkpoint path (absolute or assets-relative)."""
    if os.path.isabs(checkpoint):
        if os.path.isfile(checkpoint):
            return checkpoint
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint!r}")
    normalized = _normalize_relative(checkpoint)
    if normalized.startswith("checkpoints/") or normalized.endswith(".pt"):
        candidate = assets_root() / normalized
        if candidate.is_file():
            return str(candidate)
    return resolve_asset(checkpoint)
