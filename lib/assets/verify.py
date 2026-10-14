"""Verify inference assets under assets/."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lib.assets.paths import assets_root

REQUIRED_ASSETS: tuple[str, ...] = (
    "checkpoints/default.pt",
    "trajectories/t_pose_traj.json",
    "mujoco/g1_29dof_rev_1_0_custom_actuators/g1_29dof_rev_1_0.xml",
    "fk/g1_fk.urdf",
)

OPTIONAL_ASSETS: tuple[str, ...] = (
    "trajectories/walk_forward.json",
    "tts/alerts/motor_temp_warning.wav",
)


@dataclass(frozen=True)
class AssetIssue:
    path: str
    reason: str


def _asset_path(relative: str) -> Path:
    return assets_root() / relative


def verify_assets() -> tuple[list[AssetIssue], list[str]]:
    """Return (required issues, missing optional paths)."""
    required_issues: list[AssetIssue] = []
    missing_optional: list[str] = []

    for relative in REQUIRED_ASSETS:
        path = _asset_path(relative)
        if not path.exists():
            required_issues.append(AssetIssue(relative, "missing"))

    for relative in OPTIONAL_ASSETS:
        path = _asset_path(relative)
        if not path.exists():
            missing_optional.append(relative)

    return required_issues, missing_optional


def format_asset_report(
    required_issues: list[AssetIssue],
    missing_optional: list[str],
) -> str:
    lines: list[str] = []
    if required_issues:
        lines.append("Required assets missing or incomplete:")
        for issue in required_issues:
            lines.append(f"  - {issue.path}: {issue.reason}")
    if missing_optional:
        lines.append("Optional assets missing:")
        for relative in missing_optional:
            lines.append(f"  - {relative}")
    return "\n".join(lines)
