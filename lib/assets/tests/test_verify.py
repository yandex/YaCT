from pathlib import Path

import pytest

from lib.assets import verify


@pytest.fixture
def asset_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(verify, "assets_root", lambda: tmp_path)
    return tmp_path


def test_verify_assets_reports_only_missing_required_assets(asset_root: Path) -> None:
    for relative in verify.REQUIRED_ASSETS[1:]:
        path = asset_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    required_issues, missing_optional = verify.verify_assets()

    assert required_issues == [verify.AssetIssue(verify.REQUIRED_ASSETS[0], "missing")]
    assert missing_optional == list(verify.OPTIONAL_ASSETS)


def test_verify_assets_accepts_regular_git_files(asset_root: Path) -> None:
    for relative in (*verify.REQUIRED_ASSETS, *verify.OPTIONAL_ASSETS):
        path = asset_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"ordinary git content")

    assert verify.verify_assets() == ([], [])
