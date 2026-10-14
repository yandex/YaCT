import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from lib.cli.paths import load_image_config, registry_image_ref, repo_root
from lib.deploy.deploy_utils.ssh_utils import DockerConfig


@pytest.fixture
def publish_environment(tmp_path, monkeypatch):
    docker = tmp_path / "docker"
    log = tmp_path / "docker.jsonl"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "args = sys.argv[1:]\n"
        "with open(os.environ['YACT_TEST_DOCKER_LOG'], 'a') as handle:\n"
        "    handle.write(json.dumps(args) + '\\n')\n"
        "if args[:1] == ['login']:\n"
        "    assert sys.stdin.read() == 'synthetic-ghcr-token'\n"
        "    sys.exit(0)\n"
        "if args[:2] == ['buildx', 'inspect']:\n"
        "    sys.exit(1)\n"
        "if args[:2] == ['buildx', 'create']:\n"
        "    sys.exit(0)\n"
        "if args[:2] == ['buildx', 'build']:\n"
        "    sys.exit(int(os.environ.get('YACT_TEST_BUILD_EXIT', '0')))\n"
        "raise SystemExit('Unexpected Docker operation')\n"
    )
    docker.chmod(0o755)
    runner_temp = tmp_path / "runner temp"
    python = runner_temp / "yact-publish-venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("YACT_TEST_DOCKER_LOG", str(log))
    monkeypatch.setenv("RUNNER_TEMP", str(runner_temp))
    monkeypatch.setenv("GHCR_TOKEN", "synthetic-ghcr-token")
    monkeypatch.setenv("GITHUB_ACTOR", "workflow-user")
    return log


def run_publish_step(image):
    workflow = yaml.safe_load((repo_root() / ".github/workflows/publish-images.yml").read_text())
    step = next(s for s in workflow["jobs"]["publish"]["steps"] if s["name"] == "Publish selected image")
    return subprocess.run(
        ["bash", "-e", "-o", "pipefail", "-c", step["run"]],
        cwd=repo_root(),
        env={**os.environ, "PUBLISH_IMAGE": image},
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("image,platforms", [
    ("g1-sim", "linux/amd64"),
    ("g1-sim-cpu", "linux/amd64,linux/arm64"),
    ("lio", "linux/arm64"),
])
def test_publish_step_matches_pull_reference_and_platforms(publish_environment, image, platforms):
    result = run_publish_step(image)
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in publish_environment.read_text().splitlines()]
    assert calls[0] == ["login", "ghcr.io", "-u", "workflow-user", "--password-stdin"]
    build = next(args for args in calls if args[:2] == ["buildx", "build"])
    tags = [build[i + 1] for i, value in enumerate(build[:-1]) if value == "-t"]
    reference = f"ghcr.io/yandex/yact/{image}:0.1.0"
    assert tags == [reference, f"ghcr.io/yandex/yact/{image}:latest"]
    assert registry_image_ref(load_image_config(image)) == reference
    assert build[build.index("--platform") + 1] == platforms
    assert build[build.index("--label") + 1] == "org.opencontainers.image.source=https://github.com/yandex/YaCT"
    assert "--push" in build
    assert "synthetic-ghcr-token" not in result.stdout + result.stderr + publish_environment.read_text()


def test_publish_failure_fails_the_workflow_step(publish_environment, monkeypatch):
    monkeypatch.setenv("YACT_TEST_BUILD_EXIT", "17")
    result = run_publish_step("g1-sim")
    assert result.returncode != 0
    assert "Done. Pushed:" not in result.stdout


def test_lio_staging_uses_published_reference_and_preserves_robot_tag():
    cfg = DockerConfig()
    assert cfg.registry_image == registry_image_ref(load_image_config("lio"))
    assert cfg.image == "yact-lio:0.1.0"
