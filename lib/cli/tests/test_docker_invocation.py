import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from lib.cli import deploy_runner, docker
from lib.cli.paths import load_image_config, repo_root


@pytest.mark.parametrize("stdin_tty,stdout_tty", [(True, True), (True, False), (False, True), (False, False)])
def test_exec_preserves_input_without_forcing_a_terminal(monkeypatch, stdin_tty, stdout_tty):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: stdin_tty)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: stdout_tty)
    cfg = load_image_config("g1-sim")
    cmd = docker._exec_cmd(cfg, "test-container", ["python", "--version"], interactive=True)
    assert "-i" in cmd
    assert ("-t" in cmd) == (stdin_tty and stdout_tty)


@pytest.mark.parametrize("password", [None, "", "synthetic-test-password"])
def test_deploy_forwards_explicit_jetson_settings_without_values_in_argv(monkeypatch, password):
    names = ("YACT_JETSON_HOST", "YACT_JETSON_USER", "YACT_JETSON_PASSWORD")
    for name in names:
        monkeypatch.delenv(name, raising=False)
    if password is not None:
        monkeypatch.setenv(names[0], "192.0.2.10")
        monkeypatch.setenv(names[1], "test-user")
        monkeypatch.setenv(names[2], password)
    monkeypatch.setattr(deploy_runner, "_container_running", lambda _: True)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    calls = []

    def capture(cmd, *, env):
        calls.append((cmd, env))
        return 23

    monkeypatch.setattr(deploy_runner.subprocess, "call", capture)
    assert deploy_runner.run_docker(["--network", "test-interface"]) == 23
    cmd, env = calls[0]
    forwarded = [cmd[i + 1] for i, arg in enumerate(cmd[:-1]) if arg == "-e"]
    assert set(names).intersection(forwarded) == (set(names) if password is not None else set())
    if password is not None:
        assert env[names[2]] == password
        assert not any(arg.startswith("YACT_JETSON_PASSWORD=") for arg in cmd)
        if password:
            assert password not in " ".join(cmd)
    assert "-i" in cmd and "-t" not in cmd and "-it" not in cmd


@pytest.mark.parametrize("image", ["g1-sim", "g1-sim-cpu"])
def test_external_assets_mount_uses_the_resolved_host_directory(monkeypatch, tmp_path, image):
    assets = tmp_path / "external assets"
    assets.mkdir()
    monkeypatch.setenv("YACT_ASSETS_DIR", str(assets))
    cfg = load_image_config(image)
    args = docker._build_run_args(cfg, no_gpu=True)
    assert args[args.index("--mount") + 1] == (
        f"type=bind,src={assets.resolve()},dst={cfg['run']['mount_repo']}/assets"
    )
    monkeypatch.delenv("YACT_ASSETS_DIR")
    assert "--mount" not in docker._build_run_args(cfg, no_gpu=True)


@pytest.fixture
def fake_docker(tmp_path, monkeypatch):
    executable = tmp_path / "docker"
    log = tmp_path / "docker.jsonl"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "with open(os.environ['YACT_TEST_DOCKER_LOG'], 'a') as handle:\n"
        "    handle.write(json.dumps(args) + '\\n')\n"
        "if args[:1] == ['inspect']:\n"
        "    sys.exit(1)\n"
        "if args[:2] == ['image', 'inspect']:\n"
        "    sys.exit(0)\n"
        "if args[:1] == ['run']:\n"
        "    print('test-container')\n"
        "    sys.exit(0)\n"
        "if args[:1] == ['exec']:\n"
        "    sys.exit(9 if '-t' in args or '-it' in args else 0)\n"
        "raise SystemExit('Unexpected Docker operation: ' + args[0])\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("YACT_TEST_DOCKER_LOG", str(log))
    monkeypatch.setenv("YACT_NO_VENV", "1")
    monkeypatch.delenv("YACT_IN_DOCKER", raising=False)
    return log


def test_cpu_cli_creates_asset_mount_and_runs_without_a_tty(fake_docker, monkeypatch, tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    monkeypatch.setenv("YACT_ASSETS_DIR", str(assets))
    result = subprocess.run(
        [sys.executable, str(repo_root() / "yact"), "sim", "--cpu", "--view", "--no-wait"],
        input="",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    calls = [json.loads(line) for line in fake_docker.read_text().splitlines()]
    run = next(args for args in calls if args[0] == "run")
    assert f"type=bind,src={assets},dst=/workspace/yact/assets" in run
    execute = next(args for args in calls if args[0] == "exec")
    assert "--no-wait" in execute and "--view" in execute
    assert execute[execute.index("--device") + 1] == "cpu"
