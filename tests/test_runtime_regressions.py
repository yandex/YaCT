"""Numerical and public-setting regressions; no robot, policy load or network."""
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import argparse
import logging

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from lib.assets.paths import resolve_checkpoint
from lib.cli import deploy_runner, paths, sim
from lib.controller.controller import ExternalVRTCPVelTracker
from lib.deploy.control.mujoco_env import G1MujocoEnv
from lib.deploy.deploy_utils.circular_buffer import CircularBuffer
from lib.deploy.deploy_utils import localization_launcher, ssh_utils
from lib.deploy.vr import localization
from lib.deploy.vr.vr_pose_tracker import VRPoseState
from lib.deploy.vr.vr_replay_wrapper import VRReplayWrapper
from lib.deploy.vr.vr_tracker_wrappers import TCPVelTrackerFromPose, PoseTrackerFromTCPVel


@pytest.mark.parametrize('window,capacity', [(20, 2048), (3, 7), (5, 5), (1, 4)])
def test_history_matches_deque_across_rollovers_and_reset(window, capacity):
    history = CircularBuffer(window, capacity)
    expected = deque([0.] * window, maxlen=window)
    for sample in range(2 * capacity + 5):
        history.append(torch.tensor([float(sample)]))
        expected.append(float(sample))
        torch.testing.assert_close(history.buffer[:, 0], torch.tensor(list(expected)))
    history.reset()
    history.append(torch.tensor([42.]))
    torch.testing.assert_close(history.buffer, torch.full((window, 1), 42.))


def pose(xyz=(0., 0., 0.), rotvec=(0., 0., 0.)):
    result = np.eye(4)
    result[:3, 3] = xyz
    result[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    return result


def state(times=(10., 1., 2.), poses=None):
    poses = poses or [pose(), pose(), pose()]
    return VRPoseState(**{f'{part}_{field}': value for field, values in [('pose', poses), ('timestamp', times)]
                         for part, value in zip(['head', 'left_wrist', 'right_wrist'], values)})


def assert_poses(actual, expected):
    for part in ['head', 'left_wrist', 'right_wrist']:
        np.testing.assert_allclose(getattr(actual, part + '_pose'), getattr(expected, part + '_pose'), atol=1e-8)


@pytest.mark.parametrize('advances', [(0., .02, 0.), (0., 0., .03), (.04, 0., 0.), (.04, .01, .02), (.02, .02, .02)])
def test_independent_pose_intervals_preserve_displacement(advances):
    initial = state()
    moved = [pose((i + .1, .2, .3), (.2, .1 * i, .3)) if delta else pose()
             for i, delta in enumerate(advances)]
    final = state(tuple(t + d for t, d in zip((10., 1., 2.), advances)), moved)
    source = SimpleNamespace(get_state=Mock(side_effect=[initial, final, final]))
    adapter = TCPVelTrackerFromPose(source)
    integrated = PoseTrackerFromTCPVel(adapter)
    integrated.set_current_state(initial)
    integrated.get_state()
    assert_poses(integrated.get_state(), final)
    assert_poses(integrated.get_state(), final)


@pytest.mark.parametrize('partial_consume', [False, True])
def test_async_motion_through_real_accumulator_and_interpolator(monkeypatch, partial_consume):
    initial = state()
    first = state((10., 1.02, 2.02), [pose(), pose((1., 0., 0.), (.4, 0., 0.)), pose()])
    final = state((10.02, 1.03, 2.06), [pose((.1, .2, .3)), pose((1., 1., 0.), (0., .6, .2)), pose((0., 0., 1.))])
    source = SimpleNamespace(get_state=Mock(side_effect=[initial, first, final]))
    adapter = TCPVelTrackerFromPose(source)
    external = ExternalVRTCPVelTracker()
    integrated = PoseTrackerFromTCPVel(external, cache_between_consumes=True)
    integrated.set_current_state(initial)
    # Model a slow producer so new data arrives with unfinished interpolation.
    clock = iter([1., 1.1, 1.2])
    monkeypatch.setattr('lib.controller.controller.time.monotonic', lambda: next(clock))
    external.set_state(adapter.get_state())
    external._outer_period_ema = .1
    external.set_state(adapter.get_state())
    if partial_consume:
        external.consume()
        integrated.consume()
        integrated.get_state()
    external.set_state(adapter.get_state())
    for _ in range(10):
        external.consume()
        assert external.get_state() is external.get_state()
        integrated.consume()
        result = integrated.get_state()
        assert integrated.get_state() is result
    assert_poses(result, final)


def test_absent_components_and_replay_freeze():
    initial = state()
    absent = state((10., None, 2.02), [pose(), None, pose((0., 1., 0.))])
    adapter = TCPVelTrackerFromPose(SimpleNamespace(get_state=Mock(side_effect=[initial, absent])))
    adapter.get_state()
    velocity = adapter.get_state()
    np.testing.assert_array_equal(velocity.left_wrist.xyz, np.zeros(3))
    np.testing.assert_allclose(velocity.right_wrist.xyz * velocity.dt, [0., 1., 0.])
    replay = VRReplayWrapper()
    replay.recorded_states = [state((1., 1., 1.)), state((1.02, 1.02, 1.02), [pose((1., 0., 0.))] * 3)]
    integrated = PoseTrackerFromTCPVel(TCPVelTrackerFromPose(replay))
    integrated.set_current_state(replay.recorded_states[0])
    integrated.get_state()
    replay.step()
    expected = integrated.get_state()
    replay.step()
    for _ in range(3):
        assert_poses(integrated.get_state(), expected)


def test_requested_checkpoint_must_exist(tmp_path, monkeypatch):
    monkeypatch.setenv('YACT_ASSETS_DIR', str(tmp_path))
    default = tmp_path / 'checkpoints/default.pt'
    default.parent.mkdir()
    default.write_bytes(b'not deserialized')
    explicit = tmp_path / 'chosen.pt'
    explicit.write_bytes(b'not deserialized')
    for requested in ['chosen.pt', str(explicit), 'checkpoints/default.pt']:
        expected = default if requested == 'checkpoints/default.pt' else explicit
        assert Path(resolve_checkpoint(requested)) == expected
    missing_absolute = tmp_path / 'missing' / 'chosen.pt'
    shadow = tmp_path / str(missing_absolute).lstrip('/')
    shadow.parent.mkdir(parents=True)
    shadow.write_bytes(b'wrong policy, must not be selected')
    for requested in ['missing.pt', str(tmp_path / 'missing.pt'), 'missing-policy', str(missing_absolute)]:
        with pytest.raises(FileNotFoundError):
            resolve_checkpoint(requested)


@pytest.mark.parametrize('version', ['0.1.0', '0.2.0'])
def test_lio_staging_and_launch_share_config(monkeypatch, version):
    cfg = paths.load_image_config('lio') | {'version': version, 'registry_image': 'lio:' + version}
    monkeypatch.setattr(paths, 'load_image_config', lambda _: cfg)
    monkeypatch.setattr(deploy_runner, 'load_image_config', lambda _: cfg)
    image = 'yact-lio:' + version
    registry = paths.registry_image_ref(cfg)
    config = ssh_utils.DockerConfig()
    assert config.image == image and config.registry_image == registry
    custom = ssh_utils.DockerConfig(image='custom:1', registry_image='example/custom:1')
    assert custom.image == 'custom:1' and custom.registry_image == 'example/custom:1'
    local_commands, remote_commands = [], []
    monkeypatch.setattr(ssh_utils, '_run_local', lambda cmd: local_commands.append(cmd) or True)
    def remote(command, **kwargs):
        remote_commands.append(command)
        # Only an older version exists; it must not satisfy a new configured tag.
        return (0, 'old-id' if command == 'docker images -q yact-lio:0.0.9' else '', '')
    monkeypatch.setattr(ssh_utils.SSHExecutor, 'run', lambda self, command, **kw: remote(command, **kw))
    monkeypatch.setattr(ssh_utils.SSHExecutor, 'put_file', lambda *args, **kw: True)
    monkeypatch.setenv('YACT_JETSON_PASSWORD', 'test-only')
    deploy_runner.stage_robot_image()
    assert remote_commands[0] == 'docker images -q ' + image
    assert ['docker', 'pull', '--platform', 'linux/arm64', registry] in local_commands
    assert ['docker', 'tag', registry, image] in local_commands
    assert any(cmd[:2] == ['docker', 'save'] and cmd[-1] == image for cmd in local_commands)
    assert any(cmd.startswith('docker load -i ') for cmd in remote_commands)
    # Exercise the in-container entry point while keeping SSH at the boundary.
    launched = []
    monkeypatch.setattr(localization_launcher.FastLioLauncher, 'launch',
                        lambda self: launched.append(self.container.docker_config) or True)
    assert localization_launcher.launch_fast_lio(password='test-only')
    assert launched[0].image == image
    assert f' {image} sleep infinity' in launched[0].get_start_command()
    # Already-present configured image requires no host pull/transfer.
    local_commands.clear()
    remote_commands.clear()
    monkeypatch.setattr(ssh_utils.SSHExecutor, 'run', lambda *args, **kw: (0, 'current-id', ''))
    deploy_runner.stage_robot_image()
    assert not local_commands


@pytest.mark.parametrize('record,preview', [(False, False), (False, True), (True, False)])
def test_single_video_honors_recording_preference(tmp_path, record, preview):
    env = G1MujocoEnv.__new__(G1MujocoEnv)
    env._closed = True
    env._save_videos = record
    env._live_view = Mock() if preview else None
    env._renderer = object()
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    env.render = lambda: frame
    env.configure_single_video(str(tmp_path / 'recording.mp4'))
    assert env._save_videos is record
    assert env._should_capture_frames() is (record or preview)
    if preview:
        env.pump_live_preview()
        env._live_view.update_frame.assert_called_once_with(frame)


@pytest.mark.parametrize('missing', ['imu', 'odometry', None])
def test_default_localization_keeps_warnings(monkeypatch, tmp_path, caplog, missing):
    monkeypatch.setattr(localization, 'ChannelSubscriber', lambda *args: Mock())
    observer = localization.LocalizationObservatorImpl(log_level=None, log_dir=str(tmp_path / 'logs'))
    if missing != 'imu':
        observer.imu_torso_state = SimpleNamespace(quaternion=[1., 0., 0., 0.])
    if missing != 'odometry':
        observer.odom_state = SimpleNamespace(x=1., y=2., z=3., q_w=1., q_x=0., q_y=0., q_z=0.)
    with caplog.at_level(logging.WARNING):
        observer.get_observation()
    assert bool(caplog.records) is (missing is not None)
    assert not (tmp_path / 'logs').exists()


def test_sim_manual_default_and_explicit_auto_help(capsys):
    parser = argparse.ArgumentParser()
    sim.add_parser(parser.add_subparsers())
    assert '--auto-next-node' not in sim._deploy_argv(parser.parse_args(['sim', '--view', '--no-wait']))
    assert '--auto-next-node' in sim._deploy_argv(parser.parse_args(['sim', '--auto-next-node']))
    with pytest.raises(SystemExit):
        parser.parse_args(['sim', '--help'])
    help_text = capsys.readouterr().out
    assert 'XQuartz' not in help_text
    assert '--cpu --view' in ' '.join(help_text.split())


@pytest.mark.parametrize('platform,flags,override,expected', [
    ('darwin', ['--window'], None, 1), ('darwin', ['--window'], 'custom:0', 0),
    ('darwin', ['--view'], None, 0), ('linux', ['--window'], None, 0),
])
def test_docker_display_contract(monkeypatch, capsys, platform, flags, override, expected):
    monkeypatch.setattr(deploy_runner.sys, 'platform', platform)
    monkeypatch.setenv('DISPLAY', ':0')
    monkeypatch.delenv('YACT_X11_DISPLAY', raising=False)
    if override:
        monkeypatch.setenv('YACT_X11_DISPLAY', override)
    monkeypatch.setattr(deploy_runner, '_container_running', lambda _: True)
    call = Mock(return_value=0)
    monkeypatch.setattr(deploy_runner.subprocess, 'call', call)
    assert deploy_runner.run_docker(flags, 'g1-sim-cpu') == expected
    if expected:
        call.assert_not_called()
        assert '--cpu --view' in capsys.readouterr().err
    else:
        call.assert_called_once()
        if override:
            assert 'DISPLAY=' + override in call.call_args.args[0]
