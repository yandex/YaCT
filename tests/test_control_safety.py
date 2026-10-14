"""Control regressions using real control code with motor/input I/O replaced."""
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import Mock
import time

import numpy as np
import pytest

from lib.controller.utils.input import ControllerReader, InputReader
from lib.deploy.control.env import G1Env
from lib.deploy.control.mujoco_env import G1MujocoEnv
from lib.deploy.control.remote_controller import KeyMap
from lib.deploy.fsm.common import ActiveState, DisabledState, Edge, MoveToEdgeState, StartableNode
from lib.deploy.fsm.fsm import FiniteStateMachine
from lib.deploy.fsm.start_state_node import StartStateNode


@pytest.mark.parametrize('damp', ['damp', 0])
@pytest.mark.parametrize('first', [1, 'damp', 0])
def test_damping_preempts_pending_input(damp, first):
    reader = InputReader('damp')
    reader.push_fsm(first)
    reader.push_fsm(damp, cause='stop')
    reader.push_fsm(2)
    result = reader.consume_fsm_target()
    assert (result.target, result.cause) == (damp, 'stop')
    # Check pairing and one-shot consumption without relying on a queue implementation.
    reader.push_fsm(1)
    reader.push_fsm(damp, cause='stop')
    result = reader.consume_fsm_target()
    assert (result.target, result.cause) == (damp, 'stop')
    assert reader.consume_fsm_target().target is None
    reader.push_fsm(1)
    reader.push_fsm(2)
    assert reader.consume_fsm_target().target == 1


def test_keyboard_and_vr_damping(monkeypatch):
    from lib.vendor.util import stdin_cmd
    callbacks = {}
    monkeypatch.setattr(stdin_cmd, 'StdinCommands', lambda: SimpleNamespace(
        add_command=lambda key, fn: callbacks.update({key: fn}), start=lambda: None, stop=lambda: None))
    reader = InputReader('damp', use_stdin=True)
    reader.start()
    callbacks['1']()
    callbacks['x']()
    callbacks['2']()
    assert reader.consume_fsm_target().target == 'damp'
    reader.push_fsm(1)
    tracker = SimpleNamespace(get_controller_states=lambda: (None, SimpleNamespace(btn_a=True, btn_b=False)))
    reader.poll_vr_buttons(tracker)
    assert reader.consume_fsm_target().target == 'damp'
    reader.poll_vr_buttons(tracker)
    assert reader.consume_fsm_target().target is None
    reader.stop()


@pytest.mark.parametrize('order', [(KeyMap.A, KeyMap.X, KeyMap.B), (KeyMap.X, KeyMap.A)])
def test_physical_controller_damping(monkeypatch, order):
    from lib.deploy.deploy_utils import commander
    callbacks = {}
    def capture(remote, ranges, actions):
        callbacks.update(actions)
        return Mock()
    monkeypatch.setattr(commander, 'RemoteControllerCommander', capture)
    reader = ControllerReader('damp', None)
    for key in order:
        callbacks[key]()
    assert reader.consume_fsm_target().target == 'damp'
    assert reader.consume_fsm_target().target is None


def motor_env(cls=G1Env):
    env = cls.__new__(cls)
    env._closed = True  # No hardware/rendering resources were constructed.
    env.network = 'test-hardware-sink'
    env.config = SimpleNamespace(control_dt=.02, default_position=np.ones(2))
    env.get_current_motor_state = lambda: np.zeros(2)
    env._set_motor_target_positions = Mock()
    env.act = Mock()
    return env


@pytest.mark.parametrize('cls', [G1Env, G1MujocoEnv])
def test_interpolation_cancel_and_normal_control(cls):
    env = motor_env(cls)
    cancel = Event()
    cancel.set()
    env.go_to_state(np.ones(2), .06, cancel_event=cancel)
    env._set_motor_target_positions.assert_not_called()
    cancel.clear()
    env._set_motor_target_positions.side_effect = lambda _: cancel.set()
    env.go_to_state(np.ones(2), .06, cancel_event=cancel)
    assert env._set_motor_target_positions.call_count == 1
    env._set_motor_target_positions.reset_mock(side_effect=True)
    env.go_to_state(np.ones(2), .06)
    assert env._set_motor_target_positions.call_count == 3
    np.testing.assert_allclose(env._set_motor_target_positions.call_args.args[0], np.ones(2))
    env.go_to_state = Mock()
    env.go_to_initial_state(ask=False, cancel_event=cancel)
    assert env.go_to_state.call_args.kwargs['cancel_event'] is cancel


def test_start_state_deactivation_interrupts_real_interpolation():
    env = motor_env()
    wrote = Event()
    env._set_motor_target_positions.side_effect = lambda _: wrote.set()
    node = StartStateNode(env, console=Mock())
    node.activate()
    assert wrote.wait(2)
    started = time.monotonic()
    node.deactivate()
    assert time.monotonic() - started < .5
    count = env._set_motor_target_positions.call_count
    time.sleep(.04)
    assert env._set_motor_target_positions.call_count == count
    assert count < 10
    env.act.assert_not_called()
    assert isinstance(node.state, DisabledState)


class RecordingNode(StartableNode):
    def __init__(self, active):
        super().__init__()
        self.active = active
        self.activated = Event()
        self.activations = 0
        self.on_deactivate = lambda: None

    def activate(self):
        assert not self.active, 'two nodes active at once'
        self.active.add(self)
        self.activations += 1
        self.state = ActiveState()
        self.activated.set()

    def deactivate(self):
        self.on_deactivate()
        self.active.discard(self)
        self.state = DisabledState()


def machine():
    active = set()
    a, b, c = [RecordingNode(active) for _ in range(3)]
    for node, targets in [(a, [b, c]), (b, [a, c]), (c, [a, b])]:
        for target in targets:
            node.add_edge(Edge(node, target))
    return FiniteStateMachine(a, iter_time=.001, console=Mock()), a, b, c


def test_manual_and_automatic_transitions_share_lifecycle():
    fsm, a, b, c = machine()
    entered, release, second_deactivate, manual_started = Event(), Event(), Event(), Event()
    def block_first():
        if entered.is_set():
            second_deactivate.set()
        entered.set()
        assert release.wait(2)
    a.on_deactivate = block_first
    worker = fsm.run_async()
    assert a.activated.wait(2)
    a.state = MoveToEdgeState(payload=0, cause='auto')
    assert entered.wait(2)
    def manual():
        manual_started.set()
        fsm.move_by_name(c.name)
    thread = Thread(target=manual)
    thread.start()
    assert manual_started.wait(2)
    overlapped = second_deactivate.wait(.05)
    release.set()
    thread.join(2)
    try:
        assert not thread.is_alive()
        assert not overlapped
        assert c.activated.wait(2)
        assert (a.activations, b.activations, c.activations) == (1, 1, 1)
        assert len(fsm.transition_history) == 2
    finally:
        fsm.stop()
        worker.join(2)


@pytest.mark.parametrize('named', [False, True])
def test_pending_manual_command_cannot_reactivate_after_stop(named):
    fsm, a, b, _ = machine()
    a.activate()
    entered, release, attempted = Event(), Event(), Event()
    def block_deactivate():
        entered.set()
        assert release.wait(2)
    a.on_deactivate = block_deactivate
    stopping = Thread(target=lambda: fsm.stop(join=False))
    stopping.start()
    assert entered.wait(2)
    def move():
        attempted.set()
        fsm.move_by_name(b.name) if named else fsm.move_by_id(0)
    moving = Thread(target=move)
    moving.start()
    assert attempted.wait(2)
    release.set()
    stopping.join(2)
    moving.join(2)
    assert not stopping.is_alive() and not moving.is_alive()
    assert b.activations == 0
    assert not a.active
    fsm.run()  # initial activation after stop must also be rejected
    assert a.activations == 1


def test_stop_retains_live_worker_and_explicit_restart():
    fsm, a, b, _ = machine()
    entered, release = Event(), Event()
    def callback():
        entered.set()
        assert release.wait(2)
    worker = fsm.run_async(callbacks=[callback])
    assert entered.wait(2)
    fsm.stop(join=False)
    assert fsm.run_async() is worker
    fsm.move_by_id(0)
    assert b.activations == 0
    release.set()
    worker.join(2)
    a.activated.clear()
    restarted = fsm.run_async()
    assert restarted is not worker
    assert a.activated.wait(2)
    fsm.move_by_id(0)
    assert b.activations == 1
    fsm.stop()
    assert not restarted.is_alive()


def test_worker_exception_stop_rejects_late_manual_command():
    fsm, a, b, _ = machine()
    def broken_callback():
        raise RuntimeError('test callback failure')
    with pytest.raises(RuntimeError, match='test callback failure'):
        fsm.run(callbacks=[broken_callback])
    fsm.move_by_name(b.name)
    assert b.activations == 0
    assert not a.active


def test_stop_during_transition_does_not_activate_target():
    fsm, a, b, _ = machine()
    a.activate()
    entered, release = Event(), Event()
    def block_deactivate():
        entered.set()
        assert release.wait(2)
    a.on_deactivate = block_deactivate
    moving = Thread(target=lambda: fsm.move_by_id(0))
    moving.start()
    assert entered.wait(2)
    stopping = Thread(target=lambda: fsm.stop(join=False))
    stopping.start()
    assert fsm._stop_event.wait(2)
    release.set()
    moving.join(2)
    stopping.join(2)
    assert not moving.is_alive() and not stopping.is_alive()
    assert b.activations == 0
