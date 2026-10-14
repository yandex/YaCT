import threading
from dataclasses import dataclass
import numpy as np
from typing import Optional, Union
from lib.controller.config.commands import CommandRanges
from lib.controller.utils.edge_detector import EdgeDetector


@dataclass
class PendingFSMTarget:
    target: Union[int, str]
    cause: Optional[str] = None


class InputReader:
    """Unified input reader: optional stdin keyboard + VR buttons (polled).

    All inputs produce either an fsm_target or a velocity command.
    FSM targets use one-shot semantics: one press = one transition.
    """

    def __init__(
        self,
        damp_node_name: str,
        ranges=None,
        max_abs: float = 1.0,
        sim_reset_fn=None,
        *,
        use_stdin: bool = False,
    ):
        self._lock = threading.Lock()
        self._pending_fsm_target = None
        self._pending_fsm_cause = None
        self._started = False
        self._use_stdin = use_stdin
        self._cmds = None
        self._sim_reset_fn = sim_reset_fn
        self._damp_node_name = damp_node_name
        self._edge = EdgeDetector()

        r = ranges or CommandRanges()
        self._vel_map = {
            'w': np.array([r.lin_vel_x_range[1], 0, 0], dtype=np.float32) * max_abs,
            's': np.array([r.lin_vel_x_range[0], 0, 0], dtype=np.float32) * max_abs,
            'a': np.array([0, r.lin_vel_y_range[1], 0], dtype=np.float32) * max_abs,
            'd': np.array([0, r.lin_vel_y_range[0], 0], dtype=np.float32) * max_abs,
            'q': np.array([0, 0, r.ang_vel_z_range[1]], dtype=np.float32) * max_abs,
            'e': np.array([0, 0, r.ang_vel_z_range[0]], dtype=np.float32) * max_abs,
            ' ': np.zeros(3, dtype=np.float32),
        }
        self._current_vel = np.zeros(3, dtype=np.float32)

    @property
    def damp_node_name(self) -> str:
        return self._damp_node_name

    @property
    def vel_map(self) -> dict:
        return self._vel_map

    # --- lifecycle ---

    def start(self):
        if self._started:
            return
        self._started = True
        if not self._use_stdin:
            return
        try:
            from lib.vendor.util.stdin_cmd import StdinCommands
            self._cmds = StdinCommands()
            self._cmds.add_command('x', lambda: self._push_fsm(self._damp_node_name, cause="Keyboard button X pressed"))
            for i in range(10):
                self._cmds.add_command(str(i), lambda i=i: self._push_fsm(i, cause=f"Keyboard button {i} pressed"))
            for key in self._vel_map:
                self._cmds.add_command(key, lambda k=key: self._on_vel_key(k))
            if self._sim_reset_fn is not None:
                self._cmds.add_command('r', self._sim_reset_fn)
            self._cmds.start()
        except Exception as e:
            print(f"[InputReader] Could not start keyboard (no TTY?): {e}")

    def stop(self):
        if self._cmds is not None:
            self._cmds.stop()
            self._cmds = None
        self._started = False

    # --- programmatic input (used by live view bridge, tests, etc.) ---

    def set_velocity(self, vel: np.ndarray) -> None:
        with self._lock:
            self._current_vel = vel.copy()

    def push_fsm(self, target, cause: Optional[str] = None) -> None:
        with self._lock:
            self._push_fsm_unlocked(target, cause)

    def reset_sim(self) -> None:
        if self._sim_reset_fn is not None:
            self._sim_reset_fn()

    # --- stdin keyboard callbacks ---

    def _push_fsm_unlocked(self, target, cause: Optional[str] = None):
        if self._pending_fsm_target is None or target in (0, self._damp_node_name):
            self._pending_fsm_target = target
            self._pending_fsm_cause = cause
        else:
            print(f"[WARNING] FSM target DROPPED: {target} (pending={self._pending_fsm_target})")

    def _push_fsm(self, target, cause: Optional[str] = None):
        with self._lock:
            self._push_fsm_unlocked(target, cause)

    def _on_vel_key(self, key):
        with self._lock:
            self._current_vel = self._vel_map[key].copy()

    def get_velocity(self) -> np.ndarray:
        with self._lock:
            return self._current_vel.copy()

    def bind_key(self, key: str, callback):
        """Register or override a keyboard callback after start()."""
        if self._cmds is None:
            raise RuntimeError("InputReader stdin keyboard is not initialized")
        self._cmds.add_command(key, callback)

    def disable_key(self, key: str):
        """Disable an existing keyboard binding."""
        self.bind_key(key, lambda: None)

    def bind_velocity_alias(self, key: str, source_key: str, description: Optional[str] = None):
        """Bind a new key to an existing velocity command."""
        self._vel_map[key] = self._vel_map[source_key].copy()

        def _callback():
            self._on_vel_key(key)
            if description is not None:
                print(f"[InputReader] {description}", flush=True)

        self.bind_key(key, _callback)

    # --- VR buttons (called from main loop) ---

    def poll_vr_buttons(self, vr_pose_tracker):
        """Check VR buttons with rising-edge detection. Call once per tick."""
        if vr_pose_tracker is None or not hasattr(vr_pose_tracker, 'get_controller_states'):
            return
        left_ctrl, right_ctrl = vr_pose_tracker.get_controller_states()
        btn_a = bool(right_ctrl and right_ctrl.btn_a)
        btn_b = bool(right_ctrl and right_ctrl.btn_b)
        if self._edge.rising('btn_a', btn_a):
            self._push_fsm(self._damp_node_name, cause="VR button A pressed (right controller)")
        elif self._edge.rising('btn_b', btn_b):
            self._push_fsm(1, cause="VR button B pressed (right controller)")

    # --- consume (called from main loop) ---

    def consume_fsm_target(self) -> PendingFSMTarget:
        with self._lock:
            target = self._pending_fsm_target
            cause = self._pending_fsm_cause
            self._pending_fsm_target = None
            self._pending_fsm_cause = None
            return PendingFSMTarget(target=target, cause=cause)

    # Commander interface — allows CommandedVRPoseTracker to read velocity
    def get_command(self) -> np.ndarray:
        return self.get_velocity()


class ControllerReader:
    """Input reader using a physical game controller (RemoteControllerCommander).

    Button mapping:
        X = DampNode, A = edge 1, B = edge 2, Y = edge 3
        Left joystick  = forward/back + left/right
        Right joystick = rotate
    """

    def __init__(self, damp_node_name: str, remote_controller, ranges=None, max_abs: float = 1.0):
        from lib.deploy.deploy_utils.commander import RemoteControllerCommander
        from lib.deploy.control.remote_controller import KeyMap

        self._lock = threading.Lock()
        self._pending_fsm_target = None

        r = ranges or CommandRanges()

        def push_fsm(target):
            with self._lock:
                if self._pending_fsm_target is None or target in (0, damp_node_name):
                    self._pending_fsm_target = target

        self._commander = RemoteControllerCommander(
            remote_controller,
            r,
            {
                KeyMap.X: lambda: push_fsm(damp_node_name),
                KeyMap.A: lambda: push_fsm(1),
                KeyMap.B: lambda: push_fsm(2),
                KeyMap.Y: lambda: push_fsm(3),
            },
        )

    def start(self):
        pass  # RemoteControllerCommander starts in __init__

    def stop(self):
        self._commander.stop()

    def poll_vr_buttons(self, vr_pose_tracker):
        pass  # No VR buttons with game controller

    def consume_fsm_target(self) -> PendingFSMTarget:
        with self._lock:
            t = self._pending_fsm_target
            self._pending_fsm_target = None
            return PendingFSMTarget(target=t, cause=None)

    def get_velocity(self) -> np.ndarray:
        return self._commander.get_command()

    # Commander interface — allows CommandedVRPoseTracker to read velocity
    def get_command(self) -> np.ndarray:
        return self._commander.get_command()

    def get_vertical_command(self) -> float:
        return self._commander.get_vertical_command()
