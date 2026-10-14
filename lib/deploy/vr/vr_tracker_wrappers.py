"""
Wrapper classes for converting between VRTCPVelTracker and BaseVRPoseTracker.

TCPVelTrackerFromPose: wraps BaseVRPoseTracker, computes velocities via finite differences.
PoseTrackerFromTCPVel: wraps any VRTCPVelTracker and computes VRPoseState by integrating velocities.
"""

import time
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from lib.deploy.vr.vr_pose_tracker import BaseVRPoseTracker, VRPoseState
from lib.deploy.vr.vr_pose_tracker_with_velocity import VRPoseStateWithVelocity
from lib.deploy.vr.vr_tcp_vel_tracker import Velocity, VRTCPVelState, VRTCPVelTracker
from lib.deploy.vr.common import VRControllerState


_ZERO_VEL = Velocity(xyz=np.zeros(3), rotvec=np.zeros(3))


def _compute_velocity(
    prev_pose: Optional[np.ndarray],
    curr_pose: Optional[np.ndarray],
    prev_ts: Optional[float],
    curr_ts: Optional[float],
    dt: float,
) -> Velocity:
    if prev_pose is None or curr_pose is None or prev_ts is None or curr_ts is None:
        return Velocity(xyz=np.zeros(3), rotvec=np.zeros(3))

    if curr_ts <= prev_ts or dt <= 0:
        return Velocity(xyz=np.zeros(3), rotvec=np.zeros(3))

    linear_vel = (curr_pose[:3, 3] - prev_pose[:3, 3]) / dt

    r_prev = R.from_matrix(prev_pose[:3, :3])
    r_curr = R.from_matrix(curr_pose[:3, :3])
    angular_vel_rad = (r_curr * r_prev.inv()).as_rotvec() / dt
    angular_vel_deg = np.degrees(angular_vel_rad)

    return Velocity(xyz=linear_vel, rotvec=angular_vel_deg)


class TCPVelTrackerFromPose(VRTCPVelTracker):
    """Wraps a BaseVRPoseTracker and computes velocities from consecutive poses."""

    def __init__(self, base_tracker: BaseVRPoseTracker):
        self.base_tracker = base_tracker
        self._prev_state: Optional[VRPoseState] = None
        self._last_pose_state: Optional[VRPoseState] = None

    @property
    def last_pose_state(self) -> Optional[VRPoseState]:
        return self._last_pose_state

    def get_state(self) -> VRTCPVelState:
        state = self.base_tracker.get_state()
        self._last_pose_state = state

        if self._prev_state is None:
            self._prev_state = state
            return VRTCPVelState(
                head=Velocity(xyz=np.zeros(3), rotvec=np.zeros(3)),
                left_wrist=Velocity(xyz=np.zeros(3), rotvec=np.zeros(3)),
                right_wrist=Velocity(xyz=np.zeros(3), rotvec=np.zeros(3)),
                dt=0.0,
            )

        # Every component uses the emitted interval, so velocity * dt recovers
        # its displacement even when only a wrist timestamp advances.
        dt = max(
            (curr - prev for prev, curr in (
                (self._prev_state.head_timestamp, state.head_timestamp),
                (self._prev_state.left_wrist_timestamp, state.left_wrist_timestamp),
                (self._prev_state.right_wrist_timestamp, state.right_wrist_timestamp),
            ) if prev is not None and curr is not None),
            default=0.0,
        )
        dt = max(dt, 0.0)
        head_vel = _compute_velocity(
            self._prev_state.head_pose, state.head_pose,
            self._prev_state.head_timestamp, state.head_timestamp, dt,
        )
        left_vel = _compute_velocity(
            self._prev_state.left_wrist_pose, state.left_wrist_pose,
            self._prev_state.left_wrist_timestamp, state.left_wrist_timestamp, dt,
        )
        right_vel = _compute_velocity(
            self._prev_state.right_wrist_pose, state.right_wrist_pose,
            self._prev_state.right_wrist_timestamp, state.right_wrist_timestamp, dt,
        )

        self._prev_state = state
        return VRTCPVelState(head=head_vel, left_wrist=left_vel, right_wrist=right_vel, dt=dt)


class PoseTrackerFromTCPVel(BaseVRPoseTracker):
    """Wraps any VRTCPVelTracker and computes VRPoseState by integrating velocities.

    Call set_current_state() to provide the initial pose before the first get_state().
    If not set, a default T-pose is used (same as CommandedVRPoseTracker.DEFAULT_INITIAL_STATE).

    When cache_between_consumes=True, get_state() is idempotent: it returns the
    same result until consume() is called. This is needed when the VR observator
    chain calls get_state() multiple times per tick.
    """

    DEFAULT_INITIAL_STATE = {
        'head_pose': np.array([
            [-1.0, -0.04, 0.0, 0.0],
            [0.04, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 1.21],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32),
        'left_wrist_pose': np.array([
            [-0.47, -0.70, -0.53, -0.11],
            [0.12, -0.65, 0.75, -0.22],
            [-0.87, 0.29, 0.39, 0.74],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32),
        'right_wrist_pose': np.array([
            [-0.40, 0.56, -0.72, -0.07],
            [0.10, -0.76, -0.64, 0.24],
            [-0.91, -0.33, 0.25, 0.73],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float32),
        'head_timestamp': 0.0,
        'left_wrist_timestamp': 0.0,
        'right_wrist_timestamp': 0.0,
    }

    def __init__(
        self,
        tcp_vel_tracker: VRTCPVelTracker,
        get_time=None,
        cache_between_consumes: bool = False,
        controller_source: Optional[BaseVRPoseTracker] = None,
    ):
        self._tcp_vel_tracker = tcp_vel_tracker
        self._get_time = get_time or time.time
        self._current_state: Optional[VRPoseState] = None
        self._prev_time: Optional[float] = None
        self._caching = cache_between_consumes
        self._cached_state: Optional[VRPoseState] = None
        # When set, ``get_controller_states`` delegates (e.g. to raw VR hardware via this ref).
        self._controller_source = controller_source

    def set_current_state(self, state: VRPoseState) -> None:
        """Set the current pose state used as the base for velocity integration."""
        self._current_state = state

    @property
    def integration_base_state(self) -> Optional[VRPoseState]:
        """Integrated pose before the next velocity step (same as ``_current_state``)."""
        return self._current_state

    def consume(self) -> None:
        """Clear cached state so next get_state() integrates fresh data.

        Call once per tick before get_state() to advance the integration.
        Only meaningful when cache_between_consumes=True.
        """
        self._cached_state = None

    @staticmethod
    def _integrate_pose(pose: Optional[np.ndarray], vel: Velocity, dt: float) -> Optional[np.ndarray]:
        """Integrate velocity over dt to update a 4x4 pose matrix."""
        if pose is None:
            return None
        new_pose = pose.copy()
        new_pose[:3, 3] += vel.xyz * dt
        delta_rotvec_rad = np.radians(vel.rotvec) * dt
        if np.linalg.norm(delta_rotvec_rad) > 1e-12:
            rot_delta = R.from_rotvec(delta_rotvec_rad)
            new_pose[:3, :3] = (rot_delta * R.from_matrix(pose[:3, :3])).as_matrix()
        return new_pose

    def get_state(self) -> VRPoseState:
        if self._caching and self._cached_state is not None:
            return self._cached_state

        vel_state = self._tcp_vel_tracker.get_state()

        if self._current_state is None:
            self._current_state = VRPoseState(**self.DEFAULT_INITIAL_STATE)

        cur_time = self._get_time()
        if vel_state.dt > 0:
            dt = vel_state.dt
        else:
            dt = (cur_time - self._prev_time) if self._prev_time is not None else 0.0
        self._prev_time = cur_time

        self._current_state = VRPoseState(
            head_pose=self._integrate_pose(self._current_state.head_pose, vel_state.head, dt),
            left_wrist_pose=self._integrate_pose(self._current_state.left_wrist_pose, vel_state.left_wrist, dt),
            right_wrist_pose=self._integrate_pose(
                self._current_state.right_wrist_pose, vel_state.right_wrist, dt),
            head_timestamp=cur_time,
            left_wrist_timestamp=cur_time,
            right_wrist_timestamp=cur_time,
        )

        if self._caching:
            self._cached_state = self._current_state

        return self._current_state

    def is_connected(self) -> bool:
        return True

    def get_controller_states(self) -> Tuple[Optional[VRControllerState], Optional[VRControllerState]]:
        if self._controller_source is not None:
            return self._controller_source.get_controller_states()
        return (None, None)


class TriggerZeroingVelTracker(VRTCPVelTracker):
    """Zeroes wrist velocities when controller triggers are not pressed.

    Designed to sit between TCPVelTrackerFromPose (raw VR velocities) and
    PoseTrackerFromTCPVel so that trigger-based zeroing happens *before*
    travel-mode offsets are applied.

    When a trigger is released:
      - first_init=True  → wrist velocity zeroed completely
      - first_init=False → wrist velocity copies head (idle follows head)
    """

    def __init__(self, inner: VRTCPVelTracker, controller_source):
        self._inner = inner
        self._controller_source = controller_source
        self.first_init = True

    def set_controller_source(self, controller_source) -> None:
        """Re-point trigger polling (e.g. to ``PoseTrackerFromTCPVel`` that delegates to hardware)."""
        self._controller_source = controller_source

    def get_state(self) -> VRTCPVelState:
        state = self._inner.get_state()
        left_ctrl, right_ctrl = self._controller_source.get_controller_states()

        right_teleop = right_ctrl is not None and right_ctrl.trigger_front > 0.0
        left_teleop = left_ctrl is not None and left_ctrl.trigger_front > 0.0

        if not right_teleop:
            if self.first_init:
                state.right_wrist.xyz = np.zeros(3)
                state.right_wrist.rotvec = np.zeros(3)
            else:
                state.right_wrist.xyz = state.head.xyz.copy()
                state.right_wrist.rotvec = state.head.rotvec.copy()

        if not left_teleop:
            if self.first_init:
                state.left_wrist.xyz = np.zeros(3)
                state.left_wrist.rotvec = np.zeros(3)
            else:
                state.left_wrist.xyz = state.head.xyz.copy()
                state.left_wrist.rotvec = state.head.rotvec.copy()

        return state


class _VelStateSpy(VRTCPVelTracker):
    """Transparent wrapper that records the last returned VRTCPVelState."""

    def __init__(self, inner: VRTCPVelTracker):
        self._inner = inner
        self.last_state: Optional[VRTCPVelState] = None

    def get_state(self) -> VRTCPVelState:
        self.last_state = self._inner.get_state()
        return self.last_state


class VelPoseTrackerFromTCPVel(PoseTrackerFromTCPVel):
    """Like PoseTrackerFromTCPVel but returns VRPoseStateWithVelocity.

    Reads real velocity (m/s, deg/s) directly from the VRTCPVelState,
    avoiding the need for VrVelPoseTracker to re-estimate velocity from
    positions.

    Position preservation guarantee: ALL pose integration is delegated to
    ``super().get_state()`` — this class never touches integration logic.

    When vel_state.dt == 0 (freeze tick), the last computed velocity is
    preserved to avoid sudden drops to zero that cause jerky motion.
    """

    def __init__(self, tcp_vel_tracker: VRTCPVelTracker, **kwargs):
        self._spy = _VelStateSpy(tcp_vel_tracker)
        super().__init__(self._spy, **kwargs)
        # Last computed velocities — preserved on freeze ticks (dt=0)
        self._last_head_lin_vel = np.zeros(3)
        self._last_head_ang_vel = np.zeros(3)
        self._last_left_lin_vel = np.zeros(3)
        self._last_left_ang_vel = np.zeros(3)
        self._last_right_lin_vel = np.zeros(3)
        self._last_right_ang_vel = np.zeros(3)
        # Own cache for VRPoseStateWithVelocity (parent caches VRPoseState)
        self._cached_vel_result: Optional[VRPoseStateWithVelocity] = None

    @staticmethod
    def _extract_velocity(vel: Velocity):
        """Extract velocity as (m/s, rad/s) from a Velocity dataclass."""
        linear_vel = vel.xyz
        angular_vel = np.radians(vel.rotvec)
        return linear_vel, angular_vel

    def consume(self) -> None:
        super().consume()
        self._cached_vel_result = None

    def get_state(self) -> VRPoseStateWithVelocity:
        if self._caching and self._cached_vel_result is not None:
            return self._cached_vel_result

        # Parent handles ALL pose integration — positions are identical by construction
        pose_state = super().get_state()

        # Compute velocity from captured vel_state; preserve last on freeze ticks
        vel_state = self._spy.last_state
        if vel_state is not None and vel_state.dt > 0:
            self._last_head_lin_vel, self._last_head_ang_vel = (
                self._extract_velocity(vel_state.head))
            self._last_left_lin_vel, self._last_left_ang_vel = (
                self._extract_velocity(vel_state.left_wrist))
            self._last_right_lin_vel, self._last_right_ang_vel = (
                self._extract_velocity(vel_state.right_wrist))

        result = VRPoseStateWithVelocity(
            head_pose=pose_state.head_pose,
            left_wrist_pose=pose_state.left_wrist_pose,
            right_wrist_pose=pose_state.right_wrist_pose,
            head_timestamp=pose_state.head_timestamp,
            left_wrist_timestamp=pose_state.left_wrist_timestamp,
            right_wrist_timestamp=pose_state.right_wrist_timestamp,
            head_linear_velocity=self._last_head_lin_vel.copy(),
            head_angular_velocity=self._last_head_ang_vel.copy(),
            left_wrist_linear_velocity=self._last_left_lin_vel.copy(),
            left_wrist_angular_velocity=self._last_left_ang_vel.copy(),
            right_wrist_linear_velocity=self._last_right_lin_vel.copy(),
            right_wrist_angular_velocity=self._last_right_ang_vel.copy(),
        )

        if self._caching:
            self._cached_vel_result = result

        return result

    @property
    def base_tracker(self):
        """Duck-type compatibility with VrVelPoseTracker.

        PolicyNode inspects tracker.base_tracker to detect VR replay; returns self.
        """
        return self

    def reset_velocity_estimators(self):
        """No-op. Required for duck-type compatibility with VrVelPoseTracker."""
        pass
