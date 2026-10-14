"""
Commanded VR Pose Tracker.

Applies velocity-based offsets to an initial static pose based on commander input.
Supports both KeyboardCommander and RemoteControllerCommander.
"""

import time
import typing as tp
import numpy as np
from scipy.spatial.transform import Rotation as R

from lib.deploy.vr.vr_pose_tracker import VRPoseState
from lib.deploy.vr.vr_observations import pose_matrix_to_pos_quat
from lib.deploy.vr.util import quat_apply, heading_frame_from_head, quat_mul
from lib.deploy.deploy_utils.commander import Commander


class CommandedVRPoseTracker:
    """
    VR pose tracker that applies velocity-commanded offsets to an initial static pose.

    Takes an initial VRPoseState and applies position/rotation offsets based on
    velocity commands from a commander (keyboard or remote controller).

    The commander provides velocity commands as [vx, vy, angular_z]:
    - vx: Forward/backward velocity (positive = forward)
    - vy: Left/right velocity (positive = left)
    - angular_z: Yaw rotation velocity (positive = rotate left)

    These are integrated over time to accumulate position and rotation offsets.
    """

    # Default T-pose initial state
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
        'right_wrist_timestamp': 0.0
    }

    def __init__(
        self,
        initial_state: tp.Optional[VRPoseState] = None,
        heading_frame: str = 'pelvis',
        env: tp.Optional[tp.Any] = None,  # BaseEnv, but avoid circular import
        control_dt: float = 0.02,
        smoothing_alpha: float = .03,
    ):
        """
        Initialize the commanded tracker.

        Args:
            initial_state: Initial VR pose state to use as base. If None, uses DEFAULT_INITIAL_STATE.
            heading_frame: Heading frame source (e.g., 'pelvis' or 'head')
            env: Optional environment for time tracking (must have get_env_time() method)
            control_dt: Default control timestep for first iteration
            smoothing_alpha: Exponential smoothing factor (0.0 to 1.0).
                - 1.0 = no smoothing (instant response, default)
                - 0.0 = infinite smoothing (never changes)
                - Typical values: 0.05-0.3 for moderate smoothing
        """
        if initial_state is None:
            initial_state = VRPoseState(**self.DEFAULT_INITIAL_STATE)

        self._initial_state = initial_state
        self._heading_frame = heading_frame
        self._env = env
        self._control_dt = control_dt
        self._smoothing_alpha = np.clip(smoothing_alpha, 0.0, 1.0)

        # Commander (KeyboardCommander or RemoteControllerCommander)
        self._commander: tp.Optional[Commander] = None

        # Accumulated offsets in heading frame (target values before smoothing)
        self._offset_h = np.zeros(3, dtype=np.float32)
        self._yaw_offset = 0.0

        # Smoothed offsets (what actually gets applied)
        self._smoothed_offset_h = np.zeros(3, dtype=np.float32)
        self._smoothed_yaw_offset = 0.0

        # Time tracking
        self._prev_step_time: tp.Optional[float] = None

    def set_commander(self, commander: Commander) -> None:
        """Set the commander (KeyboardCommander or RemoteControllerCommander)."""
        self._commander = commander

    def set_env(self, env: tp.Any) -> None:
        """Bind the env so integration uses its clock (sim time in sim, wall clock
        on the real robot) instead of wall clock. Resets timing so the next step
        starts from a clean ``control_dt`` rather than a wall-vs-sim time jump."""
        self._env = env
        self._prev_step_time = None

    def _get_current_time(self) -> float:
        """Get current time from env or wall clock."""
        if self._env is not None and hasattr(self._env, 'get_env_time'):
            return self._env.get_env_time()
        return time.time()

    def _rotate_by_yaw(self, vec: np.ndarray) -> np.ndarray:
        """Rotate a 3D vector by the current yaw offset around Z axis."""
        cos_yaw = np.cos(self._yaw_offset)
        sin_yaw = np.sin(self._yaw_offset)
        return np.array([
            vec[0] * cos_yaw - vec[1] * sin_yaw,
            vec[0] * sin_yaw + vec[1] * cos_yaw,
            vec[2]
        ], dtype=np.float32)

    def _update_offsets(self, dt: float) -> None:
        """Update position and rotation offsets based on commander velocity."""
        if self._commander is None:
            return

        cmd = self._commander.get_command()
        if cmd is None:
            return

        # cmd = [vx, vy, angular_z]
        vx, vy, ang_z = float(cmd[0]), float(cmd[1]), float(cmd[2])

        # Update horizontal position offset (in heading frame, rotated by accumulated yaw)
        if abs(vx) > 1e-6 or abs(vy) > 1e-6:
            local_velocity = np.array([vx, vy, 0.0], dtype=np.float32)
            # Rotate velocity by accumulated yaw so controls are relative to robot's current heading
            rotated_velocity = self._rotate_by_yaw(local_velocity)
            self._offset_h += rotated_velocity * dt

        # Update vertical position offset (up/down)
        if hasattr(self._commander, 'get_vertical_command'):
            vz = self._commander.get_vertical_command()
            if abs(vz) > 1e-6:
                self._offset_h[2] += vz * dt
                self._offset_h[2] = np.clip(self._offset_h[2], a_min=-.3, a_max=0.)

        # Update yaw offset
        if abs(ang_z) > 1e-6:
            self._yaw_offset += ang_z * dt

    def _apply_smoothing(self) -> None:
        """Apply exponential smoothing to offsets.

        Smoothed = alpha * target + (1 - alpha) * smoothed_previous
        """
        alpha = self._smoothing_alpha
        self._smoothed_offset_h = alpha * self._offset_h + (1 - alpha) * self._smoothed_offset_h
        self._smoothed_yaw_offset = alpha * self._yaw_offset + (1 - alpha) * self._smoothed_yaw_offset

    def _apply_offsets_to_pose(
        self,
        pose: tp.Optional[np.ndarray],
        heading_pos_w: np.ndarray,
        heading_quat_w: np.ndarray,
        heading_quat_inv: np.ndarray,
        yaw_quat: np.ndarray,
    ) -> tp.Optional[np.ndarray]:
        """Apply accumulated offsets to a single pose matrix."""
        if pose is None:
            return None

        pos_w, quat_w = pose_matrix_to_pos_quat(pose)

        # Transform to heading frame
        rel_pos_h = quat_apply(heading_quat_inv, pos_w - heading_pos_w)
        rel_quat_h = quat_mul(heading_quat_inv, quat_w)

        # Apply smoothed offsets
        rel_pos_h = quat_apply(yaw_quat, rel_pos_h) + self._smoothed_offset_h
        rel_quat_h = quat_mul(yaw_quat, rel_quat_h)

        # Transform back to world frame
        pos_w_new = heading_pos_w + quat_apply(heading_quat_w, rel_pos_h)
        quat_w_new = quat_mul(heading_quat_w, rel_quat_h)

        # Build output pose matrix
        out = np.eye(4, dtype=np.float32)
        out[:3, :3] = R.from_quat(quat_w_new, scalar_first=True).as_matrix()
        out[:3, 3] = pos_w_new
        return out

    def get_state(self) -> VRPoseState:
        """Get current commanded VR pose state."""
        # Handle uninitialized state
        if self._initial_state.head_pose is None:
            return self._initial_state

        # Get current time and compute dt
        cur_time = self._get_current_time()
        dt = cur_time - self._prev_step_time if self._prev_step_time is not None else self._control_dt
        self._prev_step_time = cur_time

        # Update offsets based on commander input
        self._update_offsets(dt)

        # Apply exponential smoothing to reduce acceleration
        self._apply_smoothing()

        # Compute heading frame from initial head pose
        head_pos, head_quat = pose_matrix_to_pos_quat(self._initial_state.head_pose)
        _, heading_quat = heading_frame_from_head(head_pos, head_quat, heading_frame=self._heading_frame)
        heading_pos_w = head_pos
        heading_quat_w = heading_quat

        # Compute yaw quaternion and heading inverse (using smoothed yaw)
        yaw_half = 0.5 * self._smoothed_yaw_offset
        yaw_quat = np.array([np.cos(yaw_half), 0.0, 0.0, np.sin(yaw_half)], dtype=np.float32)
        heading_quat_inv = heading_quat_w.copy()
        heading_quat_inv[1:] *= -1.0

        # Apply offsets to all poses
        return VRPoseState(
            head_pose=self._apply_offsets_to_pose(
                self._initial_state.head_pose, heading_pos_w, heading_quat_w, heading_quat_inv, yaw_quat
            ),
            left_wrist_pose=self._apply_offsets_to_pose(
                self._initial_state.left_wrist_pose, heading_pos_w, heading_quat_w, heading_quat_inv, yaw_quat
            ),
            right_wrist_pose=self._apply_offsets_to_pose(
                self._initial_state.right_wrist_pose, heading_pos_w, heading_quat_w, heading_quat_inv, yaw_quat
            ),
            head_timestamp=cur_time,
            left_wrist_timestamp=cur_time,
            right_wrist_timestamp=cur_time,
        )

    def is_connected(self) -> bool:
        """Always return True since we're using a static initial state."""
        return True

    def get_controller_states(self) -> tp.Tuple[tp.Any, tp.Any]:
        """Return no controller states since commanded tracking has no VR buttons."""
        return (None, None)

    def reset_offsets(self) -> None:
        """Reset accumulated offsets to zero."""
        self._offset_h = np.zeros(3, dtype=np.float32)
        self._yaw_offset = 0.0
        self._smoothed_offset_h = np.zeros(3, dtype=np.float32)
        self._smoothed_yaw_offset = 0.0

    def set_smoothing_alpha(self, alpha: float) -> None:
        """Set the smoothing factor.

        Args:
            alpha: Smoothing factor (0.0 to 1.0).
                - 1.0 = no smoothing (instant response)
                - 0.0 = infinite smoothing (never changes)
                - Typical values: 0.05-0.3 for moderate smoothing
        """
        self._smoothing_alpha = np.clip(alpha, 0.0, 1.0)
