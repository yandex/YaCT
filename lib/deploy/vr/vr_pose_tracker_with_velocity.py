"""
VR Pose Tracker with Velocity Estimation for teleoperation.

This module provides VrVelPoseTracker, which wraps a BaseVRPoseTracker
and adds velocity tracking capabilities using RealTimeVelocityEstimator.
"""

from collections import deque
from typing import Deque, List, Optional, Tuple

import numpy as np
from lib.deploy.vr.vel_estimation import RealTimeVelocityEstimator
from lib.deploy.vr.vr_pose_tracker import VRPoseState as BaseVRPoseState
from lib.deploy.vr.vr_pose_tracker import BaseVRPoseTracker
from scipy.spatial.transform import Rotation as R
from pydantic import field_serializer


class VRPoseStateWithVelocity(BaseVRPoseState):
    """Extended VRPoseState that includes velocity information.

    Inherits all pose data from BaseVRPoseState and adds linear and angular
    velocity estimates for head and both wrists in the world frame.
    """
    _array_fields = ['head_linear_velocity', 'head_angular_velocity', 'left_wrist_linear_velocity', 'left_wrist_angular_velocity',
                     'right_wrist_linear_velocity', 'right_wrist_angular_velocity']

    # Head velocities (3D vectors in world frame)
    head_linear_velocity: Optional[np.ndarray] = None
    head_angular_velocity: Optional[np.ndarray] = None

    # Left wrist velocities (3D vectors in world frame)
    left_wrist_linear_velocity: Optional[np.ndarray] = None
    left_wrist_angular_velocity: Optional[np.ndarray] = None

    # Right wrist velocities (3D vectors in world frame)
    right_wrist_linear_velocity: Optional[np.ndarray] = None
    right_wrist_angular_velocity: Optional[np.ndarray] = None

    @field_serializer(*_array_fields)
    def serialize_array(self, v: Optional[np.ndarray], _info) -> Optional[List[float]]:
        """
        Converts 1D numpy velocity vectors to flat lists.
        """
        if v is None:
            return None
        return v.tolist()


class RobustPolynomialVelocityEstimator:
    """
    Buffers N frames and fits a 2nd-degree polynomial to estimate velocity
    at the center of the buffer.
    """

    def __init__(
        self,
        buffer_size: int = 5,
        min_time_delta: float = 1e-5,
        max_time_delta: float = 0.1
    ):
        """
        Args:
            buffer_size: Number of frames to buffer (must be odd).
            min_time_delta: Minimum time allowed between frames.
                            Updates closer than this are discarded to prevent
                            Singular Matrix errors in the solver.
            max_time_delta: Maximum time allowed between frames.
                            If exceeded, buffer is cleared as old data is stale.
        """
        if buffer_size < 3 or buffer_size % 2 == 0:
            raise ValueError("Buffer size must be an odd number >= 3")

        self.buffer_size = buffer_size
        self.mid_idx = buffer_size // 2
        self.min_time_delta = min_time_delta
        self.max_time_delta = max_time_delta

        # Buffer stores: (pose_matrix, timestamp)
        self.buffer: Deque[Tuple[np.ndarray, float]] = deque(maxlen=buffer_size)

        # Cache the last valid result to return if we skip a frame or fail
        self.last_valid_result = None
        self.zero_vel = np.zeros(3)

    def reset(self):
        self.buffer.clear()
        self.last_valid_result = None

    def update(self, current_pose: np.ndarray, current_timestamp: float):
        """
        Push state to buffer (if timestamp is unique) and calculate velocity.
        Returns: (mid_pose, mid_timestamp, lin_vel, ang_vel)
        """

        # --- 1. CHECK TIME DELTAS ---
        if len(self.buffer) > 0:
            last_ts = self.buffer[-1][1]
            dt = current_timestamp - last_ts

            # --- 1a. FILTER STALE DATA ---
            # If too much time has passed, old data is stale - clear buffer and restart
            if dt > self.max_time_delta:
                self.buffer.clear()
                self.last_valid_result = None
                # Fall through to add current frame as first entry

            # --- 1b. FILTER DUPLICATE TIMESTAMPS ---
            # If the update is too close to the previous one (or identical),
            # discard it to prevent Singular Matrix (SVD) errors.
            elif dt < self.min_time_delta:
                # If we have a valid previous calculation, return it.
                # Otherwise treat as warmup.
                if self.last_valid_result is not None:
                    return self.last_valid_result
                else:
                    return current_pose, current_timestamp, self.zero_vel, self.zero_vel

        # --- 2. UPDATE BUFFER ---
        self.buffer.append((current_pose, current_timestamp))

        # --- 3. CHECK WARMUP ---
        if len(self.buffer) < self.buffer_size:
            # Not enough data yet. Return current pose with 0 velocity.
            return current_pose, current_timestamp, self.zero_vel.copy(), self.zero_vel.copy()

        # --- 4. PREPARE CALCULATION ---

        # Get the center frame (this is what we return velocity for)
        pose_mid, t_mid = self.buffer[self.mid_idx]

        try:
            # Extract time relative to the CENTER frame.
            # This is crucial. t_rel will look like [-0.02, -0.01, 0.0, 0.012, 0.022]
            # Using relative time prevents floating point precision errors
            # and simplifies the derivative math (t=0).
            timestamps = np.array([item[1] for item in self.buffer])
            t_rel = timestamps - t_mid

            # --- A. LINEAR VELOCITY ---
            # Extract positions (N, 3)
            positions = np.array([item[0][:3, 3] for item in self.buffer])

            # Fit Quadratic: p(t) = at^2 + bt + c
            # We fit all 3 dimensions (x,y,z) at once.
            # polyfit returns coeffs as rows: [a, b, c]
            # deg=2 implies constant acceleration over the window.
            lin_coeffs = np.polyfit(t_rel, positions, deg=2)

            # The derivative is v(t) = 2at + b.
            # Since we centered t_rel such that the frame of interest is t=0,
            # Velocity = b (the coefficient of the term to power 1).
            # polyfit returns highest power first -> index -2 is 'b'
            linear_vel = lin_coeffs[-2]

            # --- B. ANGULAR VELOCITY ---
            # 1. Get Mid Rotation Inverse
            rot_mid_inv = R.from_matrix(pose_mid[:3, :3]).inv()

            # 2. Calculate Rotation Vectors relative to Mid
            # (This linearizes the rotation around the operation point)
            rot_vecs = []
            for item in self.buffer:
                r_curr = R.from_matrix(item[0][:3, :3])
                # Diff = Current * Mid_Inverse
                r_rel = r_curr * rot_mid_inv
                rot_vecs.append(r_rel.as_rotvec())

            rot_vecs = np.array(rot_vecs) # Shape (N, 3)

            # 3. Fit Quadratic to rotation vectors
            ang_coeffs = np.polyfit(t_rel, rot_vecs, deg=2)

            # 4. Derivative at t=0 is 'b'
            # This represents angular velocity in the local frame of 'mid',
            # but because our reference was Global, this result is in Global frame.
            angular_vel = ang_coeffs[-2]

            # Cache success
            self.last_valid_result = (pose_mid, t_mid, linear_vel, angular_vel)
            return self.last_valid_result

        except np.linalg.LinAlgError:
            # SVD Failure fallback (should be rare with the timestamp filter above)
            # Fallback to simple linear difference between start and end of buffer
            pose_start, t_start = self.buffer[0]
            pose_end, t_end = self.buffer[-1]
            total_dt = t_end - t_start

            if total_dt < 1e-5:
                 return pose_mid, t_mid, self.zero_vel.copy(), self.zero_vel.copy()

            # Linear fallback
            lin_v_fallback = (pose_end[:3, 3] - pose_start[:3, 3]) / total_dt

            # Angular fallback
            r_end = R.from_matrix(pose_end[:3, :3])
            r_start = R.from_matrix(pose_start[:3, :3])
            ang_v_fallback = (r_end * r_start.inv()).as_rotvec() / total_dt

            self.last_valid_result = (pose_mid, t_mid, lin_v_fallback, ang_v_fallback)
            return self.last_valid_result


class VrVelPoseTracker:
    def __init__(
        self,
        base_tracker: BaseVRPoseTracker,
        buffer_size: int = 5,
        max_time_delta: float = 0.5
    ):
        """
        Args:
            base_tracker: Hardware wrapper
            buffer_size: Frames to smooth. Must be Odd. 5 is a good balance.
            max_time_delta: Maximum allowed time between frames before clearing buffer.
        """
        self.base_tracker = base_tracker

        # Enforce odd buffer size
        if buffer_size % 2 == 0:
            buffer_size += 1

        self.head_est = RobustPolynomialVelocityEstimator(buffer_size, max_time_delta=max_time_delta)
        self.left_est = RobustPolynomialVelocityEstimator(buffer_size, max_time_delta=max_time_delta)
        self.right_est = RobustPolynomialVelocityEstimator(buffer_size, max_time_delta=max_time_delta)

    def get_state(self) -> VRPoseStateWithVelocity:
        # 1. Get Raw Data
        raw = self.base_tracker.get_state()

        # 2. Defaults
        h_out = (None, None, None, None)
        l_out = (None, None, None, None)
        r_out = (None, None, None, None)

        # 3. Update Estimators
        if raw.head_pose is not None and raw.head_timestamp is not None:
            h_out = self.head_est.update(raw.head_pose, raw.head_timestamp)

        if raw.left_wrist_pose is not None and raw.left_wrist_timestamp is not None:
            l_out = self.left_est.update(raw.left_wrist_pose, raw.left_wrist_timestamp)

        if raw.right_wrist_pose is not None and raw.right_wrist_timestamp is not None:
            r_out = self.right_est.update(raw.right_wrist_pose, raw.right_wrist_timestamp)

        # 4. Return State (Using Delayed/Smoothed Poses)
        return VRPoseStateWithVelocity(
            # HEAD
            head_pose=h_out[0],
            head_timestamp=h_out[1],
            head_linear_velocity=h_out[2],
            head_angular_velocity=h_out[3],

            # LEFT
            left_wrist_pose=l_out[0],
            left_wrist_timestamp=l_out[1],
            left_wrist_linear_velocity=l_out[2],
            left_wrist_angular_velocity=l_out[3],

            # RIGHT
            right_wrist_pose=r_out[0],
            right_wrist_timestamp=r_out[1],
            right_wrist_linear_velocity=r_out[2],
            right_wrist_angular_velocity=r_out[3]
        )

    def is_connected(self) -> bool:
        return self.base_tracker.is_connected()

    def get_controller_states(self):
        return self.base_tracker.get_controller_states()

    def reset_velocity_estimators(self):
        self.head_est.reset()
        self.left_est.reset()
        self.right_est.reset()

