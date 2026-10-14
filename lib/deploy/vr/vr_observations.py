import json
from pydantic import BaseModel, Field, field_validator, field_serializer, ConfigDict
from typing import Annotated, Optional, Tuple, Union, Literal
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation, Slerp
from tqdm import tqdm
from lib.deploy.vr.common import Pose
import lib.vendor.util.math as math_utils
from copy import deepcopy
from scipy.spatial.transform import Rotation as R


class WristPoses(BaseModel):
    left_wrist_pose: Pose
    right_wrist_pose: Pose


class HeadWristsPose(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    pos: Annotated[NDArray[np.float32], Field(description="3x3 positions of head, left wrist and right wrist")]
    quat: Annotated[NDArray[np.float32], Field(description="3x4 quaternions (w,x,y,z) of head, left wrist and right wrist")]

    @field_validator('pos', mode='before')
    @classmethod
    def validate_and_convert_pos(cls, v):
        arr = np.asarray(v, dtype=np.float32)
        if arr.shape != (3, 3):
            raise ValueError(f'Position must be 3x3, got {arr.shape}')
        return arr

    @field_validator('quat', mode='before')
    @classmethod
    def validate_and_convert_quat(cls, v):
        arr = np.asarray(v, dtype=np.float32)
        if arr.shape != (3, 4):
            raise ValueError(f'Quaternion must be 3x4, got {arr.shape}')
        return arr

    @field_serializer('pos', 'quat')
    def serialize_array(self, value: NDArray[np.float32]) -> list:
        return value.tolist()


class HeadWristsVelocity(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    lin_vel: Annotated[
        NDArray[np.float32],
        Field(description="3x3 matrix of linear velocities")
    ]
    ang_vel: Annotated[
        Optional[NDArray[np.float32]],
        Field(None, description="3x3 matrix of angular velocities")
    ]

    @field_validator('lin_vel', 'ang_vel', mode='before')
    @classmethod
    def validate_and_convert(cls, v) -> Optional[NDArray[np.float32]]:
        if v is None:
            return None
        arr = np.asarray(v, dtype=np.float32)
        if arr.shape != (3, 3):
            raise ValueError(f'Matrix must be 3x3, got {arr.shape}')
        return arr

    @field_serializer('lin_vel', 'ang_vel')
    def serialize_array(self, value: Optional[NDArray[np.float32]]) -> Optional[list]:
        if value is None:
            return None
        return value.tolist()

    @staticmethod
    def zero() -> 'HeadWristsVelocity':
        """Create a zero velocity observation."""
        return HeadWristsVelocity(
            lin_vel=np.zeros((3, 3), dtype=np.float32),
            ang_vel=None
        )


class VrObservations(BaseModel):
    head_wrists_pose_vrw: HeadWristsPose
    head_wrists_vel_vrw: HeadWristsVelocity


class VrHeadingObservations(BaseModel):
    head_wrists_pose_w: HeadWristsPose
    head_wrists_vel_w: HeadWristsVelocity
    head_wrists_pose_h: HeadWristsPose
    head_wrists_vel_h: HeadWristsVelocity
    heading_pose: Pose
    head_wrists_pose_vrw: Optional[HeadWristsPose] = None
    head_wrists_vel_vrw: Optional[HeadWristsVelocity] = None

    def with_zero_velocities(self) -> 'VrHeadingObservations':
        """Return a copy of this observation with all velocities set to zero."""
        return VrHeadingObservations(
            head_wrists_pose_w=self.head_wrists_pose_w,
            head_wrists_vel_w=HeadWristsVelocity(
                lin_vel=np.zeros((3, 3), dtype=np.float32),
                ang_vel=np.zeros((3, 3), dtype=np.float32) if self.head_wrists_vel_w.ang_vel is not None else None
            ),
            head_wrists_pose_h=self.head_wrists_pose_h,
            head_wrists_vel_h=HeadWristsVelocity(
                lin_vel=np.zeros((3, 3), dtype=np.float32),
                ang_vel=np.zeros((3, 3), dtype=np.float32) if self.head_wrists_vel_h.ang_vel is not None else None
            ),
            heading_pose=self.heading_pose,
            head_wrists_pose_vrw=self.head_wrists_pose_vrw,
            head_wrists_vel_vrw=HeadWristsVelocity(
                lin_vel=np.zeros((3, 3), dtype=np.float32),
                ang_vel=np.zeros((3, 3), dtype=np.float32) if self.head_wrists_vel_vrw.ang_vel is not None else None
            ) if self.head_wrists_vel_vrw is not None else None,
        )


class VrObservator:
    def get_observation(self) -> VrObservations:
        raise NotImplementedError

    @property
    def tracker(self) -> "VrVelPoseTracker":
        raise NotImplementedError


class VrHeadingObservator(VrObservator):
    def get_observation(self) -> VrHeadingObservations:
        raise NotImplementedError
    def reset_head_offset(self):
        raise NotImplementedError
    def start_pause(self):
        raise NotImplementedError
    def stop_pause(self):
        raise NotImplementedError


def create_vr_observator(vel_tracker: "VrVelPoseTracker") -> VrObservator:
    if vel_tracker is None:
        raise ValueError("vel_tracker is required")
    return VrObservatorImpl(vel_tracker=vel_tracker)


def create_vr_heading_observator(
        observator=None,
        localizator=None,
        use_self_heading_frame=True,
        heading_frame='head',
        fixed_head_z: Optional[float] = None,
        max_head_z: Optional[float] = None,
        fixed_head_quat_h: Optional[np.ndarray] = None,
        fixed_wrist_poses_h: Optional[WristPoses] = None,
        fix_head: bool = False,
        align_z_to_vr: bool = False,
    ) -> VrHeadingObservator:
    return VrHeadingObservatorImpl(
        observator=observator,
        localizator=localizator,
        use_self_heading_frame=use_self_heading_frame,
        heading_frame=heading_frame,
        fixed_head_z=fixed_head_z,
        max_head_z=max_head_z,
        fixed_head_quat_h=fixed_head_quat_h,
        fixed_wrist_poses_h=fixed_wrist_poses_h,
        fix_head=fix_head,
        align_z_to_vr=align_z_to_vr,
    )


class SnapshotVrHeadingObservator(VrHeadingObservator):
    """Wraps a VrHeadingObservator and freezes its observation on snapshot().

    Used as the source of an InterpolatingVrHeadingObservator for the
    DoTeleop → ReturnMiddle transition: call snapshot() before the wrapped
    observator's state is overwritten by the initial-pose callback, so the
    interpolation always starts from the actual DoTeleop pose.
    """

    def __init__(self, wrapped: VrHeadingObservator):
        self._wrapped = wrapped
        self._frozen: Optional[VrHeadingObservations] = None

    def snapshot(self):
        """Freeze the current observation of the wrapped observator."""
        self._frozen = self._wrapped.get_observation()

    def get_observation(self) -> VrHeadingObservations:
        if self._frozen is not None:
            return self._frozen
        return self._wrapped.get_observation()

    def reset_head_offset(self):
        pass  # frozen observation must not change

    def start_pause(self):
        pass

    def stop_pause(self):
        pass

    @property
    def tracker(self):
        return self._wrapped.tracker


def create_interpolating_vr_observator(
    source_observator: VrHeadingObservator,
    target_observator: VrHeadingObservator,
    n_steps: int = 150,
    easing: Literal['linear', 'smoothstep', 'cubic'] = 'smoothstep',
    show_progress: bool = True,
    progress_desc: str = "Interpolating VR"
) -> 'InterpolatingVrHeadingObservator':
    """
    Create an interpolating observator that smoothly transitions between two observators.

    Args:
        source_observator: The starting observator (current state)
        target_observator: The target observator to transition to
        n_steps: Number of steps for the interpolation (default 150)
        easing: Easing function type ('linear', 'smoothstep', 'cubic')
        show_progress: Whether to show tqdm progress bar (default True)
        progress_desc: Description for the progress bar

    Returns:
        InterpolatingVrHeadingObservator instance
    """
    return InterpolatingVrHeadingObservator(
        source_observator=source_observator,
        target_observator=target_observator,
        n_steps=n_steps,
        easing=easing,
        show_progress=show_progress,
        progress_desc=progress_desc
    )


# ----------------------------------------------------------------------------
from lib.deploy.vr.vr_pose_tracker_with_velocity import VrVelPoseTracker, VRPoseStateWithVelocity
from lib.deploy.vr.localization import create_localization_observator, LocalizationObs, LocalizationObservator
from lib.deploy.vr.util import heading_frame_from_head, apply_transform_from_new_base, frame2_in_frame1, transform_velocity_from_new_base, quat_mul
from lib.deploy.vr.math_utils import pose_matrix_to_pos_quat


class VrObservatorImpl(VrObservator):
    def __init__(self, vel_tracker: VrVelPoseTracker):
        self._tracker = vel_tracker
        self._state = None

    @property
    def tracker(self) -> VrVelPoseTracker:
        return self._tracker

    def get_observation(self) -> VrObservations:
        state = self.tracker.get_state()

        self._state = state

        # Convert 4x4 pose matrices to positions and quaternions
        head_pos, head_quat = pose_matrix_to_pos_quat(state.head_pose)
        left_wrist_pos, left_wrist_quat = pose_matrix_to_pos_quat(state.left_wrist_pose)
        right_wrist_pos, right_wrist_quat = pose_matrix_to_pos_quat(state.right_wrist_pose)

        # Stack positions (3, 3)
        head_wrists_pos_vrw = np.stack([
            head_pos,
            left_wrist_pos,
            right_wrist_pos
        ], axis=0)

        # Stack quaternions (3, 4)
        head_wrists_quat_vrw = np.stack([
            head_quat,
            left_wrist_quat,
            right_wrist_quat
        ], axis=0)

        # Stack linear velocities (3, 3)
        head_wrists_lin_vel_vrw = np.stack([
            state.head_linear_velocity,
            state.left_wrist_linear_velocity,
            state.right_wrist_linear_velocity,
        ], axis=0)

        return VrObservations(
            head_wrists_pose_vrw=HeadWristsPose(
                pos=head_wrists_pos_vrw,
                quat=head_wrists_quat_vrw
            ),
            head_wrists_vel_vrw=HeadWristsVelocity(lin_vel=head_wrists_lin_vel_vrw),
        )

    @property
    def state(self) -> Union[None, VRPoseStateWithVelocity]:
        return self._state

import os
AVH = os.environ.get("AVH", "0") == "1"

class VrHeadingObservatorImpl(VrHeadingObservator):
    def __init__(
            self,
            observator: Optional[VrObservator] = None,
            localizator: Optional[LocalizationObservator] = None,
            use_self_heading_frame: bool = True,
            angled_vr_heading: bool = AVH,
            heading_frame: str = 'head',
            fixed_head_z: Optional[float] = None,
            max_head_z: Optional[float] = None,
            fixed_head_quat_h: Optional[np.ndarray] = None,
            fixed_wrist_poses_h: Optional[WristPoses] = None,
            fix_head: bool = False,
            align_z_to_vr: bool = False,
        ):

        if observator is None:
            raise ValueError("observator is required")
        self.observator = observator

        if localizator is None:
            localizator = create_localization_observator()
        self.localizator = localizator

        self.use_self_heading_frame = use_self_heading_frame
        self.angled_vr_heading = angled_vr_heading
        self.head_quat_on_reset = None
        self.heading_frame = heading_frame

        self.zero_heading_frame_vrw = None
        self.zero_heading_frame_w = None

        self.fixed_head_z = fixed_head_z
        self.max_head_z = max_head_z

        self.fixed_head_quat_h = fixed_head_quat_h
        self.fixed_wrist_poses_h = fixed_wrist_poses_h

        self.fix_head = fix_head
        self.fixed_head_pos_vrw = None
        self.align_z_to_vr = align_z_to_vr

        # Pause state
        self._is_paused = False
        self._paused_pose_w: Optional[HeadWristsPose] = None
        # Cache heading only for self-heading mode (since it comes from VR)
        self._paused_heading: Optional[tuple[NDArray[np.float32], NDArray[np.float32]]] = None

    @property
    def tracker(self) -> VrVelPoseTracker:
        return self.observator.tracker

    @property
    def is_paused(self) -> bool:
        """Check if the observator is currently paused."""
        return self._is_paused

    def start_pause(self):
        """
        Start pause mode. The world-frame pose will be cached and heading-frame
        values will be recomputed based on fresh heading for each get_observation() call.
        """
        if not self._is_paused:
            current_obs = self._get_observation_impl()
            # Cache world-frame pose
            self._paused_pose_w = current_obs.head_wrists_pose_w

            # If using self heading, also cache heading (since it comes from VR)
            if self.use_self_heading_frame:
                self._paused_heading = (
                    np.asarray(current_obs.heading_pose.pos, dtype=np.float32),
                    np.asarray(current_obs.heading_pose.quat, dtype=np.float32)
                )

            self._is_paused = True

    def stop_pause(self):
        """
        Stop pause mode. Resets the head offset and continues normal operation.
        """
        if self._is_paused:
            self._is_paused = False
            self._paused_pose_w = None
            self._paused_heading = None
            # Reset offsets when resuming
            self.reset_head_offset()

    def reset_head_offset(self):
        self.tracker.reset_velocity_estimators()
        self.zero_heading_frame_vrw = self.get_cur_vr_heading_vrw()
        self.zero_heading_frame_w = self.get_cur_robot_heading_w()

        if self.align_z_to_vr:
            self.zero_heading_frame_vrw[0][2] = self.zero_heading_frame_w[0][2]

        if self.max_head_z is not None:
            available_h = self.max_head_z - self.zero_heading_frame_w[0][2]
            self.clip_head_to_vrw = self.zero_heading_frame_vrw[0][2] + available_h
        if self.fixed_head_z is not None:
            self.head_z_pos_vrw = self.fixed_head_z - self.zero_heading_frame_w[0][2] + self.zero_heading_frame_vrw[0][2]
        if self.fix_head:
            vr_obs = self.observator.get_observation()
            self.fixed_head_pos_vrw = vr_obs.head_wrists_pose_vrw.pos[0].copy()
        # self.zero_heading_frame_w[0][2] = 1.23

    def get_localization_vrw(self) -> LocalizationObs:
        return self.localizator.get_observation()

    def get_cur_robot_heading_w(self):
        obs = self.localizator.get_observation()
        return heading_frame_from_head(obs.pose.pos, obs.pose.quat, heading_frame=self.heading_frame)

    def get_cur_vr_heading_w(self):
        pos, quat = self.get_cur_vr_heading_vrw()
        return apply_transform_from_new_base(
                pos=pos,
                quat=quat,
                old_base_pos=self.zero_heading_frame_vrw[0],
                old_base_quat=self.zero_heading_frame_vrw[1],
                new_base_pos=self.zero_heading_frame_w[0],
                new_base_quat=self.zero_heading_frame_w[1],
        )

    def get_cur_vr_heading_vrw(self):
        vr_obs = self.observator.get_observation()
        if self.angled_vr_heading:
            self.head_quat_on_reset = vr_obs.head_wrists_pose_vrw.quat[0]
        return heading_frame_from_head(vr_obs.head_wrists_pose_vrw.pos[0], vr_obs.head_wrists_pose_vrw.quat[0], heading_frame=self.heading_frame)

    def get_observation(self) -> VrHeadingObservations:
        """
        Get the current observation. If paused, returns cached world-frame pose
        with heading-frame values recomputed from current heading.
        """
        if self._is_paused and self._paused_pose_w is not None:
            return self._get_paused_observation()
        return self._get_observation_impl()

    def _get_paused_observation(self) -> VrHeadingObservations:
        """
        Compute observation during pause: frozen world-frame pose with
        heading-frame values recomputed from current heading.
        """
        # Get current heading frame
        if self.use_self_heading_frame:
            # Use cached heading (from VR at pause time) - consistent with frozen VR pose
            heading_pos_w, heading_quat_w = self._paused_heading
        else:
            # Get fresh heading from robot - robot may have moved
            heading_pos_w, heading_quat_w = self.get_cur_robot_heading_w()

        # Transform cached world-frame poses to heading frame
        pos_h, quat_h = frame2_in_frame1(
            heading_pos_w, heading_quat_w,
            self._paused_pose_w.pos, self._paused_pose_w.quat
        )

        return VrHeadingObservations(
            head_wrists_pose_w=self._paused_pose_w,
            head_wrists_vel_w=HeadWristsVelocity.zero(),
            head_wrists_pose_h=HeadWristsPose(
                pos=pos_h.astype(np.float32),
                quat=quat_h.astype(np.float32)
            ),
            head_wrists_vel_h=HeadWristsVelocity.zero(),
            heading_pose=Pose(
                pos=heading_pos_w,
                quat=heading_quat_w
            ),
            head_wrists_pose_vrw=None,
            head_wrists_vel_vrw=None,
        )

    def _get_observation_impl(self) -> VrHeadingObservations:
        """
        Internal implementation of get_observation that always computes fresh values.
        """
        if self.zero_heading_frame_vrw is None:
            self.reset_head_offset()

        obs = deepcopy(self.observator.get_observation())

        if self.fix_head and self.fixed_head_pos_vrw is not None:
            obs.head_wrists_pose_vrw.pos[0] = self.fixed_head_pos_vrw
        elif self.fixed_head_z is not None:
            obs.head_wrists_pose_vrw.pos[0, 2] = self.head_z_pos_vrw
        elif self.max_head_z is not None:
            obs.head_wrists_pose_vrw.pos[0, 2] = np.clip(obs.head_wrists_pose_vrw.pos[0, 2], -1e9, self.clip_head_to_vrw)

        # Transform poses from VRW to W
        vr_pos_w, vr_quats_w = apply_transform_from_new_base(
                pos=obs.head_wrists_pose_vrw.pos,
                quat=obs.head_wrists_pose_vrw.quat,
                old_base_pos=self.zero_heading_frame_vrw[0][None, ...],
                old_base_quat=self.zero_heading_frame_vrw[1][None, ...],
                new_base_pos=self.zero_heading_frame_w[0][None, ...],
                new_base_quat=self.zero_heading_frame_w[1][None, ...],
        )
        if self.angled_vr_heading:
            cur_quat = R.from_quat(vr_quats_w[0], scalar_first=True)
            old_heading_quat = R.from_quat(self.zero_heading_frame_vrw[1], scalar_first=True)
            fix_heading_quat =  R.from_quat(self.head_quat_on_reset, scalar_first=True)
            new_q = cur_quat * fix_heading_quat.inv() * old_heading_quat
            vr_quats_w[0] = new_q.as_quat(scalar_first=True)

        heading_pos_w, heading_quat_w = self.get_cur_vr_heading_w()

        if self.fix_head:
            vr_quats_w[0] = self.zero_heading_frame_w[1]
        elif self.fixed_head_quat_h is not None:
            vr_quats_w[0] = quat_mul(heading_quat_w, self.fixed_head_quat_h)
        if self.fixed_wrist_poses_h is not None:
            pos_h = np.stack([
                self.fixed_wrist_poses_h.left_wrist_pose.pos,
                self.fixed_wrist_poses_h.right_wrist_pose.pos
            ])
            quat_h = np.stack([
                self.fixed_wrist_poses_h.left_wrist_pose.quat,
                self.fixed_wrist_poses_h.right_wrist_pose.quat
            ])

            identity_pos = np.zeros(3, dtype=np.float32)
            identity_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

            wrist_pos_w, wrist_quat_w = apply_transform_from_new_base(
                pos=pos_h,
                quat=quat_h,
                old_base_pos=identity_pos,
                old_base_quat=identity_quat,
                new_base_pos=heading_pos_w,
                new_base_quat=heading_quat_w,
            )

            vr_pos_w[1:3] = wrist_pos_w
            vr_quats_w[1:3] = wrist_quat_w


        # Transform velocities from VRW to W
        vr_lin_vel_w, vr_ang_vel_w = transform_velocity_from_new_base(
            lin_vel=obs.head_wrists_vel_vrw.lin_vel,
            ang_vel=obs.head_wrists_vel_vrw.ang_vel if obs.head_wrists_vel_vrw.ang_vel is not None else np.zeros((3, 3), dtype=np.float32),
            old_base_quat=self.zero_heading_frame_vrw[1],
            new_base_quat=self.zero_heading_frame_w[1],
        )

        if self.fix_head:
            vr_lin_vel_w[0] = 0.0
            vr_ang_vel_w[0] = 0.0

        # Get current heading frame
        if self.use_self_heading_frame:
            heading_pos_w, heading_quats_w = self.get_cur_vr_heading_w()
        else:
            heading_pos_w, heading_quats_w = self.get_cur_robot_heading_w()

        # Transform poses from W to H (heading frame)
        vr_pos_h, vr_quats_h = frame2_in_frame1(
            heading_pos_w, heading_quats_w,
            vr_pos_w, vr_quats_w
        )
        # Transform velocities from W to H (heading frame)
        # Use identity quaternion for old base (world) and heading quaternion for rotation
        identity_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        vr_lin_vel_h, vr_ang_vel_h = transform_velocity_from_new_base(
            lin_vel=vr_lin_vel_w,
            ang_vel=vr_ang_vel_w,
            old_base_quat=heading_quats_w,  # Current heading orientation
            new_base_quat=identity_quat,    # Identity gives us R_heading^(-1)
        )

        # print("HEAD QUAT H", vr_quats_h[0].tolist())
        # wrist_poses_h = WristPoses(
        #     left_wrist_pose=Pose(
        #         pos=vr_pos_h[1],
        #         quat=vr_quats_h[1]
        #     ),
        #     right_wrist_pose=Pose(
        #         pos=vr_pos_h[2],
        #         quat=vr_quats_h[2]
        #     )
        # )
        # print(json.dumps(wrist_poses_h.model_dump()))

        return VrHeadingObservations(
            head_wrists_pose_w=HeadWristsPose(
                pos=vr_pos_w,
                quat=vr_quats_w
            ),
            head_wrists_vel_w=HeadWristsVelocity(
                lin_vel=vr_lin_vel_w,
                ang_vel=vr_ang_vel_w if obs.head_wrists_vel_vrw.ang_vel is not None else None
            ),
            head_wrists_pose_h=HeadWristsPose(
                pos=vr_pos_h,
                quat=vr_quats_h
            ),
            head_wrists_vel_h=HeadWristsVelocity(
                lin_vel=vr_lin_vel_h,
                ang_vel=vr_ang_vel_h if obs.head_wrists_vel_vrw.ang_vel is not None else None
            ),
            heading_pose=Pose(
                pos=heading_pos_w,
                quat=heading_quats_w
            ),
            head_wrists_pose_vrw=obs.head_wrists_pose_vrw,
            head_wrists_vel_vrw=obs.head_wrists_vel_vrw,
        )


class InterpolatingVrHeadingObservator(VrHeadingObservator):
    """
    Smoothly interpolates between two VR observators over a specified number of steps.
    Captures start and target observations once at start, with zero velocities at endpoints.
    """

    def __init__(
        self,
        source_observator: VrHeadingObservator,
        target_observator: VrHeadingObservator,
        n_steps: int = 150,
        easing: Literal['linear', 'smoothstep', 'cubic'] = 'smoothstep',
        show_progress: bool = True,
        progress_desc: str = "Interpolating VR"
    ):
        self.source_observator = source_observator
        self.target_observator = target_observator
        self.n_steps = n_steps
        self.easing = easing
        self.show_progress = show_progress
        self.progress_desc = progress_desc

        self._current_step = 0
        self._start_obs: Optional[VrHeadingObservations] = None
        self._target_obs: Optional[VrHeadingObservations] = None
        self._is_started = False
        self._pbar: Optional[tqdm] = None

    @property
    def tracker(self) -> VrVelPoseTracker:
        return self.target_observator.tracker

    @property
    def is_finished(self) -> bool:
        return self._current_step >= self.n_steps

    @property
    def is_started(self) -> bool:
        return self._is_started

    @property
    def progress(self) -> float:
        return min(self._current_step / self.n_steps, 1.0)

    def start(self):
        """Capture start and target observations and begin interpolation."""
        self._capture_observations()
        self._current_step = 0
        self._is_started = True

        if self.show_progress:
            self._pbar = tqdm(
                total=self.n_steps,
                desc=self.progress_desc,
                unit="step",
                leave=False,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
            )

    def _capture_observations(self):
        """Capture observations from both observators with zero velocities."""
        self._start_obs = self.source_observator.get_observation().with_zero_velocities()
        self._target_obs = self.target_observator.get_observation().with_zero_velocities()

    def reset(self):
        """Reset the interpolation to start fresh."""
        self._close_progress_bar()
        self._current_step = 0
        self._start_obs = None
        self._target_obs = None
        self._is_started = False

    def reset_head_offset(self):
        """Reset head offset and re-capture observations."""
        self.target_observator.reset_head_offset()
        if self._is_started:
            self._capture_observations()
        self._current_step = 0

    def start_pause(self):
        """Not used in interpolator - velocities already zero."""
        pass

    def stop_pause(self):
        """Not used in interpolator."""
        pass

    def _close_progress_bar(self):
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None

    def get_observation(self) -> VrHeadingObservations:
        """Get the current interpolated observation."""
        if not self._is_started:
            self.start()

        # Always get fresh observation from target for heading_pose
        current_target_obs = self.target_observator.get_observation()
        heading_pose = current_target_obs.heading_pose
        heading_pos_w = np.asarray(heading_pose.pos, dtype=np.float32)
        heading_quat_w = np.asarray(heading_pose.quat, dtype=np.float32)

        # If finished, use target's world pose but recompute heading-relative
        if self.is_finished:
            return self._compute_observation_with_heading(
                self._target_obs.head_wrists_pose_w,
                heading_pos_w,
                heading_quat_w,
                heading_pose
            )

        # Calculate eased interpolation factor
        t = self._apply_easing(self._current_step / self.n_steps)

        # Interpolate only world-frame poses
        interpolated_pose_w = self._lerp_pose(
            self._start_obs.head_wrists_pose_w,
            self._target_obs.head_wrists_pose_w,
            t
        )

        # Compute full observation with fresh heading
        result = self._compute_observation_with_heading(
            interpolated_pose_w,
            heading_pos_w,
            heading_quat_w,
            heading_pose
        )

        # Advance step
        self._current_step += 1
        if self._pbar is not None:
            self._pbar.update(1)
            if self.is_finished:
                self._close_progress_bar()

        return result

    def _compute_observation_with_heading(
        self,
        pose_w: HeadWristsPose,
        heading_pos_w: NDArray[np.float32],
        heading_quat_w: NDArray[np.float32],
        heading_pose: Pose,
    ) -> VrHeadingObservations:
        """
        Compute full observation from world-frame pose and current heading.
        Similar to VrHeadingObservatorImpl._get_observation_impl().
        """
        # Transform poses from W to H (heading frame)
        pos_h, quat_h = frame2_in_frame1(
            heading_pos_w, heading_quat_w,
            pose_w.pos, pose_w.quat
        )

        return VrHeadingObservations(
            head_wrists_pose_w=pose_w,
            head_wrists_vel_w=HeadWristsVelocity.zero(),
            head_wrists_pose_h=HeadWristsPose(
                pos=pos_h.astype(np.float32),
                quat=quat_h.astype(np.float32)
            ),
            head_wrists_vel_h=HeadWristsVelocity.zero(),
            heading_pose=heading_pose,
            head_wrists_pose_vrw=None,
            head_wrists_vel_vrw=None,
        )

    def _apply_easing(self, t: float) -> float:
        if self.easing == 'linear':
            return t
        elif self.easing == 'smoothstep':
            return t * t * (3 - 2 * t)
        elif self.easing == 'cubic':
            return 4 * t * t * t if t < 0.5 else 1 - pow(-2 * t + 2, 3) / 2
        return t

    def _lerp_pose(self, start: HeadWristsPose, end: HeadWristsPose, t: float) -> HeadWristsPose:
        """Linear interpolation for positions, slerp for quaternions."""
        pos = ((1 - t) * start.pos + t * end.pos).astype(np.float32)
        quat = np.stack([
            self._slerp(start.quat[i], end.quat[i], t)
            for i in range(start.quat.shape[0])
        ])
        return HeadWristsPose(pos=pos, quat=quat)

    def _slerp(self, q1: NDArray[np.float32], q2: NDArray[np.float32], t: float) -> NDArray[np.float32]:
        """Spherical linear interpolation for quaternions (w, x, y, z)."""
        r1 = Rotation.from_quat(q1, scalar_first=True)
        r2 = Rotation.from_quat(q2, scalar_first=True)
        return Slerp([0, 1], Rotation.concatenate([r1, r2]))(t).as_quat(scalar_first=True).astype(np.float32)

    def __del__(self):
        self._close_progress_bar()
