"""
TeleopController: programmatic interface for robot deploy.

All external data flows through a single `.act(action)` method.
The caller owns the loop.

Usage:
    controller = TeleopController(checkpoint=..., network='sim_sync_mujoco')
    controller.set_vr_pose_getter(lambda: tcp_vel_tracker.last_pose_state)

    while True:
        action = TeleopAction(vr_state=vr_state, fsm_target=...)
        controller.act(action)
        time.sleep(0.01)
"""

import logging
import os
import threading
import time
import typing as tp

from scipy.spatial.transform import Rotation as Rot

from lib.deploy.control.base_env import BaseEnv

if 'MUJOCO_GL' not in os.environ:
    os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import torch

from pydantic import BaseModel
from typing import Optional, Union, List
import multiprocessing as mp

from lib.assets.paths import resolve_asset, resolve_checkpoint
from lib.deploy.configs.deploy_defaults import (
    DEFAULT_CHECKPOINT,
    DEFAULT_T_POSE_TRAJ,
    HEADING_FRAME_SOURCE,
)
from lib.vendor.mocap.constants import HeadingFrameSource
from lib.deploy.control.audio_alert_player import AudioAlertPlayer
from lib.deploy.control.config import Config
from lib.deploy.control.env import G1Env
from lib.deploy.fsm.common import Edge, Node
from lib.deploy.fsm.damp_node import DampNode
from lib.deploy.fsm.start_state_node import StartStateNode
from lib.deploy.vr.localization import LocalizationObservator, create_steppable_localization_observator
from lib.deploy.vr.vr_observations import (
    create_vr_heading_observator,
    create_vr_observator,
    create_interpolating_vr_observator,
    SnapshotVrHeadingObservator,
)
from lib.deploy.vr.vr_pose_tracker import VRPoseState
from lib.deploy.vr.vr_pose_tracker_with_velocity import VrVelPoseTracker
from lib.deploy.vr.vr_replay_wrapper import VRReplayWrapper
from lib.deploy.vr.vr_tracker_wrappers import PoseTrackerFromTCPVel, VelPoseTrackerFromTCPVel
from lib.deploy.vr.vr_tcp_vel_tracker import VRTCPVelTracker, VRTCPVelState, Velocity
from lib.deploy.configs.joint_mappings import REAL_ORDER
from lib.deploy.deploy_utils.forward_kinematics import RobotModel, RobotModelConfig
from lib.deploy.deploy_utils.networks import is_mock_network, is_sync_sim_network, MockNetworkType
from lib.deploy.deploy_utils.commander import MoveCommander
from lib.deploy.debug.logs import DeployConsole
from lib.deploy.debug.profiler import Profiler

from lib.deploy.fsm.fsm import FiniteStateMachine
from lib.vendor.env import LOCAL_ROOT

from .env_processor import EnvProcessor
from .fsm.policy_node import PolicyNode
from .utils.scheduler import try_enable_rr_scheduler
from lib.logging import ChunkedJsonlLog, ChunkedJsonlLogConfig, ControllerFsmLogger

CHECKPOINT = DEFAULT_CHECKPOINT

# ---------------------------------------------------------------------------
# TeleopAction
# ---------------------------------------------------------------------------

class GripperAction(BaseModel):
    mode: Optional[str] = None
    pos: Optional[float] = None
    teleop_button: Optional[bool] = None
    joints_targets: Optional[List[float]] = None


class TeleopAction(BaseModel):
    """All data from external loop to robot passes through this."""

    vr_state: Optional[VRTCPVelState] = None
    fsm_target: Optional[Union[int, str]] = None
    left_gripper: Optional[GripperAction] = None
    right_gripper: Optional[GripperAction] = None

    class Config:
        arbitrary_types_allowed = True

# ---------------------------------------------------------------------------
# ExternalVRTCPVelTracker
# ---------------------------------------------------------------------------

class ExternalVRTCPVelTracker(VRTCPVelTracker):
    """VRTCPVelTracker that accumulates velocity from external .set_state() calls.

    Thread-safe: set_state() is called from the outer loop thread,
    consume() and get_state() are called from the inner PolicyNode thread.

    Protocol:
        # Outer loop (producer, ~100Hz):
        tracker.set_state(vel_state)   # accumulates displacement

        # Inner loop (consumer, 50Hz), once per tick:
        tracker.consume()              # freeze accumulated data into snapshot
        tracker.get_state()            # returns frozen snapshot (idempotent)
    """

    def __init__(self):
        self._lock = threading.Lock()

        # Accumulator: linear displacement (additive — correct for translations)
        self._acc_head_xyz = np.zeros(3, dtype=np.float64)
        self._acc_left_xyz = np.zeros(3, dtype=np.float64)
        self._acc_right_xyz = np.zeros(3, dtype=np.float64)

        # Accumulator: rotational displacement (must COMPOSE, not add)
        self._acc_head_rot = Rot.identity()
        self._acc_left_rot = Rot.identity()
        self._acc_right_rot = Rot.identity()

        self._acc_dt = 0.0

        # Interpolation: total displacement to spread across inner ticks
        self._interp_head_xyz = np.zeros(3, dtype=np.float64)
        self._interp_left_xyz = np.zeros(3, dtype=np.float64)
        self._interp_right_xyz = np.zeros(3, dtype=np.float64)
        self._interp_head_rotvec_deg = np.zeros(3, dtype=np.float64)
        self._interp_left_rotvec_deg = np.zeros(3, dtype=np.float64)
        self._interp_right_rotvec_deg = np.zeros(3, dtype=np.float64)
        self._interp_n = 0       # total steps for current spread
        self._interp_step = 0    # steps delivered so far

        # Outer frequency estimation
        self._last_data_time = None
        self._outer_period_ema = None
        self._ema_alpha = 0.3
        self._inner_dt = 0.02    # 50Hz inner loop

        # Frozen snapshot returned by get_state()
        self._snapshot = self._zero_state()

    @staticmethod
    def _zero_state():
        return VRTCPVelState(
            head=Velocity(xyz=np.zeros(3), rotvec=np.zeros(3)),
            left_wrist=Velocity(xyz=np.zeros(3), rotvec=np.zeros(3)),
            right_wrist=Velocity(xyz=np.zeros(3), rotvec=np.zeros(3)),
            dt=0.0,
        )

    def set_state(self, state: VRTCPVelState):
        """Called from outer loop. Accumulates displacement.

        Linear displacement: additive (correct for translations).
        Rotational displacement: composed via rotation multiplication
        (rotvecs are NOT additive — R.from_rotvec(a+b) ≠ R(a)*R(b)).
        """
        with self._lock:
            dt = state.dt
            if dt > 0:
                self._acc_head_xyz += state.head.xyz * dt
                self._acc_left_xyz += state.left_wrist.xyz * dt
                self._acc_right_xyz += state.right_wrist.xyz * dt

                # Compose rotations (left-multiply: later rotation on the left)
                self._acc_head_rot = Rot.from_rotvec(
                    np.radians(state.head.rotvec * dt)) * self._acc_head_rot
                self._acc_left_rot = Rot.from_rotvec(
                    np.radians(state.left_wrist.rotvec * dt)) * self._acc_left_rot
                self._acc_right_rot = Rot.from_rotvec(
                    np.radians(state.right_wrist.rotvec * dt)) * self._acc_right_rot

                self._acc_dt += dt

                # Outer frequency estimation
                now = time.monotonic()
                if self._last_data_time is not None:
                    period = now - self._last_data_time
                    if self._outer_period_ema is None:
                        self._outer_period_ema = period
                    else:
                        self._outer_period_ema = (self._ema_alpha * period
                                                  + (1 - self._ema_alpha) * self._outer_period_ema)
                self._last_data_time = now

    def consume(self):
        """Freeze accumulated data into snapshot. Call once per inner tick.

        Spreads displacement evenly across inner ticks using N-step linear
        interpolation with telescoping sums, guaranteeing drift-free delivery.

        When outer loop is slower than inner (e.g. 10Hz vs 50Hz), displacement
        is spread over n_spread inner ticks instead of dumping all on the first.

        Drift-free guarantee via telescoping: each step i delivers
        total * ((i+1)/n) - total * (i/n). The sum telescopes to total exactly.
        """
        with self._lock:
            if self._acc_dt > 1e-9:
                # --- New data arrived: set up interpolation ---

                # 1. Compute un-delivered remainder from current interpolation
                if self._interp_n > 0 and self._interp_step < self._interp_n:
                    frac_remaining = 1.0 - self._interp_step / self._interp_n
                    remainder_head_xyz = self._interp_head_xyz * frac_remaining
                    remainder_left_xyz = self._interp_left_xyz * frac_remaining
                    remainder_right_xyz = self._interp_right_xyz * frac_remaining
                    # Rotation remainder as Rotation objects (rotvec scaling
                    # is exact for single-axis interpolation windows)
                    remainder_head_rot = Rot.from_rotvec(
                        np.radians(self._interp_head_rotvec_deg * frac_remaining))
                    remainder_left_rot = Rot.from_rotvec(
                        np.radians(self._interp_left_rotvec_deg * frac_remaining))
                    remainder_right_rot = Rot.from_rotvec(
                        np.radians(self._interp_right_rotvec_deg * frac_remaining))
                else:
                    remainder_head_xyz = np.zeros(3, dtype=np.float64)
                    remainder_left_xyz = np.zeros(3, dtype=np.float64)
                    remainder_right_xyz = np.zeros(3, dtype=np.float64)
                    remainder_head_rot = Rot.identity()
                    remainder_left_rot = Rot.identity()
                    remainder_right_rot = Rot.identity()

                # 2. Combine remainder + new accumulation
                # Linear: additive (exact)
                self._interp_head_xyz = remainder_head_xyz + self._acc_head_xyz
                self._interp_left_xyz = remainder_left_xyz + self._acc_left_xyz
                self._interp_right_xyz = remainder_right_xyz + self._acc_right_xyz
                # Rotation: compose (new rotation on top of remainder)
                # rotvec addition is WRONG when axes differ — must compose.
                self._interp_head_rotvec_deg = np.degrees(
                    (self._acc_head_rot * remainder_head_rot).as_rotvec())
                self._interp_left_rotvec_deg = np.degrees(
                    (self._acc_left_rot * remainder_left_rot).as_rotvec())
                self._interp_right_rotvec_deg = np.degrees(
                    (self._acc_right_rot * remainder_right_rot).as_rotvec())

                # 3. Estimate n_spread from outer period
                if self._outer_period_ema is not None:
                    self._interp_n = max(1, round(self._outer_period_ema / self._inner_dt))
                else:
                    self._interp_n = 1  # First few samples: deliver immediately

                self._interp_step = 0

                # 4. Clear accumulator
                self._clear_acc()

            # --- Deliver one interpolated step ---
            if self._interp_step < self._interp_n:
                inv_n_dt = 1.0 / (self._interp_n * self._inner_dt)

                self._snapshot = VRTCPVelState(
                    head=Velocity(
                        xyz=self._interp_head_xyz * inv_n_dt,
                        rotvec=self._interp_head_rotvec_deg * inv_n_dt,
                    ),
                    left_wrist=Velocity(
                        xyz=self._interp_left_xyz * inv_n_dt,
                        rotvec=self._interp_left_rotvec_deg * inv_n_dt,
                    ),
                    right_wrist=Velocity(
                        xyz=self._interp_right_xyz * inv_n_dt,
                        rotvec=self._interp_right_rotvec_deg * inv_n_dt,
                    ),
                    dt=self._inner_dt,
                )
                self._interp_step += 1
            else:
                # All interpolation steps consumed — freeze position.
                self._snapshot = self._zero_state()

    def get_state(self) -> VRTCPVelState:
        """Returns frozen snapshot. Idempotent between consume() calls."""
        return self._snapshot

    def reset(self):
        """Clear all state. Call on node activation."""
        with self._lock:
            self._clear_acc()
            self._snapshot = self._zero_state()
            # Clear interpolation state
            self._interp_head_xyz[:] = 0
            self._interp_left_xyz[:] = 0
            self._interp_right_xyz[:] = 0
            self._interp_head_rotvec_deg[:] = 0
            self._interp_left_rotvec_deg[:] = 0
            self._interp_right_rotvec_deg[:] = 0
            self._interp_n = 0
            self._interp_step = 0
            self._last_data_time = None
            self._outer_period_ema = None

    def _clear_acc(self):
        self._acc_head_xyz[:] = 0
        self._acc_left_xyz[:] = 0
        self._acc_right_xyz[:] = 0
        self._acc_head_rot = Rot.identity()
        self._acc_left_rot = Rot.identity()
        self._acc_right_rot = Rot.identity()
        self._acc_dt = 0.0


# ---------------------------------------------------------------------------
# ExternalVRPolicyNode
# ---------------------------------------------------------------------------

class ExternalVRPolicyNode(PolicyNode):
    """PolicyNode that sets VR pose via callback when DoTeleop activates.

    Manages the consume() cycle for ExternalVRTCPVelTracker and
    PoseTrackerFromTCPVel so that get_state() is idempotent within each tick.
    """

    def set_vr_pose_callback(self, callback):
        self._vr_pose_callback = callback

    def set_vel_tracker(self, tracker: 'ExternalVRTCPVelTracker'):
        self._vel_tracker = tracker

    def set_pose_from_vel_tracker(self, tracker: PoseTrackerFromTCPVel):
        self._pose_from_vel = tracker

    def add_pre_activate_hook(self, fn):
        """Register a callable to run at the very start of _node_loop(), before
        _vr_pose_callback() overwrites the VR pose state."""
        if not hasattr(self, '_pre_activate_hooks'):
            self._pre_activate_hooks = []
        self._pre_activate_hooks.append(fn)

    def _node_loop(self):
        for hook in getattr(self, '_pre_activate_hooks', []):
            hook()
        # Set initial VR pose BEFORE reset_head_offset() in super()._node_loop()
        if hasattr(self, '_vr_pose_callback') and hasattr(self, '_pose_from_vel'):
            pose = self._vr_pose_callback()
            if pose is not None:
                self._pose_from_vel.set_current_state(pose)
                # Reset _prev_time so the first get_state() uses dt=0
                # instead of a huge (cur_time - stale_prev_time).
                self._pose_from_vel._prev_time = None

        # Clear stale accumulated velocity from before this activation
        if hasattr(self, '_vel_tracker'):
            self._vel_tracker.reset()
        if hasattr(self, '_pose_from_vel'):
            self._pose_from_vel.consume()

        super()._node_loop()

    def _get_observation(self, carries, timestep: int, dry_run: bool = False):
        # Freeze accumulated velocity and clear pose cache before building
        # the observation. This makes get_state() idempotent for all
        # subsequent calls within the same tick (VR observator chain calls
        # it 2-3 times).
        if hasattr(self, '_vel_tracker'):
            self._vel_tracker.consume()
        if hasattr(self, '_pose_from_vel'):
            self._pose_from_vel.consume()
        return super()._get_observation(carries, timestep, dry_run=dry_run)


class RealtimePolicyNode(PolicyNode):
    def _node_loop(self):
        try_enable_rr_scheduler(self.name)
        super()._node_loop()


class RealtimeExternalVRPolicyNode(ExternalVRPolicyNode):
    def _node_loop(self):
        try_enable_rr_scheduler(self.name)
        super()._node_loop()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_env(config: Config, network: str, save_videos: bool, window: bool = False) -> BaseEnv:
    if network == MockNetworkType.SIM_SYNC_MUJOCO:
        from lib.deploy.control.mujoco_env import G1MujocoEnv
        # window=True forces the interactive MuJoCo viewer (headless=False);
        # otherwise auto-detect from DISPLAY.
        return G1MujocoEnv(
            config, network=network, save_videos=save_videos,
            headless=False if window else None,
        )
    return G1Env(config, network=network)


def _get_localization_observator(
        env, network: str, robot_model: RobotModel, log_level: int = logging.INFO,
        latency_range_ms: tp.Tuple[int, int] = (0, 0),
        decimation_range: tp.Tuple[int, int] = (1, 1),
        pos_noise_m: float = 0.0,
        quat_noise_deg: float = 0.0,
        use_fk_for_z: bool = True,
) -> LocalizationObservator:

    from lib.deploy.configs.joint_mappings import REAL_ORDER
    from lib.deploy.vr.localization import (
        create_delayed_sim_localization_observator_v2,
        create_localization_observator_v2,
        create_sim_dds_localization_observator,
        create_noisy_localization_observator,
    )
    ORD = REAL_ORDER

    def _getter():
        _joints = env.get_qj()
        return {joint_name: (_joints[joint_idx]) for joint_idx, joint_name in enumerate(ORD)}

    if network == MockNetworkType.SIM:
        if use_fk_for_z:
            raise NotImplementedError()
        else:
            return create_sim_dds_localization_observator()
    elif is_sync_sim_network(network):
        delayed_observator = create_delayed_sim_localization_observator_v2(
            env.sim_env,
            latency_range_ms=latency_range_ms,
            decimation_range=decimation_range,
            use_fk_height=use_fk_for_z,
            robot_model=robot_model,
            joint_state_getter=_getter,
        )
        return create_noisy_localization_observator(
            base_observator=delayed_observator,
            pos_noise_m=pos_noise_m,
            quat_noise_deg=quat_noise_deg,
        )
    else:
        return create_localization_observator_v2(
            log_level=log_level,
            use_fk_height=use_fk_for_z,
            robot_model=robot_model,
            joint_state_getter=_getter,
        )


# ---------------------------------------------------------------------------
# TeleopController
# ---------------------------------------------------------------------------

class TeleopController:
    """Programmatic interface for the teleoperation deploy pipeline.

    Builds the teleoperation FSM graph::

        ┌────────────┐    ┌──────────────────┐    ┌────────────┐    ┌────────────┐    ┌───────────┐
        │ DampNode   │───>│StartStateNode    │───>│JustStand   │───>│MiddleNode  │───>│DoTeleop   │
        └────────────┘    └──────────────────┘    └────────────┘    └────────────┘    └───────────┘
                                                                                          │    ^
                                                                                          V    │
                                                                                        ┌─────────────┐
                                                                                        │ReturnMiddle │
                                                                                        └─────────────┘

        All nodes ──edge──> DampNode  (safety/reset)

    The caller owns the loop::

        controller = TeleopController(...)
        while True:
            action = TeleopAction(vr_state=..., fsm_target=...)
            controller.act(action)
            time.sleep(0.01)
    """

    def __init__(
        self,
        checkpoint: str = CHECKPOINT,
        network: str = None,
        device: str = 'cuda:0',
        heading_frame_source: HeadingFrameSource = HEADING_FRAME_SOURCE,
        dex3: bool = False,
        latency_range_ms: tp.Optional[tp.Tuple[int, int]] = None,
        decimation_range: tp.Optional[tp.Tuple[int, int]] = None,
        save_videos: tp.Optional[bool] = None,
        restart_localization: bool = False,
        debug: bool = False,
        profile: bool = False,
        vel_v2: bool = False,
        window: bool = False,
        hands: Optional[str] = 'right',
        env: tp.Optional[BaseEnv] = None,
        log_path: tp.Optional[str] = None,
        log_chunk_duration_s: float = 300.0,
        log_keep_last_n_chunks: int = 12,
        log_previous_session_prune_min_age_s: float = 86400.0,
        log_snapshot_every_n: int = 1,
    ):
        self._network = network
        self._dex3 = dex3
        self._device = device
        self._heading_frame_source = heading_frame_source
        self._started = False
        self._last_vr_update_time = time.monotonic()
        self._fsm_logger: tp.Optional[ControllerFsmLogger] = None

        log_level = logging.DEBUG if debug else None

        if save_videos is None:
            self._save_videos = not os.environ.get("DISPLAY", False)
        else:
            self._save_videos = save_videos

        self._is_sync_sim = is_sync_sim_network(network)
        mp.set_start_method('spawn', force=True)

        # --- policy ---
        policy_path = resolve_checkpoint(checkpoint)
        policy = torch.jit.load(policy_path)
        policy.to(device)
        policy = torch.jit.optimize_for_inference(policy)
        policy.eval()

        # --- config & env ---
        import lib.deploy.configs as _c
        config = Config(os.path.join(os.path.dirname(_c.__file__), "g1.yaml"))

        self.env = env if env is not None else _create_env(
            config, network, self._save_videos, window=window,
        )

        # ----- Dex3 -----
        self._left_gripper_value = None
        self._right_gripper_value = None
        self.hand_policy = None
        self._hand_ctrl = None

        if self._dex3:
            from multiprocessing import Value
            from lib.deploy.dex3 import Dex3_1_Gripper_ControllerPolicy
            self._left_gripper_value = Value('d', 0.0, lock=True)
            self._right_gripper_value = Value('d', 0.0, lock=True)
            self.hand_policy = Dex3_1_Gripper_ControllerPolicy(
                self._left_gripper_value, self._right_gripper_value, swap_hands=False)
            if is_mock_network(network):
                raise NotImplementedError(
                    "Dex3 hands require a real robot network interface; not supported on MuJoCo sim."
                )
            else:
                from multiprocessing import Array, Lock
                from lib.deploy.dex3 import Dex3_1_Controller
                dual_hand_data_lock = Lock()
                dual_hand_state_array = Array('d', 14, lock=False)
                dual_hand_sensor_array = Array('d', 2 * 9 * 12, lock=False)
                dual_hand_action_array = Array('d', 14, lock=False)
                self._hand_ctrl = Dex3_1_Controller(
                    self.hand_policy,
                    self._left_gripper_value, self._right_gripper_value,
                    dual_hand_data_lock, dual_hand_state_array,
                    dual_hand_sensor_array, dual_hand_action_array,
                    simulation_mode=False, hands=hands, swap_hands=False,
                    network=network,
                )

        # --- Fast-LIO localization (real robot only) ---
        if not is_mock_network(network):
            from lib.deploy.deploy_utils.localization_launcher import (
                is_dds_localization_running,
                launch_fast_lio,
            )

            # Always-visible status (deploy logging is WARNING-level, so the
            # launcher's own info stream is hidden) — the point is a clear
            # succeeded/failed verdict, not the step-by-step log firehose.
            if not is_dds_localization_running() or restart_localization:
                print(
                    "Localization: launching Fast-LIO on the Jetson "
                    "(restart lio container + start Fast-LIO, ~15-30s)…",
                    flush=True,
                )
                if not launch_fast_lio():
                    print(
                        "Localization: ✗ FAILED (see the errors above; rerun with "
                        "--debug for the full launch log).",
                        flush=True,
                    )
                    raise RuntimeError("Failed to launch Fast-LIO localization")
                print("Localization: ✓ Fast-LIO running (DDS odometry live).", flush=True)
            else:
                print(
                    "Localization: ✓ already running (DDS odometry present; "
                    "pass --restart-localization to relaunch).",
                    flush=True,
                )

        # --- robot model ---
        robot_model = RobotModel(config=RobotModelConfig(
            heading_frame_source=heading_frame_source,
            device=device,
        ))

        # --- localization ---
        localization_observator = create_steppable_localization_observator(
            _get_localization_observator(
                self.env, network=network, robot_model=robot_model,
                log_level=log_level,
                decimation_range=decimation_range or (1, 1),
                latency_range_ms=latency_range_ms or (0, 0),
            )
        )

        # --- profiler ---
        profiler = Profiler(enabled=profile)

        # ----- Dummy VR pipeline (T-pose, used by JustStand) -----
        dummy_tracker = VrVelPoseTracker(VRReplayWrapper(
            resolve_asset(DEFAULT_T_POSE_TRAJ)))
        dummy_vr_observator = create_vr_observator(vel_tracker=dummy_tracker)
        dummy_vr_heading_observator = create_vr_heading_observator(
            localizator=localization_observator,
            observator=dummy_vr_observator,
            use_self_heading_frame=False,
            heading_frame=heading_frame_source.value,
        )

        # ----- External VR pipeline (used by DoTeleop) -----
        self._external_vel_tracker = ExternalVRTCPVelTracker()
        if vel_v2:
            # vel_v2: carry velocity directly from displacement, skip VrVelPoseTracker re-estimation
            self._pose_from_vel = VelPoseTrackerFromTCPVel(
                self._external_vel_tracker, cache_between_consumes=True)
            real_vr_observator = create_vr_observator(vel_tracker=self._pose_from_vel)
        else:
            self._pose_from_vel = PoseTrackerFromTCPVel(
                self._external_vel_tracker, cache_between_consumes=True)
            real_vel_tracker = VrVelPoseTracker(self._pose_from_vel)
            real_vr_observator = create_vr_observator(vel_tracker=real_vel_tracker)

        self.on_pause = False
        self.real_vr_heading_observator = create_vr_heading_observator(
            localizator=localization_observator,
            observator=real_vr_observator,
            use_self_heading_frame=False,
            heading_frame=heading_frame_source.value,
        )

        # ----- Middle (interpolation) VR pipeline -----
        middle_observator = create_interpolating_vr_observator(
            dummy_vr_heading_observator, self.real_vr_heading_observator,
            show_progress=debug,
        )

        # ----- Return-middle (interpolation) VR pipeline -----
        # Interpolates from the DoTeleop pose (snapshot) back to initial_pose.
        # snapshot_obs freezes real_vr's state before _vr_pose_callback() resets it.
        self._snapshot_obs = SnapshotVrHeadingObservator(self.real_vr_heading_observator)
        return_middle_observator = create_interpolating_vr_observator(
            self._snapshot_obs, self.real_vr_heading_observator,
            progress_desc="Returning to initial pose",
            show_progress=debug,
        )

        # ----- EnvProcessors -----
        dummy_env_processor = EnvProcessor(
            robot_model=robot_model,
            vr_observator=dummy_vr_heading_observator,
            localization_observator=localization_observator,
            robot_heading_frame_source=heading_frame_source,
            log_level=log_level,
            device=device,
            profiler=profiler,
        )
        middle_env_processor = EnvProcessor(
            robot_model=robot_model,
            vr_observator=middle_observator,
            localization_observator=localization_observator,
            robot_heading_frame_source=heading_frame_source,
            log_level=log_level,
            device=device,
            profiler=profiler,
        )
        real_env_processor = EnvProcessor(
            robot_model=robot_model,
            vr_observator=self.real_vr_heading_observator,
            localization_observator=localization_observator,
            robot_heading_frame_source=heading_frame_source,
            log_level=log_level,
            device=device,
            profiler=profiler,
        )
        return_middle_env_processor = EnvProcessor(
            robot_model=robot_model,
            vr_observator=return_middle_observator,
            localization_observator=localization_observator,
            robot_heading_frame_source=heading_frame_source,
            log_level=log_level,
            device=device,
            profiler=profiler,
        )

        # ----- FSM nodes (teleoperation branch) -----
        self._deploy_console = DeployConsole(
            name="TeleopController",
            audio_alert_player=AudioAlertPlayer(
                self.env.network, config.audio_alerts),
            quiet=not debug,
        )

        damp_node = DampNode(self.env, console=self._deploy_console)
        # Use the configured standing joint angles; startup needs no motion asset.
        start_state_node = StartStateNode(
            self.env,
            console=self._deploy_console,
        )

        just_stand_node = RealtimePolicyNode(
            env=self.env, policy=policy, device=device,
            vr_observator=dummy_vr_heading_observator,
            localization_observator=localization_observator,
            env_processor=dummy_env_processor,
            profiler=profiler,
            name='JustStand',
            motor_temp_cfg=config.motor_temperature,
            console=self._deploy_console,
        )
        middle_node = RealtimeExternalVRPolicyNode(
            env=self.env, policy=policy, device=device,
            vr_observator=middle_observator,
            localization_observator=localization_observator,
            env_processor=middle_env_processor,
            profiler=profiler,
            name='MiddleNode',
            motor_temp_cfg=config.motor_temperature,
            console=self._deploy_console,
        )
        self._do_teleop_node = RealtimeExternalVRPolicyNode(
            env=self.env, policy=policy, device=device,
            vr_observator=self.real_vr_heading_observator,
            localization_observator=localization_observator,
            env_processor=real_env_processor,
            profiler=profiler,
            name='DoTeleop',
            motor_temp_cfg=config.motor_temperature,
            console=self._deploy_console,
        )
        return_middle_node = ExternalVRPolicyNode(
            env=self.env, policy=policy, device=device,
            vr_observator=return_middle_observator,
            localization_observator=localization_observator,
            env_processor=return_middle_env_processor,
            profiler=profiler,
            name='ReturnMiddle',
            motor_temp_cfg=config.motor_temperature,
            console=self._deploy_console,
        )
        # Snapshot must be taken before _vr_pose_callback() resets the VR pose.
        return_middle_node.add_pre_activate_hook(self._snapshot_obs.snapshot)

        # ----- Teleoperation branch edges -----
        all_nodes: tp.List[Node] = [
            damp_node, start_state_node,
            self._do_teleop_node, just_stand_node, middle_node, return_middle_node,
        ]
        for node in all_nodes:
            node.add_edge(Edge(node, damp_node))

        damp_node.add_edge(Edge(source=damp_node, target=start_state_node))
        start_state_node.add_edge(Edge(source=start_state_node, target=just_stand_node))
        just_stand_node.add_edge(Edge(source=just_stand_node, target=middle_node))
        middle_node.add_edge(Edge(source=middle_node, target=self._do_teleop_node))
        self._do_teleop_node.add_edge(Edge(source=self._do_teleop_node, target=return_middle_node))
        return_middle_node.add_edge(Edge(source=return_middle_node, target=self._do_teleop_node))

        # ----- FSM -----
        self._fsm = FiniteStateMachine(
            damp_node, print_every=1e6, console=self._deploy_console)

        # Store refs
        self._damp_node = damp_node
        self._start_state_node = start_state_node
        self._just_stand_node = just_stand_node
        self._middle_node: ExternalVRPolicyNode = middle_node
        self._middle_observator = middle_observator
        self._return_middle_node: ExternalVRPolicyNode = return_middle_node
        self._return_middle_observator = return_middle_observator
        self._config = config

        # ----- Velocity commander (no stdin thread — controller is a pure receiver) -----
        self.commander = MoveCommander()
        self._policy_node = self._do_teleop_node

        # ----- Start FSM (activate damp node so the robot is under control) -----
        callbacks: tp.List = []
        if debug or log_path is not None:
            default_log_dir = os.path.join(LOCAL_ROOT, "tmp", "yact_controller_logs")
            resolved_log_dir = default_log_dir if log_path is None else (
                (os.path.dirname(log_path) or default_log_dir) if log_path.endswith(".jsonl") else log_path
            )
            log_name = "yact_controller" if log_path is None else (
                os.path.splitext(os.path.basename(log_path))[0]
                if log_path.endswith(".jsonl")
                else "yact_controller"
            )
            chunked_log = ChunkedJsonlLog(
                name=log_name,
                config=ChunkedJsonlLogConfig(
                    log_dir=resolved_log_dir,
                    chunk_duration_s=log_chunk_duration_s,
                    keep_last_n_chunks=log_keep_last_n_chunks,
                    previous_session_prune_min_age_s=log_previous_session_prune_min_age_s,
                ),
            )
            self._fsm_logger = ControllerFsmLogger(
                self,
                chunked_log,
                snapshot_every_n=log_snapshot_every_n,
            )
            if debug:
                print(
                    f"[TeleopController] FSM logger session={self._fsm_logger.session_id} "
                    f"log_dir={self._fsm_logger.log_dir} "
                    f"chunk_duration_s={log_chunk_duration_s} "
                    f"keep_last_n_chunks={log_keep_last_n_chunks} "
                    f"previous_session_prune_min_age_s={log_previous_session_prune_min_age_s} "
                    f"snapshot_every_n={log_snapshot_every_n}"
                )
            callbacks = [self._fsm_logger]
        self.start(callbacks)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def start(self, callbacks: tp.Optional[List] = None):
        """Activate the FSM (starts the initial node). Called automatically by act()."""
        if not self._started:
            self._fsm.run_async(callbacks=callbacks or [])
            self._started = True

    def act(self, action: TeleopAction):
        """Single tick of the control loop.

        Call this in your own while-loop.
        """

        # 1. Feed vr_state to the VR tracker
        if action.vr_state is not None:
            self._external_vel_tracker.set_state(action.vr_state)
            self._last_vr_update_time = time.monotonic()

        # 2. Process FSM transition
        if action.fsm_target is not None:
            # TODO: add proper transition guards to FSM architecture and remove this block
            if (isinstance(self._fsm._current_node, DampNode)
                    and self._config is not None
                    and self._config.motor_temperature is not None):
                temps = self.env.get_motor_temperatures()
                max_temp = max(temps)
                cooldown_thresh = self._config.motor_temperature.warning_threshold
                if max_temp > cooldown_thresh:
                    hot = ", ".join(f"{n}({t:.0f})" for n, t in zip(REAL_ORDER, temps) if t > cooldown_thresh)
                    print(f"[Controller.act] BLOCKED: motors too hot to leave DampNode "
                          f"(max={max_temp:.0f}°C > {cooldown_thresh:.0f}°C): {hot}. "
                          f"Wait for cooldown.")
                    action.fsm_target = None

            if action.fsm_target is not None:
                if isinstance(action.fsm_target, int):
                    self._fsm.move_by_id(action.fsm_target, cause="Outer loop")
                else:
                    self._fsm.move_by_name(action.fsm_target, cause="Outer loop")

        # 3. Update dex3 grippers
        if self._dex3:
            if action.left_gripper is not None and self._left_gripper_value is not None and action.left_gripper.teleop_button:
                self._left_gripper_value.value = action.left_gripper.pos if action.left_gripper.pos is not None else 0.0
            if action.right_gripper is not None and self._right_gripper_value is not None and action.right_gripper.teleop_button:
                self._right_gripper_value.value = action.right_gripper.pos if action.right_gripper.pos is not None else 0.0

            if self.hand_policy:
                mode = {}
                if action.left_gripper is not None and action.left_gripper.mode is not None:
                    mode['left'] = action.left_gripper.mode
                if action.right_gripper is not None and action.right_gripper.mode is not None:
                    mode['right'] = action.right_gripper.mode

                if self.hand_policy:
                    self.hand_policy.set_mode(mode)

    def stop(self, close_env: bool = True):
        """Deactivate the current FSM node and clean up."""
        try:
            if self._started:
                self._fsm.stop()
                self._started = False
        finally:
            if self._fsm_logger is not None:
                self._fsm_logger.close()
            if close_env:
                self.env.close()

    def set_vr_pose_getter(self, callback):
        """Set callback returning current VRPoseState for DoTeleop/Middle activation."""
        for node in (self._middle_node, self._do_teleop_node, self._return_middle_node):
            node.set_vr_pose_callback(callback)
            node.set_vel_tracker(self._external_vel_tracker)
            node.set_pose_from_vel_tracker(self._pose_from_vel)

    @property
    def current_node_name(self) -> str:
        return self._fsm._current_node.name

    @property
    def current_node(self) -> Node:
        return self._fsm._current_node

    @property
    def is_teleoperating(self) -> bool:
        """True while the DoTeleop node (full VR teleop) is the active FSM node.

        Used to gate VR replay playback so the trajectory only advances once
        teleoperation has actually started, not from the moment the run loop begins.
        """
        return self._fsm._current_node is self._do_teleop_node

    def fsm_graph(self) -> tp.Dict[str, tp.Any]:
        """FSM structure for the terminal/web graph: nodes (with a ``paused`` flag
        for states that don't step/render the sim) and edges with their trigger
        hints (keyboard number, plus ``x``/VR-A -> damp and VR-B -> forward edge).
        """
        from lib.deploy.debug.logs import _display_fsm_node
        from lib.deploy.fsm.common import edge_trigger_keys

        order = [
            self._damp_node, self._start_state_node, self._just_stand_node,
            self._middle_node, self._do_teleop_node, self._return_middle_node,
        ]
        paused = {self._damp_node, self._start_state_node}
        nodes = [{"name": _display_fsm_node(n.name), "paused": n in paused} for n in order]

        edges = []
        for node in order:
            src = _display_fsm_node(node.name)
            for edge_id, edge in enumerate(node.edges):
                if edge.target is node:
                    continue  # skip the damp self-loop
                keys = edge_trigger_keys(edge_id, edge.target is self._damp_node)
                edges.append({"from": src, "to": _display_fsm_node(edge.target.name), "keys": keys})
        return {"nodes": nodes, "edges": edges}

    @property
    def network(self) -> str:
        return self._network

    @property
    def deploy_console(self) -> DeployConsole:
        """Rich logger used by FSM nodes."""
        return self._deploy_console

    @property
    def log_path(self) -> tp.Optional[str]:
        return None if self._fsm_logger is None else self._fsm_logger.path

    @property
    def log_dir(self) -> tp.Optional[str]:
        return None if self._fsm_logger is None else self._fsm_logger.log_dir

    def get_state(self) -> tp.Dict[str, tp.Any]:
        """Return current FSM state and proprioception."""
        obs = self.env.get_observation()
        if self._dex3:
            # with self.dual_hand_data_lock:
            left_ee_state, right_ee_state = self._hand_ctrl.get_ee_state()
            left_ee_sensor, right_ee_sensor = self._hand_ctrl.get_ee_sensor()
            obs['left/gripper'] = np.array(left_ee_state, dtype=np.float32)
            obs['right/gripper'] = np.array(right_ee_state, dtype=np.float32)
            obs['left/sensor'] = np.array(left_ee_sensor)
            obs['right/sensor'] = np.array(right_ee_sensor)
            obs['left/gripper_open'] = self.gripper_open_state(self.hand_policy.LEFT_MAPPED_OPEN, self.hand_policy.LEFT_MAPPED_CLOSED, left_ee_state)
            obs['right/gripper_open'] = self.gripper_open_state(self.hand_policy.RIGHT_MAPPED_OPEN, self.hand_policy.RIGHT_MAPPED_CLOSED, right_ee_state)

        node_info = self._fsm._current_node.get_info()

        obs['node_observation'] = node_info

        return {
            'fsm_node': self._fsm._current_node.name,
            'observation': obs,
        }

    def gripper_open_state(self, open_pos, close_pos, gripper_pos):
        middle = (np.array(open_pos[:]) + np.array(close_pos[:])) / 2
        return np.array([int(np.mean(gripper_pos > middle) < 0.5)], dtype=np.uint8)

    # -----------------------------------------------------------------------
    # Auto-next-node helper (for sim testing)
    # -----------------------------------------------------------------------

    def get_auto_next_target(self) -> tp.Optional[str]:
        """Return the next FSM target name for auto-traversal, or None.

        Follows the teleoperation branch: DampNode → StartState → JustStand → Middle → DoTeleop.
        """
        using_mock = is_mock_network(self._network)
        if not using_mock and self._config.div < 1000:
            return None
        name = self._fsm._current_node.name

        if name == self._damp_node.name:
            return self._start_state_node.name
        elif name == self._start_state_node.name:
            return self._just_stand_node.name
        elif name == self._just_stand_node.name:
            return self._middle_node.name
        elif name == self._middle_node.name and self._middle_observator.is_finished:
            return self._do_teleop_node.name
        elif name == self._return_middle_node.name and self._return_middle_observator.is_finished:
            return self._do_teleop_node.name
        return None

    def get_env_time(self) -> float:
        return self.env.get_env_time()

    def time(self) -> float:
        """Universal clock: sim time for sim networks, wall clock for real."""
        return self.env.get_env_time()

    def sleep(self, sleep_for: float, sleep_dt = 1e-3):
        if not is_mock_network(self._network):
            time.sleep(sleep_for)
        else:
            if isinstance(self.current_node, StartStateNode) or isinstance(self.current_node, DampNode):
                time.sleep(sleep_for)
                return
            start_time = self.get_env_time()
            target_time = start_time + sleep_for
            while self.get_env_time() < target_time:
                time.sleep(sleep_dt)
