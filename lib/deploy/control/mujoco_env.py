"""
MuJoCo-based environment wrapper for G1 robot.

Implements BaseSimEnv on the MuJoCo physics backend for sim_sync testing, using
native MuJoCo position control actuators.

IMPORTANT:
- Must set MUJOCO_GL=egl before any imports that might use mujoco to ensure EGL backend is used.
"""

import os
import logging
import threading
from lib.deploy.vr.vr_observations import VrHeadingObservator
from lib.deploy.deploy_utils.forward_kinematics import RobotModel
from lib.deploy.configs.joint_mappings import REAL_ORDER
from lib.deploy.control.util import get_gravity_orientation
from lib.deploy.control.config import Config
from lib.deploy.control.base_sim_env import BaseSimEnv
from lib.assets.paths import resolve_asset
from lib.deploy.configs.deploy_defaults import DEFAULT_MUJOCO_XML
from lib.vendor.env import print_path_lines
from lib.deploy.control.live_view import LiveViewServer
from scipy.spatial.transform import Rotation as R
import torch
import mujoco
import numpy as np
from typing import Dict, List, Union, Callable, Tuple, Optional
import time
from collections import deque
import warnings

logger = logging.getLogger(__name__)


class G1MujocoEnv(BaseSimEnv):
    """
    MuJoCo-based environment wrapper for G1 robot.

    Implements BaseSimEnv on the MuJoCo physics backend for sim_sync testing:
    - step(action): Execute action and return observation
    - reset(): Reset environment and return initial observation
    - get_observation(): Get current observation without taking action
    - apply_external_force_torque_to_link(): Apply external forces for testing
    """

    def _inject_vis_markers(self, xml_content: str) -> str:
        """Inject visualization markers into XML."""
        fk_frame_geom = """
            <geom type="cylinder" size="0.005 0.10" fromto="0 0 0 0.10 0 0" rgba="1 0 0 1" contype="0" conaffinity="0"/>
            <geom type="cylinder" size="0.005 0.10" fromto="0 0 0 0 0.10 0" rgba="0 1 0 1" contype="0" conaffinity="0"/>
            <geom type="cylinder" size="0.005 0.10" fromto="0 0 0 0 0 0.10" rgba="0 0 1 1" contype="0" conaffinity="0"/>
        """

        tgt_frame_geom = """
            <geom type="sphere" size="0.014" rgba="1 0.82 0.15 0.9" contype="0" conaffinity="0"/>
            <geom type="cylinder" size="0.004 0.08" fromto="0 0 0 0.08 0 0" rgba="1 0.45 0.05 0.8" contype="0" conaffinity="0"/>
            <geom type="cylinder" size="0.004 0.08" fromto="0 0 0 0 0.08 0" rgba="0.55 0.25 0.95 0.8" contype="0" conaffinity="0"/>
            <geom type="cylinder" size="0.004 0.08" fromto="0 0 0 0 0 0.08" rgba="0.15 0.75 0.95 0.8" contype="0" conaffinity="0"/>
        """

        markers = ""
        for name in ["vis_fk_head", "vis_fk_left_wrist", "vis_fk_right_wrist", "vis_fk_heading"]:
            markers += f"""
            <body name="{name}" mocap="true" pos="0 0 0">
                {fk_frame_geom}
            </body>"""

        for name in ["vis_tgt_head", "vis_tgt_left_wrist", "vis_tgt_right_wrist"]:
            markers += f"""
            <body name="{name}" mocap="true" pos="0 0 0">
                {tgt_frame_geom}
            </body>"""

        last_wb_idx = xml_content.rfind("</worldbody>")
        if last_wb_idx != -1:
            return xml_content[:last_wb_idx] + markers + xml_content[last_wb_idx:]
        return xml_content

    def _inject_visual_config(self, xml_content: str) -> str:
        """Inject visual config for offscreen rendering."""
        target_attrs = 'offwidth="1280" offheight="720"'

        if "<visual>" in xml_content:
            visual_start = xml_content.find("<visual>")
            visual_end = xml_content.find("</visual>", visual_start)

            if visual_start != -1 and visual_end != -1:
                visual_block = xml_content[visual_start:visual_end]

                if "<global" in visual_block:
                    new_visual_block = visual_block.replace(
                        "<global", f"<global {target_attrs}")
                    return xml_content[:visual_start] + new_visual_block + xml_content[visual_end:]
                else:
                    new_visual_block = visual_block.replace(
                        "<visual>", f"<visual>\n    <global {target_attrs}/>")
                    return xml_content[:visual_start] + new_visual_block + xml_content[visual_end:]

        visual_section = f"""
  <visual>
    <global {target_attrs}/>
  </visual>
"""
        idx = xml_content.find("<worldbody>")
        if idx != -1:
            return xml_content[:idx] + visual_section + xml_content[idx:]

        return xml_content

    def _inject_floor_texture(self, xml_content: str, xml_dir: str) -> str:
        """Use textures/floor.{png,jpg,jpeg} when present; else keep builtin checker."""
        tex_dir = os.path.join(xml_dir, "textures")
        floor_file = None
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = os.path.join(tex_dir, f"floor{ext}")
            if os.path.isfile(candidate):
                floor_file = f"floor{ext}"
                break
        if floor_file is None:
            return xml_content

        logger.info("Floor texture: %s", os.path.join(tex_dir, floor_file))

        builtin_texture = (
            '    <texture type="2d" name="groundplane" builtin="checker" mark="edge"\n'
            '      rgb1="0.84 0.86 0.89" rgb2="0.78 0.80 0.83" markrgb="0.64 0.66 0.69"\n'
            '      width="512" height="512"/>'
        )
        file_texture = f'    <texture type="2d" name="groundplane" file="{floor_file}"/>'
        if builtin_texture not in xml_content:
            return xml_content
        xml_content = xml_content.replace(builtin_texture, file_texture)

        builtin_material = (
            '<material name="groundplane" texture="groundplane" texuniform="true" '
            'texrepeat="16 16" reflectance="0"/>'
        )
        # One floor image spans 4 m x 2 m, giving the bundled texture ~33 x 17 cm planks.
        file_material = (
            '<material name="groundplane" texture="groundplane" texuniform="true" '
            'texrepeat="0.25 0.5" reflectance="0"/>'
        )
        return xml_content.replace(builtin_material, file_material)

    def __init__(
        self,
        config: Config,
        command_source: Optional[Callable[[],
                                          Tuple[float, float, float]]] = None,
        network: Optional[str] = None,
        max_steps: int = 10**10,
        headless: Optional[bool] = None,
        render_width: int = 1280,
        render_height: int = 720,
        save_video_every: Optional[int] = 250,
        obs_latency_ms: float = 0.0,
        save_videos: bool = True
    ):
        """
        Initialize the MuJoCo environment.

        Args:
            config: Configuration object for the robot
            command_source: Optional callback for velocity commands
            network: Network type identifier
            max_steps: Maximum number of steps per episode
            headless: If True, run without visualization. If None, auto-detect.
            render_width: Width for offscreen rendering (default: 1280, lower = faster)
            render_height: Height for offscreen rendering (default: 720, lower = faster)
        """

        self.config = config
        self.command_source = command_source
        self.network = network
        self.action_mask = config.action_mask
        self.max_steps = max_steps
        self.render_width = render_width
        self.render_height = render_height
        self._save_videos = save_videos

        if headless is None:
            headless = not os.environ.get('DISPLAY')
        self.headless = headless

        self.viewer = None
        self._renderer = None

        xml_path = resolve_asset(DEFAULT_MUJOCO_XML)

        logger.debug("Loading MuJoCo model from: %s", xml_path)

        with open(xml_path, 'r') as f:
            xml_content = f.read()

        xml_dir = os.path.dirname(xml_path)
        xml_content = xml_content.replace(
            'meshdir="meshes"', f'meshdir="{xml_dir}/meshes"')
        xml_content = xml_content.replace(
            'texturedir="textures"', f'texturedir="{xml_dir}/textures"')

        xml_content = self._inject_vis_markers(xml_content)
        xml_content = self._inject_floor_texture(xml_content, xml_dir)
        xml_content = self._inject_visual_config(xml_content)

        self.mj_model = mujoco.MjModel.from_xml_string(xml_content)
        self.mj_data = mujoco.MjData(self.mj_model)

        logger.debug(
            "Total robot mass: %.4f kg", np.sum(self.mj_model.body_mass))

        self.mj_model.opt.timestep = 0.001

        self.sim_dt = self.mj_model.opt.timestep
        self.sim_decimation = max(1, int(config.control_dt / self.sim_dt))
        logger.debug(
            "sim_dt=%s, control_dt=%s, decimation=%s",
            self.sim_dt, config.control_dt, self.sim_decimation)
        logger.debug("headless=%s", self.headless)

        self._init_mappings()

        self._update_actuator_gains()

        self._configure_solver()

        if not self.headless:
            try:
                from mujoco import viewer as mj_viewer
                self.viewer = mj_viewer.launch_passive(
                    self.mj_model, self.mj_data,
                    show_left_ui=False, show_right_ui=False
                )
                self.viewer.cam.distance = 3.46
                self.viewer.cam.elevation = -35.3
                self.viewer.cam.azimuth = 45
                logger.debug("Viewer launched")
            except Exception as e:
                logger.warning("Could not launch viewer: %s", e)
                self.viewer = None

        self.qj = np.zeros(config.num_joints, dtype=np.float32)
        self.dqj = np.zeros(config.num_joints, dtype=np.float32)
        self.action = np.zeros(config.action_mask.sum(), dtype=np.float32)
        self.cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.obs = None

        self.common_step_counter = 0
        self._external_force_steps_remaining = 0
        self._external_force_body_id = None

        self.img_buf = deque(maxlen=2000)
        self.last_frame_step = -1
        self.save_video_every = save_video_every
        self._video_output_path: Optional[str] = None
        self._video_saved = False
        self._closed = False
        self._live_view: Optional[LiveViewServer] = None
        self._render_lock = threading.Lock()
        self._last_render_error_log_mono = 0.0
        self._init_timestamp_s = int(time.time())
        self.block = False

        # Observation latency: interpolate between prev and current step
        self._obs_latency_ms = obs_latency_ms
        control_dt_ms = config.control_dt * 1000
        assert obs_latency_ms <= control_dt_ms, \
            f"obs_latency_ms ({obs_latency_ms}) must be <= control_dt ({control_dt_ms} ms)"
        if obs_latency_ms > 0:
            self._obs_alpha = obs_latency_ms / control_dt_ms
            self._obs_history = deque(maxlen=2)

        # Realtime pacing: throttle each step to wall clock so the sim doesn't run
        # faster than realtime. On by default when a viewer window is open; also
        # turned on for the browser preview (see start_live_view). Headless/no-view
        # runs uncapped (fast) for CI / sim tests.
        self._realtime = not self.headless
        self._last_step_time = None
        self._target_step_dt = 0.02  # 50 Hz = 0.02 seconds per step

        # YACT_LOG_STEP_TIME=1 -> print a rolling step-timing summary (~1 Hz):
        # step compute cost vs the loop period / effective rate.
        self._log_step_time = os.environ.get("YACT_LOG_STEP_TIME", "").lower() in ("1", "true")
        self._step_compute_ms: list = []
        self._step_period_prev: Optional[float] = None
        self._step_period_ms: list = []

        self._renderer = None
        self._render_camera = None
        self._init_body_ids()
        self._init_vis_ids()

        self._enforce_contact_props()

        logger.debug("Initialization complete")
        logger.debug("Ensure XML uses <position> actuators, not <motor>")

    def _configure_solver(self):
        """Configure solver settings for stability with position control."""
        self.mj_model.opt.iterations = 20
        self.mj_model.opt.ls_iterations = 5
        self.mj_model.opt.tolerance = 1e-6

        self.mj_model.opt.noslip_iterations = 5
        self.mj_model.opt.noslip_tolerance = 1e-6

        logger.debug("Solver configured for position control stability")

    def _enforce_contact_props(self):
        """Ensure ground plane and robot feet have correct contact parameters."""
        # 1. Ground plane
        for i in range(self.mj_model.ngeom):
            name = mujoco.mj_id2name(
                self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name in ['floor', 'ground'] or (self.mj_model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE):
                logger.debug("Setting ground props for geom '%s' (id %s)", name, i)

                self.mj_model.geom_friction[i] = [1.0, 0.005, 0.0001]
                self.mj_model.geom_condim[i] = 3
                self.mj_model.geom_contype[i] = 1
                self.mj_model.geom_conaffinity[i] = 1

        # 2. Feet friction
        feet_links = ["left_ankle_roll_link", "right_ankle_roll_link"]
        for link_name in feet_links:
            body_id = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_BODY, link_name)
            if body_id >= 0:
                start = self.mj_model.body_geomadr[body_id]
                count = self.mj_model.body_geomnum[body_id]
                for i in range(start, start + count):
                    if self.mj_model.geom_contype[i] != 0 or self.mj_model.geom_conaffinity[i] != 0:
                        self.mj_model.geom_friction[i] = [1, 0.005, 0.0001]
                        self.mj_model.geom_condim[i] = 3

    def set_friction(self, mu: float):
        """Set primary friction coefficient for ground and feet geoms."""
        for i in range(self.mj_model.ngeom):
            name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name in ['floor', 'ground'] or self.mj_model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
                self.mj_model.geom_friction[i, 0] = mu
        feet_links = ["left_ankle_roll_link", "right_ankle_roll_link"]
        for link_name in feet_links:
            body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, link_name)
            if body_id >= 0:
                start = self.mj_model.body_geomadr[body_id]
                count = self.mj_model.body_geomnum[body_id]
                for i in range(start, start + count):
                    if self.mj_model.geom_contype[i] != 0 or self.mj_model.geom_conaffinity[i] != 0:
                        self.mj_model.geom_friction[i, 0] = mu

    def _init_offscreen_renderer(self):
        """Initialize offscreen renderer."""
        with self._render_lock:
            if self._renderer is not None:
                return
            self._renderer = mujoco.Renderer(
                self.mj_model, height=720, width=1280,
            )

    def _init_mappings(self):
        """Initialize joint order mappings between MuJoCo and REAL_ORDER."""
        self.num_joints = len(REAL_ORDER)

        self.real_to_actuator = np.zeros(self.num_joints, dtype=np.int32)
        self.real_to_qpos = np.zeros(self.num_joints, dtype=np.int32)
        self.real_to_qvel = np.zeros(self.num_joints, dtype=np.int32)

        # Torque limits fallback
        TORQUE_LIMITS = {
            'hip_pitch': 88.0,
            'hip_roll': 139.0,
            'hip_yaw': 88.0,
            'knee': 139.0,
            'ankle_pitch': 25.0,
            'ankle_roll': 25.0,
            'waist_yaw': 88.0,
            'waist_roll': 25.0,
            'waist_pitch': 25.0,
            'shoulder_pitch': 25.0,
            'shoulder_roll': 25.0,
            'shoulder_yaw': 25.0,
            'elbow': 25.0,
            'wrist': 5.0,
        }

        for real_idx, joint_name in enumerate(REAL_ORDER):
            actuator_id = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
            if actuator_id < 0:
                raise ValueError(
                    f"Actuator '{joint_name}' not found in MuJoCo model")
            self.real_to_actuator[real_idx] = actuator_id

            joint_id = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(
                    f"Joint '{joint_name}' not found in MuJoCo model")

            qpos_adr = self.mj_model.jnt_qposadr[joint_id]
            qvel_adr = self.mj_model.jnt_dofadr[joint_id]
            self.real_to_qpos[real_idx] = qpos_adr
            self.real_to_qvel[real_idx] = qvel_adr

            self.mj_model.dof_armature[qvel_adr] = 0.01

        logger.debug("Joint mappings initialized: %s joints", self.num_joints)

    def _update_actuator_gains(self):
        logger.debug("Updating General Actuator parameters...")

        for real_idx in range(self.num_joints):
            kp = self.config.kps[real_idx]
            kd = self.config.kds[real_idx]

            actuator_id = self.real_to_actuator[real_idx]

            # Set Kp (The Gain)
            # For general actuator, gainprm[0] is the scaling factor
            self.mj_model.actuator_gainprm[actuator_id, 0] = kp

            # Set Bias (The Feedback)
            # We need biasprm = [Offset, Weight_Pos, Weight_Vel]
            # We want: Offset=0, Weight_Pos = -Kp, Weight_Vel = -Kd

            self.mj_model.actuator_biasprm[actuator_id, 0] = 0.0
            self.mj_model.actuator_biasprm[actuator_id, 1] = -kp
            self.mj_model.actuator_biasprm[actuator_id, 2] = -kd

            dof_adr = self.real_to_qvel[real_idx]
            self.mj_model.dof_damping[dof_adr] = 0.0

        logger.debug("Active Damping configured via General Actuators.")

    def _init_body_ids(self):
        """Initialize body IDs for key links."""
        self.body_ids = {}

        key_bodies = [
            "pelvis",
            "head_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
            "torso_link",
            "left_ankle_roll_link",
            "right_ankle_roll_link",
        ]

        for body_name in key_bodies:
            try:
                body_id = mujoco.mj_name2id(
                    self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                if body_id >= 0:
                    self.body_ids[body_name] = body_id
            except Exception:
                pass

        logger.debug("Found %s key bodies", len(self.body_ids))

    def _init_vis_ids(self):
        """Initialize mocap IDs for visualization."""
        self.vis_mocap_ids = {}
        vis_names = [
            "vis_fk_head", "vis_fk_left_wrist", "vis_fk_right_wrist", "vis_fk_heading",
            "vis_tgt_head", "vis_tgt_left_wrist", "vis_tgt_right_wrist"
        ]

        for name in vis_names:
            try:
                body_id = mujoco.mj_name2id(
                    self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
                if body_id >= 0:
                    mocap_id = self.mj_model.body_mocapid[body_id]
                    if mocap_id >= 0:
                        self.vis_mocap_ids[name] = mocap_id
            except Exception:
                pass

        logger.debug("Found %s visualization markers", len(self.vis_mocap_ids))

    def configure_single_video(self, path: str) -> None:
        """Set the output file, preserving the recording preference."""
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        self._video_output_path = path
        self.save_video_every = 0

    def start_live_view(
        self,
        port: int = 8766,
        bind: str = "0.0.0.0",
        *,
        mode: str = "wasd",
        trajectory: Optional[str] = None,
        realtime: bool = True,
    ) -> LiveViewServer:
        """Start MJPEG HTTP preview of offscreen renders.

        Caps the sim to realtime by default so the preview plays at wall-clock
        speed; pass ``realtime=False`` (``--fast``) to run uncapped.
        """
        self._realtime = realtime
        if self._live_view is None:
            self._live_view = LiveViewServer(port=port, bind=bind)
            self._live_view.configure(mode=mode, trajectory=trajectory, web_keyboard=True)
            self._live_view.start()
        return self._live_view

    @property
    def live_view_url(self) -> Optional[str]:
        if self._live_view is None:
            return None
        return self._live_view.url

    @property
    def live_view_server(self) -> Optional[LiveViewServer]:
        return self._live_view

    def pump_live_preview(self) -> None:
        """Push one offscreen frame to the live view (e.g. while waiting to start)."""
        if self._live_view is None:
            return
        if self._renderer is None:
            self._init_offscreen_renderer()
        img = self.render()
        if img is not None:
            self._live_view.update_frame(img)

    def _should_capture_frames(self) -> bool:
        return self._save_videos or self._live_view is not None

    def step(self, action: Union[np.ndarray, torch.Tensor]) -> Dict[str, np.ndarray]:
        """
        Execute an action and return the observation.

        Args:
            action: Action array/tensor of shape (num_actions,)
                   Can be numpy array or torch tensor

        Returns:
            observation: Current observation dictionary
        """
        self.act(action)
        observation = self.get_observation()
        return observation

    def act(self, action: Union[np.ndarray, torch.Tensor]):
        """Apply action (joint position targets via native MuJoCo position control)."""
        st = time.time()

        # Realtime pacing: hold each step to 50 Hz wall clock (viewer window or
        # browser preview). Headless/no-view runs uncapped.
        if self._realtime and self._last_step_time is not None:

            elapsed_since_last = st - self._last_step_time
            if elapsed_since_last < self._target_step_dt:
                # Sleep to maintain 50 Hz
                sleep_time = self._target_step_dt - elapsed_since_last
                time.sleep(sleep_time)
                st = time.time()  # Update start time after sleep

        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()

        self.action = action.copy()

        # Handle external forces
        if self._external_force_steps_remaining > 0:
            self._external_force_steps_remaining -= 1
        else:
            self._reset_external_force()

        while self.block:
            time.sleep(0.01)

        self._execute_action(action)

        spent_time = time.time() - st

        # Warn if we can't keep realtime pacing (only meaningful when capped).
        if self._realtime:
            if spent_time > self._target_step_dt:
                warnings.warn(
                    f"[G1MujocoEnv] Act took {spent_time:.4f}s > target_dt={self._target_step_dt:.4f}s (50 Hz). "
                    f"Running slower than real-time!"
                )
            self._last_step_time = time.time()

        if self._log_step_time:
            self._log_step_timing(spent_time)

        if self._obs_latency_ms > 0:
            self._obs_history.append(self._read_obs_raw())



        if self._should_capture_frames():
            if self._renderer is None:
                self._init_offscreen_renderer()
            self.save_frame_maybe()

    def _log_step_timing(self, compute_s: float) -> None:
        """Accumulate per-step timing and print a rolling summary (~1 Hz).

        ``compute`` is the cost of the sim step itself; ``period`` is the wall
        time between steps (includes the realtime throttle sleep when capped), so
        ``period`` ~= 20 ms under the realtime cap and the raw loop rate uncapped.
        """
        now = time.time()
        if self._step_period_prev is not None:
            self._step_period_ms.append((now - self._step_period_prev) * 1000.0)
        self._step_period_prev = now
        self._step_compute_ms.append(compute_s * 1000.0)

        if len(self._step_compute_ms) >= 50:
            comp = np.asarray(self._step_compute_ms)
            per = np.asarray(self._step_period_ms) if self._step_period_ms else comp
            mean_period = float(per.mean())
            hz = 1000.0 / mean_period if mean_period > 0 else 0.0
            print(
                f"[step-time] compute mean={comp.mean():.2f}ms max={comp.max():.2f}ms | "
                f"period mean={mean_period:.2f}ms ({hz:.1f} Hz) over {len(comp)} steps"
                f"{'' if self._realtime else '  [uncapped]'}",
                flush=True,
            )
            self._step_compute_ms.clear()
            self._step_period_ms.clear()

    def _read_obs_raw(self):
        """Read interpolatable sensor fields from MuJoCo state."""
        gyro_adr = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "angular-velocity")
        sensor_adr = self.mj_model.sensor_adr[gyro_adr]
        quat = self.mj_data.qpos[3:7].copy()
        return {
            'ang_vel': self.mj_data.sensordata[sensor_adr:sensor_adr+3].copy().astype(np.float32),
            'imu_quat': quat.astype(np.float32),
            'gravity_orientation': get_gravity_orientation(quat).astype(np.float32),
            'qj': self.mj_data.qpos[self.real_to_qpos].copy().astype(np.float32),
            'dqj': self.mj_data.qvel[self.real_to_qvel].copy().astype(np.float32),
        }

    def _execute_action(self, action: np.ndarray):
        """
        Execute action using native MuJoCo position control.
        """
        target_position = self.config.default_position.copy()
        target_position[self.action_mask] = action

        target_position = np.clip(
            target_position,
            self.config.joint_limits[:, 0],
            self.config.joint_limits[:, 1]
        )

        self.mj_data.ctrl[self.real_to_actuator] = target_position

        for _ in range(self.sim_decimation):
            mujoco.mj_step(self.mj_model, self.mj_data)

        self.common_step_counter += 1

        if self.viewer is not None and self.viewer.is_running():
            if "pelvis" in self.body_ids:
                pelvis_pos = self.mj_data.xpos[self.body_ids["pelvis"]]
                self.viewer.cam.lookat[:] = pelvis_pos
            self.viewer.sync()

    def get_observation(self) -> Dict[str, np.ndarray]:
        """
        Get the current observation without taking any action.

        Returns:
            observation: Current observation dictionary with:
                - ang_vel: Angular velocity from IMU (3,)
                - imu_quat: IMU quaternion w,x,y,z (4,)
                - gravity_orientation: Projected gravity vector (3,)
                - cmd: Velocity command (3,)
                - qj: Joint positions in REAL_ORDER (num_joints,)
                - dqj: Joint velocities in REAL_ORDER (num_joints,)
                - action: Last action (num_actions,)
        """
        self.qj[:] = self.mj_data.qpos[self.real_to_qpos]
        self.dqj[:] = self.mj_data.qvel[self.real_to_qvel]

        # qpos[3:7] = (w, x, y, z)
        quat = self.mj_data.qpos[3:7].copy()

        gyro_adr = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "angular-velocity")
        sensor_adr = self.mj_model.sensor_adr[gyro_adr]
        ang_vel = self.mj_data.sensordata[sensor_adr:sensor_adr+3].copy()

        gravity_orientation = get_gravity_orientation(quat)

        if self.command_source is not None:
            self.cmd[:] = self.command_source()[:]

        # Apply observation latency via lerp between previous and current step
        if self._obs_latency_ms > 0 and len(self._obs_history) == 2:
            prev, curr = self._obs_history
            a = self._obs_alpha
            # a = 1.05
            b = 1 - a
            obs_ang_vel = a * prev['ang_vel'] + b * curr['ang_vel']
            obs_quat = a * prev['imu_quat'] + b * curr['imu_quat']
            obs_quat /= np.linalg.norm(obs_quat)
            obs_grav = a * prev['gravity_orientation'] + b * curr['gravity_orientation']
            obs_qj = a * prev['qj'] + b * curr['qj']
            obs_dqj = a * prev['dqj'] + b * curr['dqj']
        else:
            obs_ang_vel = ang_vel
            obs_quat = quat
            obs_grav = gravity_orientation
            obs_qj = self.qj
            obs_dqj = self.dqj

        self.obs = {
            "ang_vel": obs_ang_vel.astype(np.float32),
            "imu_quat": obs_quat.astype(np.float32),
            "gravity_orientation": obs_grav.astype(np.float32),
            "cmd": self.cmd.copy(),
            "qj": obs_qj.copy().astype(np.float32),
            "dqj": obs_dqj.copy().astype(np.float32),
            "action": self.action.copy(),
        }
        return self.obs

    def get_head_pos_quat_w(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get head position and quaternion in world frame."""
        body_id = self.body_ids["torso_link"]

        pos = self.mj_data.xpos[body_id].copy()
        quat = self.mj_data.xquat[body_id].copy()

        from lib.vendor.mocap.constants import HEAD_OFFSET
        import lib.vendor.util.math as math_utils
        import torch

        quat_t = torch.from_numpy(quat).float().unsqueeze(0)  # (1, 4)
        offset_t = torch.tensor(HEAD_OFFSET).float().unsqueeze(0)  # (1, 3)

        rotated_offset = math_utils.quat_apply(quat_t, offset_t)

        pos += rotated_offset.numpy().flatten()

        return pos.astype(np.float32), quat.astype(np.float32)

    def apply_external_force_torque_to_link(
        self,
        link_name: str,
        force: Union[np.ndarray, torch.Tensor],
        torque: Union[np.ndarray, torch.Tensor],
        duration_steps: int = 1
    ):
        """Apply external force/torque to a link."""
        if isinstance(force, torch.Tensor):
            force = force.detach().cpu().numpy()
        if isinstance(torque, torch.Tensor):
            torque = torque.detach().cpu().numpy()

        try:
            body_id = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_BODY, link_name)
        except Exception:
            logger.warning("Body '%s' not found", link_name)
            return

        if body_id < 0:
            logger.warning("Body '%s' not found", link_name)
            return

        self.mj_data.xfrc_applied[body_id, :3] = force
        self.mj_data.xfrc_applied[body_id, 3:] = torque

        self._external_force_steps_remaining = duration_steps
        self._external_force_body_id = body_id

    def _reset_external_force(self):
        """Clear all external forces."""
        self.mj_data.xfrc_applied[:] = 0
        self._external_force_body_id = None

    def reset(self) -> Dict[str, np.ndarray]:
        """Soft reset - reset action and command but keep simulation state."""
        self.action = np.zeros_like(self.action)
        self.cmd = np.zeros_like(self.cmd)
        self.obs = None
        return self.get_observation()

    def reset_sim(self) -> Dict[str, np.ndarray]:
        """Hard reset - reset simulation to initial state."""
        mujoco.mj_resetData(self.mj_model, self.mj_data)

        self.mj_data.qpos[:3] = [0, 0, 0.8]  # Initial height
        self.mj_data.qpos[3:7] = [1, 0, 0, 0]  # Identity quaternion
        # Match the configured StartStateNode pose, not just its motor targets.
        self.mj_data.qpos[self.real_to_qpos] = self.config.default_position

        self.mj_data.qvel[:] = 0

        self.mj_data.ctrl[self.real_to_actuator] = self.config.default_position.copy(
        )

        mujoco.mj_forward(self.mj_model, self.mj_data)

        self.common_step_counter = 0
        self.action = np.zeros_like(self.action)
        self.cmd = np.zeros_like(self.cmd)
        self.obs = None
        if self._obs_latency_ms > 0:
            self._obs_history.clear()

        obs = self.get_observation()
        return obs

    def get_current_motor_state(self) -> np.ndarray:
        """Get current motor (joint) positions in REAL_ORDER."""
        result = self.mj_data.qpos[self.real_to_qpos].astype(np.float32)
        return result

    def get_qj(self) -> List[float]:
        """Get current joint positions as list."""
        return self.get_current_motor_state().tolist()

    def get_motor_temperatures(self) -> np.ndarray:
        return np.zeros(29)

    def get_env_time(self) -> float:
        """
        Get the current environment time.

        Returns simulation time based on step counter.
        """
        return self.common_step_counter * self.config.control_dt

    def go_to_initial_state(self, ask: bool = True, *, cancel_event=None):
        """Go to initial state."""
        if ask:
            logger.debug(
                "Moving to initial state: %s", self.config.default_position)
        self.go_to_state(self.config.default_position.copy(), cancel_event=cancel_event)
        logger.debug("Moving to initial state done")

    def go_to_state(self, target_positions: np.ndarray, time_to_move: float = 4.0, *, cancel_event=None):
        """Smoothly move to target joint positions."""
        import tqdm

        current_positions = self.get_current_motor_state()
        n_steps = int(time_to_move / self.config.control_dt)
        tgt_linspace = np.linspace(
            current_positions, target_positions, n_steps)

        for cur_tgt in tqdm.tqdm(tgt_linspace, desc="Moving to state"):
            if cancel_event is not None and cancel_event.is_set():
                break
            self._set_motor_target_positions(cur_tgt)

    def _set_motor_target_positions(self, target_positions: np.ndarray):
        """Set motor commands for all joints."""
        target_positions_clipped = np.clip(
            target_positions,
            self.config.joint_limits[:, 0],
            self.config.joint_limits[:, 1]
        )

        action = target_positions_clipped[self.action_mask]
        self.step(action)

    def render(self) -> Optional[np.ndarray]:
        """Render and return image using offscreen renderer."""
        with self._render_lock:
            if self._renderer is None:
                return None

            try:
                if self._render_camera is None:
                    self._render_camera = mujoco.MjvCamera()
                    self._render_camera.distance = 3.46
                    self._render_camera.elevation = -35.3
                    self._render_camera.azimuth = -90

                if "pelvis" in self.body_ids:
                    pelvis_pos = self.mj_data.xpos[self.body_ids["pelvis"]]
                    self._render_camera.lookat[0] = pelvis_pos[0]
                    self._render_camera.lookat[1] = pelvis_pos[1]
                    self._render_camera.lookat[2] = pelvis_pos[2]

                self._renderer.update_scene(
                    self.mj_data, camera=self._render_camera)

                return self._renderer.render()
            except Exception as e:
                now = time.monotonic()
                if now - self._last_render_error_log_mono >= 2.0:
                    self._last_render_error_log_mono = now
                    logger.warning("Render error: %s", e)
                return None

    def save_frame_maybe(self):
        """Save frame to buffer for video recording."""
        if self.last_frame_step == self.common_step_counter:
            return

        self.block = True
        self.last_frame_step = self.common_step_counter

        img = self.render()
        if img is not None:
            if self._save_videos:
                self.img_buf.append(img)
            if self._live_view is not None:
                self._live_view.update_frame(img)

        if self.save_video_every and self.save_video_every > 0 and self.common_step_counter % self.save_video_every == 0:
            self.save_video()

        self.block = False

    def save_video(self, path: Optional[str] = None, *, announce: bool = False):
        """Save buffered frames as video."""
        if self._video_saved or not self.img_buf:
            return

        try:
            from lib.vendor.util.compression import np2mp4bytes

            if path is None:
                path = self._video_output_path or (
                    f"mujoco_env_{self._init_timestamp_s}_{self.common_step_counter}.mp4"
                )

            logger.debug("Saving video to %s", path)
            if announce:
                print("Saving sim video…", flush=True)
            with open(path, 'wb') as f:
                f.write(np2mp4bytes(list(self.img_buf), fps=50))
            self._video_saved = True
            if announce:
                print_path_lines("Sim video saved to ", path)
        except Exception as e:
            logger.warning("Failed to save video: %s", e)

    def update_fk_vis(self, robot_model: RobotModel):
        """Update forward kinematics visualization."""
        if not hasattr(self, 'vis_mocap_ids') or not self.vis_mocap_ids:
            return

        keypoint_pos_w, keypoint_quat_w = robot_model.get_keypoint_pos_quat_w()
        heading_pos_w, heading_quat_w = robot_model.get_heading_pos_quat_w()

        def to_numpy(x):
            if hasattr(x, "detach"):
                return x.detach().cpu().numpy()
            return np.array(x)

        kp_pos = to_numpy(keypoint_pos_w).reshape(-1, 3)
        kp_quat = to_numpy(keypoint_quat_w).reshape(-1, 4)

        mapping = [
            ("vis_fk_head", 0),
            ("vis_fk_left_wrist", 1),
            ("vis_fk_right_wrist", 2)
        ]

        for name, idx in mapping:
            if name in self.vis_mocap_ids:
                mid = self.vis_mocap_ids[name]
                if idx < len(kp_pos):
                    self.mj_data.mocap_pos[mid] = kp_pos[idx]
                    self.mj_data.mocap_quat[mid] = kp_quat[idx]

        if "vis_fk_heading" in self.vis_mocap_ids:
            mid = self.vis_mocap_ids["vis_fk_heading"]
            self.mj_data.mocap_pos[mid] = to_numpy(heading_pos_w).flatten()
            self.mj_data.mocap_quat[mid] = to_numpy(heading_quat_w).flatten()

    def update_tgt_vis(self, vr_heading_observator: VrHeadingObservator):
        """Update target visualization."""
        if not hasattr(self, 'vis_mocap_ids') or not self.vis_mocap_ids:
            return

        tgt_pose = vr_heading_observator.get_observation().head_wrists_pose_w

        def to_numpy(x):
            if hasattr(x, "detach"):
                return x.detach().cpu().numpy()
            return np.array(x)

        pos = to_numpy(tgt_pose.pos).reshape(-1, 3)
        quat = to_numpy(tgt_pose.quat).reshape(-1, 4)

        mapping = [
            ("vis_tgt_head", 0),
            ("vis_tgt_left_wrist", 1),
            ("vis_tgt_right_wrist", 2)
        ]

        for name, idx in mapping:
            if name in self.vis_mocap_ids:
                mid = self.vis_mocap_ids[name]
                if idx < len(pos):
                    self.mj_data.mocap_pos[mid] = pos[idx]
                    self.mj_data.mocap_quat[mid] = quat[idx]

    def zero_torque_mode(self):
        """Set robot to zero torque mode."""
        current_pos = self.mj_data.qpos[self.real_to_qpos]
        self.mj_data.ctrl[self.real_to_actuator] = current_pos

    def damping_mode(self):
        """Set robot to damping mode."""
        current_pos = self.mj_data.qpos[self.real_to_qpos]
        self.mj_data.ctrl[self.real_to_actuator] = current_pos

    def get_remote_controller(self):
        """Get the remote controller object (not available in simulation)."""
        return None

    @property
    def disconnected(self) -> bool:
        """Check if robot is disconnected (always False in simulation)."""
        return False

    @property
    def state(self):
        """Get the current robot state (not applicable in MuJoCo)."""
        return None

    @property
    def command(self):
        """Get the current command (not applicable in MuJoCo)."""
        return None

    @property
    def sim_env(self):
        """For compatibility with code that accesses sim_env attribute."""
        return self

    @property
    def device(self) -> str:
        """Device for tensor operations (CPU for MuJoCo)."""
        return "cpu"

    def close(self):
        """Clean up resources."""
        if self._closed:
            return
        self._closed = True

        if self._save_videos:
            self.save_video(announce=True)

        if self._live_view is not None:
            self._live_view.stop()
            self._live_view = None

        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception as e:
                logger.warning("Failed to close viewer: %s", e)
            self.viewer = None

        if self._renderer is not None:
            with self._render_lock:
                try:
                    self._renderer.close()
                except Exception as e:
                    logger.warning("Failed to close renderer: %s", e)
                self._renderer = None

    def __del__(self):
        if not self._closed:
            self.close()
