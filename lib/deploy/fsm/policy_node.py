import threading
from typing import Optional
from lib.deploy.deploy_utils.networks import MockNetworkType, is_mock_network, is_sync_sim_network
from lib.vendor.mocap.constants import HeadingFrameSource
import torch
from lib.deploy.vr.localization import LocalizationObservator
from lib.deploy.vr.vr_observations import VrHeadingObservator

from lib.deploy.fsm.common import MoveToEdgeState, Node, DisabledState, ActiveState
from lib.deploy.control.sleeper import create_sleeper_from_g1_env
from lib.deploy.control.base_env import BaseEnv
from lib.deploy.control.config import AudioAlertKey, MotorTemperatureCfg
from lib.deploy.configs.joint_mappings import REAL_ORDER
from lib.deploy.debug.logs import DeployConsole, emit_sleep_warnings
from lib.deploy.fsm.damp_node import DampNode
from lib.deploy.vr.vr_replay_wrapper import VRReplayWrapper
from lib.deploy.debug.profiler import Profiler


def _hot_motors_str(motor_temperatures, threshold: float) -> str:
    return ", ".join(
        f"{name}({t:.1f})"
        for name, t in zip(REAL_ORDER, motor_temperatures)
        if t > threshold
    )

class PolicyNode(Node):
    def __init__(
        self,
        env: BaseEnv,
        policy,
        device: str,
        vr_observator: VrHeadingObservator,
        localization_observator: LocalizationObservator,
        env_processor,
        debug_vis: bool = False,
        profiler: Optional[Profiler] = None,
        name: Optional[str] = None,
        motor_temp_cfg: Optional[MotorTemperatureCfg] = None,
        *,
        console: Optional[DeployConsole] = None,
    ):
        super().__init__()
        self.env = env
        display_name = name or self.__class__.__name__
        self.console = (
            console
            if console is not None
            else DeployConsole(display_name)
        )
        self.sleeper = create_sleeper_from_g1_env(
            env,
            enable_logging=False,
        )
        self.profiler = profiler if profiler is not None else Profiler(
            enabled=False)
        self.policy = policy
        self.device = device
        self.vr_observator = vr_observator
        self.localization_observator = localization_observator
        self.debug_vis = debug_vis
        self._name = name
        self._thread = None
        self._stop_event = threading.Event()
        self.motor_temp_cfg = motor_temp_cfg
        self._last_motor_temp_warning_timestamp = 0
        self.processor = env_processor

        self.raw_act_buffer = None
        self._warmup()

    @torch.no_grad()
    @torch.inference_mode()
    def _warmup(self):
        carries = self.processor.init_carries()
        obs, _, _ = self._get_observation(carries, 0)
        for _ in range(4):
            self.policy(obs)
        self._reset()

    def activate(self):
        """Start the damping thread"""
        self.console.info(f"Activating {self.name}")
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._find_damp_node()
            self._thread = threading.Thread(
                target=self._node_loop, daemon=True)
            self._thread.start()
            self.state = ActiveState()

    def deactivate(self):
        """Signal the thread to stop and wait for it to finish"""
        self.console.info(f"Deactivating {self.name}")
        if self._thread is not None and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join()
        self.state = DisabledState()

    def _trigger_damp(self, cause: str, message: str):
        self.console.critical(message)
        self.state = MoveToEdgeState(
            payload=self._find_damp_node(),
            cause=f"[{self.__class__.__name__}] {cause}"
        )

    def _get_observation(self, carries, timestep: int) -> torch.Tensor:
        obs = self.env.get_observation()
        action_carry, observation_carry = carries
        proc_obs, observation_carry = self.processor.process_obs(
            obs, timestep, action_carry, observation_carry)
        return proc_obs, obs, (action_carry, observation_carry)

    def _get_action(self, obs: torch.Tensor, carries):
        with torch.no_grad() and torch.inference_mode():
            if self.raw_act_buffer is None:
                self.raw_act_buffer = self.policy(obs)
            else:
                self.raw_act_buffer[:] = self.policy(obs)

        action_carry, observation_carry = carries
        act, action_carry = self.processor.process_action(
            self.raw_act_buffer, action_carry, observation_carry)
        return act, (action_carry, observation_carry)

    def _reset(self):
        self.env.reset()
        self.processor.reset()

    @property
    def name(self):
        if self._name is not None:
            return self._name
        else:
            return super().name

    def _find_damp_node(self) -> int:
        for i, e in enumerate(self.edges):
            if isinstance(e.target, DampNode):
                return i
        raise ValueError(
            "Damping node not found! Connect this node with DampNode")

    def _node_loop(self):
        if is_sync_sim_network(self.env.network):
            with torch.inference_mode():
                if not hasattr(self.env, 'stands'):
                    self.env.reset_sim()
                    self.env.stands = True
        self.localization_observator.step()
        self.vr_observator.reset_head_offset()
        if isinstance(self.vr_observator.tracker.base_tracker, VRReplayWrapper):
            self.vr_observator.tracker.base_tracker.current_index = 0
        with torch.no_grad():
            with torch.inference_mode():
                self._reset()
                carries = self.processor.init_carries()

                self.localization_observator.step()
                self.vr_observator.reset_head_offset()
                self.sleeper.reset_timing()
                timestep = 0
                observation, raw_obs, carries = self._get_observation(
                    carries, timestep=timestep)

                while not self._stop_event.is_set():

                    if self.env.disconnected:
                        self._trigger_damp(
                            cause="Robot disconnected",
                            message="Robot disconnected. Switching to damp mode."
                        )
                    if is_sync_sim_network(self.env.network):
                        self.env.update_fk_vis(self.processor.robot_model)
                        self.env.update_tgt_vis(self.vr_observator)
                    if self.motor_temp_cfg is not None:
                        motor_temperatures = self.env.get_motor_temperatures()
                        max_temp = max(motor_temperatures)
                        cfg = self.motor_temp_cfg
                        if max_temp > cfg.warning_threshold and (timestep - self._last_motor_temp_warning_timestamp) >= 1:
                            hot_motors = _hot_motors_str(motor_temperatures, cfg.warning_threshold)
                            self._last_motor_temp_warning_timestamp = timestep
                            self.console.warning(
                                f"High motor temperatures ({max_temp:.1f}): {hot_motors}",
                                alert_id=AudioAlertKey.MOTOR_TEMP_WARNING,
                            )
                        if max_temp > cfg.critical_threshold:
                            hot_motors = _hot_motors_str(motor_temperatures, cfg.critical_threshold)
                            self._trigger_damp(
                                cause="Motor temperatures critical",
                                message=f"Motor temperatures critical ({max_temp:.1f}), switching to damp mode: {hot_motors}"
                            )

                    with self.profiler.time("Node Step"):

                        with self.profiler.time("Policy step"):
                            action, carries = self._get_action(
                                observation, carries)

                        with self.profiler.time("Env step"):
                            self.env.act(action)

                        with self.profiler.time("Sleep"):
                            if not is_mock_network(self.env.network):
                                time_slept = self.sleeper.sleep()
                                emit_sleep_warnings(
                                    self.console,
                                    time_slept,
                                    self.sleeper.last_oversleep_time,
                                )
                                if time_slept < -0.3:
                                    self._trigger_damp(
                                        cause="Massive overtime detected",
                                        message="Massive overtime detected. Switching to DampNode."
                                    )

                        timestep += 1

                        with self.profiler.time("VR Tracker step"):
                            if isinstance(self.vr_observator.tracker.base_tracker, VRReplayWrapper):
                                self.vr_observator.tracker.base_tracker.step()
                        with self.profiler.time("Localization Observator step"):
                            self.localization_observator.step()

                        with self.profiler.time("Get observation step"):
                            observation, raw_obs, carries = self._get_observation(
                                carries, timestep=timestep)
