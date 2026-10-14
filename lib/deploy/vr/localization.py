import datetime
import logging
import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Union

import numpy as np
import torch
from pydantic import BaseModel
from lib.deploy.deploy_utils.forward_kinematics import RobotModel
from lib.deploy.vr.common import Pose
from scipy.spatial.transform import Rotation as R

try:
    from cyclonedds.idl import IdlStruct
    from cyclonedds.idl.types import float32, float64
    from unitree_sdk2py.core.channel import ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_

    @dataclass
    class OdometryData(IdlStruct):
        x: float32
        y: float32
        z: float32
        q_w: float32
        q_x: float32
        q_y: float32
        q_z: float32
        stamp: float64

except:
    print("No unitree libs found!")


class LocalizationObs(BaseModel):
    pose: Pose


class LocalizationObservator:
    def get_observation(self) -> LocalizationObs:
        raise NotImplementedError


class LocalizationObservatorWithFK(LocalizationObservator):
    robot_model: RobotModel


class SteppableLocalizationObservator(LocalizationObservator):
    def step(self) -> None:
        raise NotImplementedError


class SimpleLocalizationObservatorWithFK(LocalizationObservatorWithFK):
    def __init__(self,
                 base_observator: LocalizationObservator,
                 robot_model: RobotModel,
                 joint_state_getter: Callable[[], Union[torch.Tensor, Dict[str, float]]],
                 ):
        self.base_observator = base_observator
        self.robot_model = robot_model
        self.joint_state_getter = joint_state_getter

    def get_observation(self):
        obs = self.base_observator.get_observation()

        pos = obs.pose.pos.copy()
        quat = obs.pose.quat.copy()

        joint_positions = self.joint_state_getter()

        head_pos_t = torch.tensor(
            pos, device=self.robot_model.device, dtype=torch.float32)
        head_quat_t = torch.tensor(
            quat, device=self.robot_model.device, dtype=torch.float32)

        self.robot_model.recompute_forward_kinematics(
            joint_positions=joint_positions,
            head_quat_w=head_quat_t,
            head_pos_w=head_pos_t
        )

        return LocalizationObs(pose=Pose(pos=pos, quat=quat))


# ----------------------------------------------------------------------------
# FK Height Wrapper
# ----------------------------------------------------------------------------

class FKHeightLocalizationWrapper(LocalizationObservator):
    """
    Wrapper that computes z-coordinate using forward kinematics.
    Wraps any LocalizationObservator and replaces the z-coordinate with FK-computed height.
    """

    def __init__(
        self,
        base_observator: LocalizationObservator,
        joint_state_getter: Callable[[], Union[torch.Tensor, Dict[str, float]]],
        robot_model_config: Optional['RobotModelConfig'] = None,
        relative_to_ground: bool = True,
        fallback_on_error: bool = False
    ):
        """
        Args:
            base_observator: The underlying localization observator to wrap.
            joint_state_getter: Callback that returns current joint positions.
            robot_model_config: Optional config for the robot model.
            relative_to_ground: If True, compute height relative to ground (lowest point).
            fallback_on_error: If True, use original z on FK computation failure.
        """
        self.base_observator = base_observator
        self.joint_state_getter = joint_state_getter
        self.relative_to_ground = relative_to_ground
        self.fallback_on_error = fallback_on_error

        from lib.deploy.deploy_utils.forward_kinematics import (
            RobotModel, RobotModelConfig)
        self.robot_model = RobotModel(robot_model_config or RobotModelConfig())

    def _compute_fk_height(self, head_pos: np.ndarray, head_quat: np.ndarray) -> float:
        """Compute z-coordinate using forward kinematics."""
        joint_positions = self.joint_state_getter()

        head_pos_t = torch.tensor(
            head_pos, device=self.robot_model.device, dtype=torch.float32)
        head_quat_t = torch.tensor(
            head_quat, device=self.robot_model.device, dtype=torch.float32)

        self.robot_model.recompute_forward_kinematics(
            joint_positions=joint_positions,
            head_quat_w=head_quat_t,
            head_pos_w=head_pos_t
        )

        fk_height = self.robot_model.get_heading_frame_height(
            relative_to_ground=self.relative_to_ground)
        return fk_height.item()

    def get_observation(self) -> LocalizationObs:
        obs = self.base_observator.get_observation()

        pos = obs.pose.pos.copy()
        quat = obs.pose.quat

        try:
            fk_height = self._compute_fk_height(pos, quat)
            pos[2] = fk_height
        except Exception as e:
            if not self.fallback_on_error:
                raise
            print(
                f"[FKHeightLocalizationWrapper] FK height computation failed: {e}")

        return LocalizationObs(pose=Pose(pos=pos, quat=quat))


class FKHeightLocalizationWrapperV2(LocalizationObservatorWithFK):
    """
    Wrapper that computes z-coordinate using forward kinematics.
    Wraps any LocalizationObservator and replaces the z-coordinate with FK-computed height.
    """

    def __init__(
        self,
        base_observator: LocalizationObservatorWithFK,
        relative_to_ground: bool = True,
    ):
        """
        Args:
            base_observator: The underlying localization observator to wrap.
            robot_model_config: Optional config for the robot model.
            relative_to_ground: If True, compute height relative to ground (lowest point).
        """
        self.base_observator = base_observator
        self.relative_to_ground = relative_to_ground

        self.robot_model = base_observator.robot_model

    def _compute_fk_height(self) -> float:
        """Compute z-coordinate using forward kinematics."""
        fk_height = self.robot_model.get_heading_frame_height(
            relative_to_ground=self.relative_to_ground)
        return fk_height.item()

    def get_observation(self) -> LocalizationObs:
        obs = self.base_observator.get_observation()

        pos = obs.pose.pos.copy()
        quat = obs.pose.quat

        fk_height = self._compute_fk_height()
        pos[2] = fk_height

        return LocalizationObs(pose=Pose(pos=pos, quat=quat))


# ----------------------------------------------------------------------------
# Factory Functions
# ----------------------------------------------------------------------------

def create_steppable_localization_observator(base_observator: LocalizationObservator):
    return SteppableLocalizationObservatorImpl(base_observator=base_observator)


def create_noisy_localization_observator(base_observator: LocalizationObservator, pos_noise_m: float = 0.0, quat_noise_deg: float = 0.0):
    return NoisyLocalizationObservatorImpl(base_observator=base_observator, pos_noise_m=pos_noise_m, quat_noise_deg=quat_noise_deg)


def create_delayed_sim_localization_observator(
    env,
    latency_range_ms: Tuple[int, int],
    decimation_range: Tuple[int, int],
    use_fk_height: bool = False,
    robot_model_config: Optional['RobotModelConfig'] = None,
    relative_to_ground: bool = True,
    env_index: int = 0,
    joint_state_getter=None,
):
    base_observator = DelayedSimLocalizationObservatorImpl(
        env,
        latency_range_ms=latency_range_ms,
        decimation_range=decimation_range
    )

    if use_fk_height:
        return FKHeightLocalizationWrapper(
            base_observator=base_observator,
            joint_state_getter=(lambda: env.scene['robot'].data.joint_pos[env_index]
                                ) if joint_state_getter is None else joint_state_getter,
            robot_model_config=robot_model_config,
            relative_to_ground=relative_to_ground
        )

    return base_observator


def create_simple_localization_observator_with_fk(
    base_observator: LocalizationObservator,
    robot_model: RobotModel,
    joint_state_getter: Callable[[], Union[torch.Tensor, Dict[str, float]]],
):
    return SimpleLocalizationObservatorWithFK(
        base_observator=base_observator,
        robot_model=robot_model,
        joint_state_getter=joint_state_getter
    )


def create_delayed_sim_localization_observator_v2(
    env,
    latency_range_ms: Tuple[int, int],
    decimation_range: Tuple[int, int],
    robot_model: RobotModel,
    joint_state_getter: Callable[[], Union[torch.Tensor, Dict[str, float]]],
    use_fk_height: bool = False,
    relative_to_ground: bool = True,
    env_index: int = 0,
):
    base_observator = DelayedSimLocalizationObservatorImpl(
        env,
        latency_range_ms=latency_range_ms,
        decimation_range=decimation_range
    )

    observator_with_fk = create_simple_localization_observator_with_fk(
        base_observator=base_observator,
        robot_model=robot_model,
        joint_state_getter=joint_state_getter
    )

    if use_fk_height:
        return FKHeightLocalizationWrapperV2(
            base_observator=observator_with_fk,
            relative_to_ground=relative_to_ground
        )

    return observator_with_fk


def create_localization_observator(
    log_level: Optional[int] = None,
    log_dir: str = "./logs",
    log_file_prefix: str = "localization",
    use_fk_height: bool = False,
    joint_state_getter: Optional[Callable[[],
                                          Union[torch.Tensor, Dict[str, float]]]] = None,
    robot_model_config: Optional['RobotModelConfig'] = None,
    relative_to_ground: bool = True
) -> LocalizationObservator:
    base_observator = LocalizationObservatorImpl(
        log_level=log_level,
        log_dir=log_dir,
        log_file_prefix=log_file_prefix
    )

    if use_fk_height:
        if joint_state_getter is None:
            raise ValueError(
                "joint_state_getter must be provided when use_fk_height=True")
        return FKHeightLocalizationWrapper(
            base_observator=base_observator,
            joint_state_getter=joint_state_getter,
            robot_model_config=robot_model_config,
            relative_to_ground=relative_to_ground
        )

    return base_observator


def create_localization_observator_v2(
    joint_state_getter: Callable[[], Union[torch.Tensor, Dict[str, float]]],
    robot_model: RobotModel,
    log_level: Optional[int] = None,
    log_dir: str = "./logs",
    log_file_prefix: str = "localization",
    use_fk_height: bool = False,
    relative_to_ground: bool = True
) -> LocalizationObservator:
    base_observator = LocalizationObservatorImpl(
        log_level=log_level,
        log_dir=log_dir,
        log_file_prefix=log_file_prefix
    )

    observator_with_fk = create_simple_localization_observator_with_fk(
        base_observator=base_observator,
        robot_model=robot_model,
        joint_state_getter=joint_state_getter
    )

    if use_fk_height:
        return FKHeightLocalizationWrapperV2(
            base_observator=observator_with_fk,
            relative_to_ground=relative_to_ground
        )

    return base_observator


def create_sim_localization_observator(
    env,
    use_fk_height: bool = False,
    robot_model_config: Optional['RobotModelConfig'] = None,
    relative_to_ground: bool = True,
    env_index: int = 0
) -> LocalizationObservator:
    base_observator = LocalizationSimObservatorImpl(env)

    if use_fk_height:
        return FKHeightLocalizationWrapper(
            base_observator=base_observator,
            joint_state_getter=lambda: (env.robot.data.joint_pos[env_index] if hasattr(
                env, 'robot') else env.scene['robot'].data.joint_pos[env_index]),
            robot_model_config=robot_model_config,
            relative_to_ground=relative_to_ground
        )

    return base_observator


def create_sim_dds_localization_observator() -> LocalizationObservator:
    return LocalizationSimDDSObservatorImpl()


# ----------------------------------------------------------------------------
# Original Implementations (unchanged)
# ----------------------------------------------------------------------------

class SteppableLocalizationObservatorImpl(SteppableLocalizationObservator):
    def __init__(self, base_observator: LocalizationObservator):
        self.base_observator = base_observator
        self.data = self.base_observator.get_observation()

    def step(self):
        self.data = self.base_observator.get_observation()

    def get_observation(self):
        return self.data


LATEST_IMU = None


def SecondaryIMUHandler(msg: "IMUState_"):
    global LATEST_IMU
    LATEST_IMU = msg


class LocalizationObservatorImpl(LocalizationObservator):
    def __init__(self, log_level: Optional[int] = logging.INFO, log_dir: str = "./logs",
                 log_file_prefix: str = "localization", zmq_port=5555):
        if log_level is not None:
            os.makedirs(log_dir, exist_ok=True)

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_file = os.path.join(
                log_dir, f"{log_file_prefix}_{timestamp}.log")

            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.setLevel(log_level)

            if log_level is not None:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(logging.Formatter(
                    "[%(asctime)s.%(msecs)03d] [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S"
                ))

                file_handler = logging.FileHandler(log_file, mode='a')
                file_handler.setFormatter(logging.Formatter(
                    "[%(asctime)s.%(msecs)03d] [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S"
                ))
                self.logger.addHandler(console_handler)
                self.logger.addHandler(file_handler)

                self.logger.info(f"Logging to file: {log_file}")
        else:
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.setLevel(logging.WARNING)

        self.imu_sub = ChannelSubscriber("rt/secondary_imu", IMUState_)
        self.imu_sub.Init(self.SecondaryIMUHandler, 10)
        self.imu_torso_state = None

        self.odom_sub = ChannelSubscriber("lidar/raw_odometry", OdometryData)
        self.odom_sub.Init(self.OdometryHandler, 10)
        self.odom_state = None

    def SecondaryIMUHandler(self, msg: "IMUState_"):
        self.imu_torso_state = msg

    def OdometryHandler(self, msg: "OdometryData"):
        self.odom_state = msg

    def debug_log(self, *args, **kwargs):
        if self.logger is not None:
            self.logger.debug(*args, **kwargs)

    def warning_log(self, *args, **kwargs):
        if self.logger is not None:
            self.logger.warning(*args, **kwargs)

    def error_log(self, *args, **kwargs):
        if self.logger is not None:
            self.logger.error(*args, **kwargs)

    def get_observation(self) -> LocalizationObs:
        if self.odom_state is None or self.imu_torso_state is None:
            self.warning_log("No DDS odometry message received yet.")
            return LocalizationObs(pose=Pose(pos=np.array([0., 0., 0.]), quat=np.array([1., 0., 0., 0.])))
        from scipy.spatial.transform import Rotation as R

        # y and z are inverted relative to head frame therefore we need to use them with negative sign
        # pos = np.array([self.msg['x'], -self.msg['y'], -self.msg['z']], dtype=np.float32)
        # quat = np.array([self.msg['q_w'], self.msg['q_x'], self.msg['q_y'], self.msg['q_z']], dtype=np.float32)
        pos = np.array([self.odom_state.x, -self.odom_state.y, -
                       self.odom_state.z], dtype=np.float32)
        quat = np.array([self.odom_state.q_w, self.odom_state.q_x,
                        self.odom_state.q_y, self.odom_state.q_z], dtype=np.float32)
        r = R.from_quat([1.0, 0.0, 0.0, 0.0])
        q = R.from_quat(quat[[1, 2, 3, 0]])
        quat = (r * q * r).as_quat()[[3, 0, 1, 2]]

        self.debug_log(f"Generated observation: pos={pos}, quat={quat}")

        MOCK_POS = False
        MOCK_QUAT = False

        dds_quat = self.imu_torso_state.quaternion
        # print('quat', quat, 'dds_quat', dds_quat)
        # quat = dds_quat
        return LocalizationObs(
            pose=Pose(pos=np.array([0., 0., 0.]) if MOCK_POS else pos, quat=np.array(
                [0.8102, 0.0071, 0.0195, -0.5858]) if MOCK_QUAT else quat)
        )

    def __del__(self):
        self._running = False


class LocalizationSimObservatorImpl(LocalizationObservator):
    def __init__(self, env):
        self.env = env

    def get_observation(self) -> LocalizationObs:
        pos, quat = self.env.get_head_pos_quat_w()
        return LocalizationObs(
            pose=Pose(pos=pos.astype(np.float32),
                      quat=quat.astype(np.float32))
        )


class NoisyLocalizationObservatorImpl(LocalizationObservator):
    def __init__(self, base_observator: LocalizationObservator, pos_noise_m: float = 0.0, quat_noise_deg: float = 0.0):
        self.base_observator = base_observator
        self.pos_noise_m = pos_noise_m
        self.quat_noise_deg = quat_noise_deg

    def get_observation(self) -> LocalizationObs:
        base_obs = self.base_observator.get_observation()

        pos = base_obs.pose.pos.copy()
        quat = base_obs.pose.quat.copy()

        if self.pos_noise_m > 0:
            noise_pos = np.random.uniform(
                low=-self.pos_noise_m, high=self.pos_noise_m, size=3)
            pos += noise_pos

        if self.quat_noise_deg > 0 and np.linalg.norm(quat) > 1e-6:
            sigma_rad = np.deg2rad(self.quat_noise_deg)

            noise_rot_vec = np.random.uniform(
                low=-sigma_rad, high=sigma_rad, size=3)

            rot_noise = R.from_rotvec(noise_rot_vec)
            rot_current = R.from_quat(quat[[1, 2, 3, 0]])

            rot_noisy = rot_current * rot_noise

            quat = rot_noisy.as_quat()[[3, 0, 1, 2]]

        return LocalizationObs(
            pose=Pose(
                pos=pos.astype(np.float32),
                quat=quat.astype(np.float32)
            )
        )


class DelayedSimLocalizationObservatorImpl(LocalizationObservator):
    def __init__(self, env, latency_range_ms: Tuple[int, int], decimation_range: Tuple[int, int]):
        from lib.deploy.vr.delayed_observation import (
            WithDiscretization, WithLatency)
        self.env = env
        self.latency_range_ms = latency_range_ms
        self.decimation_range = decimation_range

        def get_head_pose(env):
            if hasattr(env, 'get_head_pos_quat_w'):
                return env.get_head_pos_quat_w()
            return (
                torch.zeros((1, 3), device='cuda:0'),
                torch.tensor([[0.0, 0.0, 0.0, 1.0]], device='cuda:0'),
            )

        self.term = WithLatency(
            WithDiscretization(
                get_head_pose,
                self.decimation_range
            ),
            self.latency_range_ms
        )

    def get_observation(self) -> LocalizationObs:
        pos, quat = self.term(self.env)
        if isinstance(pos, torch.Tensor):
            pos = pos.cpu().numpy()
            quat = quat.cpu().numpy()
        pos = pos.reshape(3).astype(np.float32)
        quat = quat.reshape(4).astype(np.float32)
        return LocalizationObs(
            pose=Pose(pos=pos, quat=quat)
        )


class LocalizationSimDDSObservatorImpl(LocalizationObservator):
    def __init__(self):
        from unitree_sdk2py.core.channel import ChannelSubscriber
        from unitree_sdk2py.idl.nav_msgs.msg.dds_ import Odometry_

        self.latest_odom = None
        self.subscriber = ChannelSubscriber("rt/odommodestate", Odometry_)
        self.subscriber.Init(self._odom_callback, 1)
        print(
            "[LocalizationSimObservator] Initialized DDS subscriber for rt/odommodestate")

    def _odom_callback(self, msg: 'Odometry_') -> None:
        """Callback to store the latest odometry message"""
        self.latest_odom = msg

    def get_observation(self) -> LocalizationObs:
        if self.latest_odom is None:
            print("[LocalizationSimObservator] Warning: No odometry data received yet")
            return LocalizationObs(
                pose=Pose(
                    pos=np.zeros(3, dtype=np.float32),
                    quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                )
            )

        pos = np.array([
            self.latest_odom.pose.pose.position.x,
            -self.latest_odom.pose.pose.position.y,
            -self.latest_odom.pose.pose.position.z
        ], dtype=np.float32)

        quat = np.array([
            self.latest_odom.pose.pose.orientation.w,
            self.latest_odom.pose.pose.orientation.x,
            self.latest_odom.pose.pose.orientation.y,
            self.latest_odom.pose.pose.orientation.z
        ], dtype=np.float32)

        return LocalizationObs(
            pose=Pose(pos=pos, quat=quat)
        )

    def __del__(self):
        if hasattr(self, 'subscriber'):
            pass
