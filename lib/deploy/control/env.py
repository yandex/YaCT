import time
from typing import Dict, Union, Callable, Tuple, Optional

import numpy as np
import torch
import tqdm

from lib.deploy.control.config import Config
from lib.deploy.control.controller import Controller
from lib.deploy.control.util import get_gravity_orientation
from lib.deploy.control.base_env import BaseEnv


def quaternion_conjugate(q):
    """Return conjugate of quaternion [w, x, y, z]"""
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quaternion_multiply(q1, q2):
    """Multiply two quaternions [w, x, y, z]"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return np.array([w, x, y, z])

def rotate_vector_by_quaternion(q, v):
    """Rotate vector v by quaternion q"""
    # Convert vector to quaternion form [0, vx, vy, vz]
    v_quat = np.array([0, v[0], v[1], v[2]])

    # Perform rotation: q * v * q^(-1)
    q_conj = quaternion_conjugate(q)
    result = quaternion_multiply(quaternion_multiply(q, v_quat), q_conj)

    # Extract vector part
    return result[1:4]


class G1Env(BaseEnv):
    """
    Environment wrapper for Unitree robot that accepts numpy arrays or torch tensors as actions.

    This environment provides a simple interface for controlling the Unitree robot:
    - step(action): Execute action and return observation
    - reset(): Reset environment and return initial observation
    - get_observation(): Get current observation without taking action
    """

    def __init__(
            self,
            config: Config,
            command_source: Optional[Callable[[], Tuple[float, float, float]]] = None,
            network: Optional[str] = None,
            max_steps: int = 10 ** 10
        ):
        """
        Initialize the environment.

        Args:
            config: Configuration object for the robot
            max_steps: Maximum number of steps per episode
        """
        self.config = config
        self.command_source = command_source
        self.action_mask = self.config.action_mask
        self.max_steps = max_steps

        # Initialize observation-related variables
        self.qj = np.zeros(config.num_joints, dtype=np.float32)
        self.dqj = np.zeros(config.num_joints, dtype=np.float32)
        self.action = np.zeros(config.action_mask.sum(), dtype=np.float32)
        self.obs = None
        self.cmd = np.array([0.0, 0.0, 0.0])

        self.controller = Controller(config, network=network)
        self.network = network
        print('init sequence done!')


    def step(self, action: Union[np.ndarray, torch.Tensor]) -> Dict[str, np.ndarray]:
        """
        Execute an action and return the observation.

        Args:
            action: Action array/tensor of shape (num_actions,)
                   Can be numpy array or torch tensor

        Returns:
            observation: Current observation array
        """
        st = time.time()

        self.act(action)

        # Get observation
        observation = self.get_observation()

        spent_time = time.time() - st
        time.sleep(max(0, self.config.control_dt - spent_time))
        if spent_time > self.config.control_dt:
            print(f"WARNING: {spent_time=}")

        return observation

    def act(self, action):
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        self.action = action.copy()
        self._execute_action(action)

    def reset(self) -> Dict[str, np.ndarray]:
        self.action = np.zeros_like(self.action)
        self.obs = None
        self.cmd = np.zeros_like(self.cmd)
        observation = self.get_observation()
        return observation

    def get_qj(self):
        for i in range(self.config.num_joints):
            self.qj[i] = self.controller.state.motor_state[i].q
        return self.qj.tolist()

    def get_motor_temperatures(self) -> np.ndarray:
        return np.array([max(motor_state.temperature) for motor_state in self.controller.state.motor_state])

    def get_observation(self) -> Dict[str, np.ndarray]:
        """
        Get the current observation without taking any action.

        Returns:
            observation: Current observation array
        """
        # Get the current joint position and velocity
        for i in range(self.config.num_joints):
            self.qj[i] = self.controller.state.motor_state[i].q
            self.dqj[i] = self.controller.state.motor_state[i].dq

        # imu_state quaternion: w, x, y, z
        quat = self.controller.state.imu_state.quaternion
        ang_vel = np.array(self.controller.state.imu_state.gyroscope, dtype=np.float32)

        # create observation
        gravity_orientation = get_gravity_orientation(quat)

        if self.command_source is not None:
            self.cmd[:] = self.command_source()[:]

        self.obs = {
            "ang_vel": np.array(ang_vel),
            "imu_quat": np.array(quat),
            "gravity_orientation": np.array(gravity_orientation),
            "cmd": np.array(self.cmd),
            "qj": np.array(self.qj),
            "dqj": np.array(self.dqj),
            "action": np.array(self.action)
        }

        return self.obs

    def get_current_motor_state(self):
        """Get current motor state for leg joints."""
        dof_size = self.config.num_joints

        dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            dof_pos[i] = self.controller.state.motor_state[i].q

        return dof_pos

    def get_env_time(self) -> float:
        """
        Get the current environment time.

        For real robot, returns wall clock time.
        """
        return time.time()

    def go_to_initial_state(self, ask: bool = True, *, cancel_event=None):
        """Go to initial state."""
        if self.network != 'sim' and ask:
            input(f"Press Enter to move to initial state: {self.config.default_position}")
        print("Moving to initial state...")
        target_positions = self.config.default_position.copy()
        self.go_to_state(target_positions, cancel_event=cancel_event)
        print("moving to initial state Done!")

    def go_to_state(self, target_positions, time_to_move: float = 4, *, cancel_event=None):
        current_positions = self.get_current_motor_state()
        n_steps = int(time_to_move / self.config.control_dt)
        tgt_linspace = np.linspace(current_positions, target_positions, n_steps)
        for cur_tgt in tqdm.tqdm(tgt_linspace):
            if cancel_event is not None and cancel_event.is_set():
                break
            self._set_motor_target_positions(cur_tgt)
            if cancel_event is None:
                time.sleep(self.config.control_dt)
            elif cancel_event.wait(self.config.control_dt):
                break


    def _set_motor_target_positions(self, target_positions: np.ndarray):
        """Set motor commands for leg joints."""
        # print("Before clip:", target_positions)
        target_positions_clipped = np.clip(target_positions, self.config.joint_limits[:, 0], self.config.joint_limits[:, 1])
        # print("After clip:", target_positions_clipped)
        for i in range(self.config.num_joints):
            self.controller.low_cmd.motor_cmd[i].q = target_positions_clipped[i]
            self.controller.low_cmd.motor_cmd[i].dq = 0  # Target velocity
            self.controller.low_cmd.motor_cmd[i].kp = self.config.kps[i]  # Position gain from config
            self.controller.low_cmd.motor_cmd[i].kd = self.config.kds[i]  # Velocity gain from config
            self.controller.low_cmd.motor_cmd[i].tau = 0  # Feedforward torque
        self.controller.send_cmd(self.controller.low_cmd)

    def _execute_action(self, action: np.ndarray):
        target_position = self.config.default_position.copy()
        target_position[self.action_mask] = action
        self._set_motor_target_positions(
            target_position
        )

    def zero_torque_mode(self):
        """Set robot to zero torque mode."""
        self.controller.zero_torque_mode()

    def damping_mode(self):
        """Set robot to damping mode."""
        self.controller.damping_mode()

    def get_remote_controller(self):
        """Get the remote controller object."""
        return self.controller.get_remote_controller()

    @property
    def disconnected(self):
        """Get whether robot is disconnected"""
        return self.controller.disconnected

    @property
    def state(self):
        """Get the current robot state."""
        return self.controller.state

    @property
    def command(self):
        """Get the current command."""
        return self.controller.command
