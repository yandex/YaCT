# Derived from Unitree RL Gym deploy_real
# (https://github.com/unitreerobotics/unitree_rl_gym), Copyright (c) 2016-2023
# HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics"), BSD 3-Clause. See NOTICES.
from typing import List, Union, Any

import numpy as np
from scipy.spatial.transform import Rotation as R
from lib.deploy.configs.joint_mappings import MOTOR_ID_2_ACTION_ID

try:
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
except:
    pass

class MotorMode:
    PR = 0  # Series Control for Pitch/Roll Joints
    AB = 1  # Parallel Control for A/B Joints


def create_damping_cmd(cmd: Union["LowCmdGo", "LowCmdHG"]):
    size = len(cmd.motor_cmd)
    for i in range(size):
        cmd.motor_cmd[i].q = 0
        cmd.motor_cmd[i].qd = 0
        cmd.motor_cmd[i].kp = 0
        cmd.motor_cmd[i].kd = 8
        cmd.motor_cmd[i].tau = 0


def create_zero_cmd(cmd: Union["LowCmdGo", "LowCmdHG"]):
    size = len(cmd.motor_cmd)
    for i in range(size):
        cmd.motor_cmd[i].q = 0
        cmd.motor_cmd[i].qd = 0
        cmd.motor_cmd[i].kp = 0
        cmd.motor_cmd[i].kd = 0
        cmd.motor_cmd[i].tau = 0


def init_cmd_hg(cmd: "LowCmdHG", mode_machine: int, mode_pr: int):
    cmd.mode_machine = mode_machine
    cmd.mode_pr = mode_pr
    size = len(cmd.motor_cmd)
    for i in range(size):
        cmd.motor_cmd[i].mode = 1
        cmd.motor_cmd[i].q = 0
        cmd.motor_cmd[i].qd = 0
        cmd.motor_cmd[i].kp = 0
        cmd.motor_cmd[i].kd = 0
        cmd.motor_cmd[i].tau = 0


def init_cmd_go(cmd: "LowCmdGo", weak_motor: list):
    cmd.head[0] = 0xFE
    cmd.head[1] = 0xEF
    cmd.level_flag = 0xFF
    cmd.gpio = 0
    PosStopF = 2.146e9
    VelStopF = 16000.0
    size = len(cmd.motor_cmd)
    for i in range(size):
        if i in weak_motor:
            cmd.motor_cmd[i].mode = 1
        else:
            cmd.motor_cmd[i].mode = 0x0A
        cmd.motor_cmd[i].q = PosStopF
        cmd.motor_cmd[i].qd = VelStopF
        cmd.motor_cmd[i].kp = 0
        cmd.motor_cmd[i].kd = 0
        cmd.motor_cmd[i].tau = 0


def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3)

    # Alternative formulation
    gravity_orientation[0] = -2 * (qx * qz - qw * qy)
    gravity_orientation[1] = -2 * (qy * qz + qw * qx)
    gravity_orientation[2] = -(1 - 2 * (qx * qx + qy * qy))

    return gravity_orientation


def action_to_lowcmd(
    action: np.ndarray,
    cmd: "LowCmdHG",
    default_positions: List[float],
    kp: List[float] = [3.0] * 35,
    kd: List[float] = [1.0] * 35,
    ) -> "LowCmdHG":
    """
    Convert action array (in the same format as action_from_motor) to LowCmd_ object.

    Args:
        action: numpy array of shape (num_envs, num_joints) or (num_joints,)
               containing joint position targets in the order defined by JOINT_LIST
        cmd: LowCmdHG object to populate
        kp: position gain for all motors
        kd: velocity gain for all motors

    Returns:
        LowCmdHG: populated command object with motor commands in correct G1 motor order
    """
    # Handle single action case
    if action.ndim == 1:
        action = action.reshape(1, -1)

    # Use first environment's action if multiple environments
    joint_positions = action[0] if action.shape[0] > 1 else action.flatten()

    # Initialize all motors with default positions
    for motor_id in range(35):  # G1 has 35 motors
        cmd.motor_cmd[motor_id].mode = 1  # Position control mode
        cmd.motor_cmd[motor_id].q = default_positions[motor_id]
        cmd.motor_cmd[motor_id].dq = 0.0  # Target velocity
        cmd.motor_cmd[motor_id].kp = kp[motor_id]
        cmd.motor_cmd[motor_id].kd = kd[motor_id]
        cmd.motor_cmd[motor_id].tau = 0.0  # Feedforward torque
        cmd.motor_cmd[motor_id].reserve = 0

    # Map action positions to corresponding G1 motors
    for motor_id, action_idx in MOTOR_ID_2_ACTION_ID.items():
        if action_idx < len(joint_positions):
            cmd.motor_cmd[motor_id].q = float(joint_positions[action_idx])

    return cmd


def transform_imu_data(waist_yaw, waist_yaw_omega, imu_quat, imu_omega):
    RzWaist = R.from_euler("z", waist_yaw).as_matrix()
    R_torso = R.from_quat([imu_quat[1], imu_quat[2], imu_quat[3], imu_quat[0]]).as_matrix()
    R_pelvis = np.dot(R_torso, RzWaist.T)
    w = np.dot(RzWaist, imu_omega[0]) - np.array([0, 0, waist_yaw_omega])
    return R.from_matrix(R_pelvis).as_quat()[[3, 0, 1, 2]], w
