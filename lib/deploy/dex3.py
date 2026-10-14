# Derived from the Dex3-1 hand controller in Unitree XR Teleoperate
# (https://github.com/unitreerobotics/xr_teleoperate), Copyright 2025
# HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics"), Apache License 2.0. Modified by YANDEX LLC. See NOTICES.
# for dex3-1
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_                               # idl
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
# for gripper
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_                           # idl
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_

import numpy as np
from enum import IntEnum
from dataclasses import dataclass
import time
import threading
from multiprocessing import Process, shared_memory, Array, Value, Lock

import logging
logger_mp = logging.getLogger(__name__)


Dex3_Num_Motors = 7
kTopicDex3LeftCommand = "rt/dex3/left/cmd"
kTopicDex3RightCommand = "rt/dex3/right/cmd"
kTopicDex3LeftState = "rt/dex3/left/state"
kTopicDex3RightState = "rt/dex3/right/state"


class Dex3_1_Controller:
    def __init__(self, policy, left_hand_in, right_hand_in, dual_hand_data_lock = None, dual_hand_state_array_out = None,  dual_hand_sensor_array_out = None,
                       dual_hand_action_array_out = None, fps = 100.0, simulation_mode = False,
                       hands = 'both', swap_hands = False, network = None):
        """
        [note] A *_array type parameter requires using a multiprocessing Array, because it needs to be passed to the internal child process

        left_hand_in: [input] Left hand skeleton data (required from XR device) to hand_ctrl.control_process

        right_hand_in: [input] Right hand skeleton data (required from XR device) to hand_ctrl.control_process

        dual_hand_data_lock: Data synchronization lock for dual_hand_state_array and dual_hand_action_array

        dual_hand_state_array_out: [output] Return left(7), right(7) hand motor state

        dual_hand_action_array_out: [output] Return left(7), right(7) hand motor action

        fps: Control frequency

                simulation_mode: Whether to use simulation mode (default is False, which means using real robot)
        """
        logger_mp.info("Initialize Dex3_1_Controller...")

        self.policy = policy
        self.fps = fps
        self.simulation_mode = simulation_mode
        self.network = network

        if self.simulation_mode:
            ChannelFactoryInitialize(1)
        elif self.network:
            ChannelFactoryInitialize(0, self.network)
        else:
            ChannelFactoryInitialize(0)

        self.left_hand = hands in ['both', 'left']
        self.right_hand = hands in ['both', 'right']

        self.dual_hand_data_lock = dual_hand_data_lock

        self.left_hand_command_topic = kTopicDex3LeftCommand
        left_hand_state_topic = kTopicDex3LeftState

        self.right_hand_command_topic = kTopicDex3RightCommand
        right_hand_state_topic = kTopicDex3RightState

        self.swap_hands_coeff = 1
        if swap_hands:
            self.left_hand_command_topic, self.right_hand_command_topic = self.right_hand_command_topic, self.left_hand_command_topic
            left_hand_state_topic, right_hand_state_topic = right_hand_state_topic, left_hand_state_topic
            self.swap_hands_coeff = -1

        # initialize handcmd publisher and handstate subscriber
        # NOTE: publishers are also created in control_process (child process) to avoid pickling issues
        # with multiprocessing spawn. The main process publishers are kept for external API calls.
        if self.left_hand:
            self.LeftHandCmb_publisher = ChannelPublisher(self.left_hand_command_topic, HandCmd_)
            self.LeftHandCmb_publisher.Init()
            self.LeftHandState_subscriber = ChannelSubscriber(left_hand_state_topic, HandState_)
            self.LeftHandState_subscriber.Init()
        else:
            self.LeftHandCmb_publisher = None
            self.LeftHandState_subscriber = None

        if self.right_hand:
            self.RightHandCmb_publisher = ChannelPublisher(self.right_hand_command_topic, HandCmd_)
            self.RightHandCmb_publisher.Init()
            self.RightHandState_subscriber = ChannelSubscriber(right_hand_state_topic, HandState_)
            self.RightHandState_subscriber.Init()
        else:
            self.RightHandCmb_publisher = None
            self.RightHandState_subscriber = None

        # Initialize command messages for external API calls
        q, dq, tau, kp, kd = 0.0, 0.0, 0.0, 0.75, 0.2
        self.left_msg = unitree_hg_msg_dds__HandCmd_()
        for id in Dex3_1_Left_JointIndex:
            ris_mode = self._RIS_Mode(id=id, status=0x01)
            motor_mode = ris_mode._mode_to_uint8()
            self.left_msg.motor_cmd[id].mode = motor_mode
            self.left_msg.motor_cmd[id].q = q
            self.left_msg.motor_cmd[id].dq = dq
            self.left_msg.motor_cmd[id].tau = tau
            self.left_msg.motor_cmd[id].kp = kp
            self.left_msg.motor_cmd[id].kd = kd

        self.right_msg = unitree_hg_msg_dds__HandCmd_()
        for id in Dex3_1_Right_JointIndex:
            ris_mode = self._RIS_Mode(id=id, status=0x01)
            motor_mode = ris_mode._mode_to_uint8()
            self.right_msg.motor_cmd[id].mode = motor_mode
            self.right_msg.motor_cmd[id].q = q
            self.right_msg.motor_cmd[id].dq = dq
            self.right_msg.motor_cmd[id].tau = tau
            self.right_msg.motor_cmd[id].kp = kp
            self.right_msg.motor_cmd[id].kd = kd

        # Shared Arrays for hand states
        self.left_hand_state_array  = Array('d', Dex3_Num_Motors, lock=True)
        self.right_hand_state_array = Array('d', Dex3_Num_Motors, lock=True)

        self.left_hand_sensor_array = Array('d', 9 * 12, lock=True)
        self.right_hand_sensor_array = Array('d', 9 * 12, lock=True)

        # initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        while True:
            if (not self.left_hand or any(self.left_hand_state_array)) and (not self.right_hand or any(self.right_hand_state_array)):
                break
            time.sleep(0.01)
            logger_mp.warning("[Dex3_1_Controller] Waiting to subscribe dds...")
        logger_mp.info("[Dex3_1_Controller] Subscribe dds ok.")

        self.hand_control_process = Process(target=Dex3_1_Controller._control_process_wrapper, daemon=True, args=(
            self.policy, self.fps, self.simulation_mode, self.network, self.left_hand, self.right_hand,
            self.left_hand_command_topic, self.right_hand_command_topic, self.swap_hands_coeff,
            left_hand_in, right_hand_in, self.left_hand_state_array, self.right_hand_state_array,
            self.left_hand_sensor_array, self.right_hand_sensor_array,
            dual_hand_data_lock, dual_hand_state_array_out, dual_hand_sensor_array_out, dual_hand_action_array_out
        ))

        self.hand_control_process.start()

        logger_mp.info("Initialize Dex3_1_Controller OK!")

    def start(self):
        self.hand_control_process.start()

    def get_ee_state(self):
        with self.dual_hand_data_lock:
            left_ee_state = self.left_hand_state_array[:]
            right_ee_state = self.right_hand_state_array[:]
        return left_ee_state, right_ee_state

    def get_ee_sensor(self):
        with self.dual_hand_data_lock:
            left_ee_sensor = self.left_hand_sensor_array[:]
            right_ee_sensor = self.right_hand_sensor_array[:]
        return left_ee_sensor, right_ee_sensor

    def _subscribe_hand_state(self):
        while True:
            left_hand_msg  = self.LeftHandState_subscriber.Read() if self.LeftHandState_subscriber is not None else None
            right_hand_msg = self.RightHandState_subscriber.Read() if self.RightHandState_subscriber is not None else None

            if left_hand_msg is not None:
                for idx, id in enumerate(Dex3_1_Left_JointIndex):
                    self.left_hand_state_array[idx] = left_hand_msg.motor_state[id].q * (self.swap_hands_coeff if idx else 1)
                for sensor_idx in range(9):
                    for value_idx in range(12):
                        self.left_hand_sensor_array[sensor_idx * 12 + value_idx] = \
                            left_hand_msg.press_sensor_state[sensor_idx].pressure[value_idx] / 10000

            if right_hand_msg is not None:
                for idx, id in enumerate(Dex3_1_Right_JointIndex):
                    self.right_hand_state_array[idx] = right_hand_msg.motor_state[id].q * (self.swap_hands_coeff if idx else 1)
                for sensor_idx in range(9):
                    for value_idx in range(12):
                        self.right_hand_sensor_array[sensor_idx * 12 + value_idx] = \
                            right_hand_msg.press_sensor_state[sensor_idx].pressure[value_idx] / 10000

            time.sleep(0.002)

    class _RIS_Mode:
        def __init__(self, id=0, status=0x01, timeout=0):
            self.motor_mode = 0
            self.id = id & 0x0F  # 4 bits for id
            self.status = status & 0x07  # 3 bits for status
            self.timeout = timeout & 0x01  # 1 bit for timeout

        def _mode_to_uint8(self):
            self.motor_mode |= (self.id & 0x0F)
            self.motor_mode |= (self.status & 0x07) << 4
            self.motor_mode |= (self.timeout & 0x01) << 7
            return self.motor_mode

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        """set current left, right hand motor state target q"""
        for idx, id in enumerate(Dex3_1_Left_JointIndex):
            self.left_msg.motor_cmd[id].q = left_q_target[idx]
        for idx, id in enumerate(Dex3_1_Right_JointIndex):
            self.right_msg.motor_cmd[id].q = right_q_target[idx]

        if self.LeftHandCmb_publisher is not None:
            self.LeftHandCmb_publisher.Write(self.left_msg)
        if self.RightHandCmb_publisher is not None:
            self.RightHandCmb_publisher.Write(self.right_msg)
        # logger_mp.debug("hand ctrl publish ok.")

    def relax(self):
        publishers = []
        if self.LeftHandCmb_publisher is not None:
            publishers.append(self.LeftHandCmb_publisher)
        if self.RightHandCmb_publisher is not None:
            publishers.append(self.RightHandCmb_publisher)
        for publisher in publishers:
            relax_msg  = unitree_hg_msg_dds__HandCmd_()
            for id in Dex3_1_Left_JointIndex:
                ris_mode = self._RIS_Mode(id=id, status=0x01, timeout=0x01)
                motor_mode = ris_mode._mode_to_uint8()
                relax_msg.motor_cmd[id].mode = motor_mode
                relax_msg.motor_cmd[id].q    = 0.0
                relax_msg.motor_cmd[id].dq   = 0.0
                relax_msg.motor_cmd[id].tau  = 0.0
                relax_msg.motor_cmd[id].kp   = 0.0
                relax_msg.motor_cmd[id].kd   = 0.0
            publisher.Write(relax_msg)
        logger_mp.info("Hands relaxed (motors disabled).")

    @staticmethod
    def _control_process_wrapper(policy, fps, simulation_mode, network, left_hand_enabled, right_hand_enabled,
                                  left_hand_command_topic, right_hand_command_topic, swap_hands_coeff,
                                  left_hand_in, right_hand_in, left_hand_state_array, right_hand_state_array,
                                  left_hand_sensor_array, right_hand_sensor_array,
                                  dual_hand_data_lock=None,
                                  dual_hand_state_array_out=None, dual_hand_sensor_array_out=None, dual_hand_action_array_out=None):
        """
        Static wrapper for control process that initializes publishers in the child process.
        This avoids pickling issues with Channel objects when using multiprocessing spawn.
        """
        # Initialize channel factory in child process
        if simulation_mode:
            ChannelFactoryInitialize(1)
        elif network:
            ChannelFactoryInitialize(0, network)
        else:
            ChannelFactoryInitialize(0)

        # Create publishers in the child process
        if left_hand_enabled:
            LeftHandCmb_publisher = ChannelPublisher(left_hand_command_topic, HandCmd_)
            LeftHandCmb_publisher.Init()
        else:
            LeftHandCmb_publisher = None

        if right_hand_enabled:
            RightHandCmb_publisher = ChannelPublisher(right_hand_command_topic, HandCmd_)
            RightHandCmb_publisher.Init()
        else:
            RightHandCmb_publisher = None

        running = True

        left_q_target  = np.full(Dex3_Num_Motors, 0)
        right_q_target = np.full(Dex3_Num_Motors, 0)

        q = 0.0
        dq = 0.0
        tau = 0.0
        # Original params:
        # kp = 1.5
        # kd = 0.2
        kp = 0.75
        kd = 0.2

        # Helper class for RIS mode (same as _RIS_Mode inner class)
        class _RIS_Mode:
            def __init__(self, id=0, status=0x01, timeout=0):
                self.motor_mode = 0
                self.id = id & 0x0F  # 4 bits for id
                self.status = status & 0x07  # 3 bits for status
                self.timeout = timeout & 0x01  # 1 bit for timeout

            def _mode_to_uint8(self):
                self.motor_mode |= (self.id & 0x0F)
                self.motor_mode |= (self.status & 0x07) << 4
                self.motor_mode |= (self.timeout & 0x01) << 7
                return self.motor_mode

        # initialize dex3-1's left hand cmd msg
        left_msg = unitree_hg_msg_dds__HandCmd_()
        for id in Dex3_1_Left_JointIndex:
            ris_mode = _RIS_Mode(id=id, status=0x01)
            motor_mode = ris_mode._mode_to_uint8()
            left_msg.motor_cmd[id].mode = motor_mode
            left_msg.motor_cmd[id].q    = q
            left_msg.motor_cmd[id].dq   = dq
            left_msg.motor_cmd[id].tau  = tau
            left_msg.motor_cmd[id].kp   = kp
            left_msg.motor_cmd[id].kd   = kd

        # initialize dex3-1's right hand cmd msg
        right_msg = unitree_hg_msg_dds__HandCmd_()
        for id in Dex3_1_Right_JointIndex:
            ris_mode = _RIS_Mode(id=id, status=0x01)
            motor_mode = ris_mode._mode_to_uint8()
            right_msg.motor_cmd[id].mode = motor_mode
            right_msg.motor_cmd[id].q    = q
            right_msg.motor_cmd[id].dq   = dq
            right_msg.motor_cmd[id].tau  = tau
            right_msg.motor_cmd[id].kp   = kp
            right_msg.motor_cmd[id].kd   = kd

        def ctrl_dual_hand(left_q_target, right_q_target):
            """set current left, right hand motor state target q"""
            for idx, id in enumerate(Dex3_1_Left_JointIndex):
                left_msg.motor_cmd[id].q = left_q_target[idx]
            for idx, id in enumerate(Dex3_1_Right_JointIndex):
                right_msg.motor_cmd[id].q = right_q_target[idx]

            if LeftHandCmb_publisher is not None:
                LeftHandCmb_publisher.Write(left_msg)
            if RightHandCmb_publisher is not None:
                RightHandCmb_publisher.Write(right_msg)

        def relax():
            publishers = []
            if LeftHandCmb_publisher is not None:
                publishers.append(LeftHandCmb_publisher)
            if RightHandCmb_publisher is not None:
                publishers.append(RightHandCmb_publisher)
            for publisher in publishers:
                relax_msg = unitree_hg_msg_dds__HandCmd_()
                for id in Dex3_1_Left_JointIndex:
                    ris_mode = _RIS_Mode(id=id, status=0x01, timeout=0x01)
                    motor_mode = ris_mode._mode_to_uint8()
                    relax_msg.motor_cmd[id].mode = motor_mode
                    relax_msg.motor_cmd[id].q    = 0.0
                    relax_msg.motor_cmd[id].dq   = 0.0
                    relax_msg.motor_cmd[id].tau  = 0.0
                    relax_msg.motor_cmd[id].kp   = 0.0
                    relax_msg.motor_cmd[id].kd   = 0.0
                publisher.Write(relax_msg)
            logger_mp.info("Hands relaxed (motors disabled).")

        try:
            while running:
                start_time = time.time()

                # Read left and right q_state from shared arrays
                state_data = np.concatenate((np.array(left_hand_state_array[:]), np.array(right_hand_state_array[:])))
                sensor_data = np.concatenate((np.array(left_hand_sensor_array[:]), np.array(right_hand_sensor_array[:])))

                left_q, right_q = policy.get_targets()
                if left_q is not None and right_q is not None:
                    left_q_target = left_q
                    right_q_target = right_q

                # get dual hand action
                action_data = np.concatenate((left_q_target, right_q_target))
                if dual_hand_state_array_out and dual_hand_sensor_array_out and dual_hand_action_array_out:
                    with dual_hand_data_lock:
                        dual_hand_state_array_out[:] = state_data
                        dual_hand_sensor_array_out[:] = sensor_data
                        dual_hand_action_array_out[:] = action_data

                ctrl_dual_hand(left_q_target, right_q_target)
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            relax()
            logger_mp.info("Dex3_1_Controller has been closed.")


class Dex3_1_Gripper_ControllerPolicy:
    def __init__(self, left_gripper_value, right_gripper_value, swap_hands = False):
        self.swap_hands = swap_hands
        self.left_gripper_value = left_gripper_value
        self.right_gripper_value = right_gripper_value

        # self.left_gripper_value = Value('d', 100.0, lock=True) # [input]
        # self.right_gripper_value = Value('d', 100.0, lock=True) # [input]

        self.LEFT_MAPPED_CLOSED = Array('d', 7, lock=True)
        self.LEFT_MAPPED_OPEN = Array('d', 7, lock=True)
        self.RIGHT_MAPPED_CLOSED = Array('d', 7, lock=True)
        self.RIGHT_MAPPED_OPEN = Array('d', 7, lock=True)

        # Base configurations for each hand type
        # the order is:
        # [thumb agle, thumb joint 1, pinky 1, pinky 2, index 1, index 2]
        self.base_configs = {
            'normal': {
                'left': {
                    'closed': np.array([0, 0.92, 1.74, -1.57, -1.74, -1.57, -1.74]),
                    'open': np.array([0, -0.72, 0, 0, 0, 0, 0])
                },
                'right': {
                    'closed': np.array([0, -0.92, -1.74, 1.57, 1.74, 1.57, 1.74]),
                    'open': np.array([0, 0.72, 0, 0, 0, 0, 0])
                }
            },
            'pointer': {
                'left': {
                    'closed': np.array([-0.61, 1.2, 1.14, -1.57, -1.74, -1.57, -1.74]),
                    'open': np.array([-0.61, -0.72, 0, -1.57, -1.74, 0, 0])
                },
                'right': {
                    'closed': np.array([-0.61, -1.2, -1.14, 1.57, 1.74, 1.45, 1.55]),
                    'open': np.array([-0.61, 0.72, 0, 1.57, 1.74, 0, 0])
                }
            },
            'pincher': {
                'left': {
                    'closed': np.array([-0.61, 1.2, 1.14, 0, 0, -1.45, -1.55]),
                    'open': np.array([-0.61, -0.72, 0, 0, 0, 0, 0])
                },
                'right': {
                    'closed': np.array([-0.61, -1.2, -1.14, 0, 0, 1.45, 1.55]),
                    'open': np.array([-0.61, 0.72, 0, 0, 0, 0, 0])
                }
            }
        }

        self.left_mode = None
        self.right_mode = None
        self.set_mode({'left': 'normal', 'right': 'normal'})

    def set_tele_data(self, left_pinch_value, right_pinch_value):
        if left_pinch_value is not None:
            with self.left_gripper_value.get_lock():
                self.left_gripper_value.value = left_pinch_value
        if right_pinch_value is not None:
            with self.right_gripper_value.get_lock():
                self.right_gripper_value.value = right_pinch_value

    def get_targets(self):
        with self.left_gripper_value.get_lock():
            left_hand_value = self.left_gripper_value.value
        with self.right_gripper_value.get_lock():
            right_hand_value = self.right_gripper_value.value

        left_hand_value = np.interp(left_hand_value, [0.0, 100.0], [0.15, 0.85])
        right_hand_value = np.interp(right_hand_value, [0.0, 100.0], [0.15, 0.85])

        left_q_target = np.array([
            np.interp(left_hand_value, [0.0, 1.0], [q_closed, q_open])
            for q_closed, q_open in zip(self.LEFT_MAPPED_CLOSED, self.LEFT_MAPPED_OPEN)
        ])

        right_q_target = np.array([
            np.interp(right_hand_value, [0.0, 1.0], [q_closed, q_open])
            for q_closed, q_open in zip(self.RIGHT_MAPPED_CLOSED, self.RIGHT_MAPPED_OPEN)
        ])

        return left_q_target, right_q_target

    def set_mode(self, mode: dict[str, str]):

        if mode.get('left') and mode['left'] != self.left_mode:
            self.left_mode = mode['left']
            self.LEFT_MAPPED_CLOSED[:] = self.base_configs[self.left_mode]['left']['closed']
            self.LEFT_MAPPED_OPEN[:] = self.base_configs[self.left_mode]['left']['open']

        if mode.get('right') and mode['right'] != self.right_mode:
            self.right_mode = mode['right']
            self.RIGHT_MAPPED_CLOSED[:] = self.base_configs[self.right_mode]['right']['closed']
            self.RIGHT_MAPPED_OPEN[:] = self.base_configs[self.right_mode]['right']['open']

        if self.swap_hands:
            self.LEFT_MAPPED_CLOSED, self.RIGHT_MAPPED_CLOSED = self.RIGHT_MAPPED_CLOSED, self.LEFT_MAPPED_CLOSED
            self.LEFT_MAPPED_OPEN, self.RIGHT_MAPPED_OPEN = self.RIGHT_MAPPED_OPEN, self.LEFT_MAPPED_OPEN



class Dex3_1_Left_JointIndex(IntEnum):
    kLeftHandThumb0 = 0
    kLeftHandThumb1 = 1
    kLeftHandThumb2 = 2
    kLeftHandMiddle0 = 3
    kLeftHandMiddle1 = 4
    kLeftHandIndex0 = 5
    kLeftHandIndex1 = 6

class Dex3_1_Right_JointIndex(IntEnum):
    kRightHandThumb0 = 0
    kRightHandThumb1 = 1
    kRightHandThumb2 = 2
    kRightHandIndex0 = 3
    kRightHandIndex1 = 4
    kRightHandMiddle0 = 5
    kRightHandMiddle1 = 6


kTopicGripperLeftCommand = "rt/dex1/left/cmd"
kTopicGripperLeftState = "rt/dex1/left/state"
kTopicGripperRightCommand = "rt/dex1/right/cmd"
kTopicGripperRightState = "rt/dex1/right/state"

class Gripper_JointIndex(IntEnum):
    kGripper = 0

class SimDex3Controller:
    """Dex3 controller facade for sync sim.

    No DDS, no subprocess. The controller owns the same policy as the real
    controller, applies policy targets to the sim env, and reports actual sim
    joint state back through the Dex3_1_Controller-shaped API.
    """

    _SENSOR_ZEROS: list = [0.0] * (9 * 12)

    def __init__(
        self,
        policy: "Dex3_1_Gripper_ControllerPolicy",
        hands: str = 'both',
        env=None,
    ):
        self.policy = policy
        self.left_hand  = hands in ('both', 'left')
        self.right_hand = hands in ('both', 'right')
        self.env = env

    def attach_env(self, env):
        self.env = env

    def apply_targets(self):
        if self.env is None:
            raise RuntimeError("SimDex3Controller requires an env before applying targets")

        left_q, right_q = self.policy.get_targets()
        left = list(left_q) if self.left_hand else [0.0] * Dex3_Num_Motors
        right = list(right_q) if self.right_hand else [0.0] * Dex3_Num_Motors
        self.env.apply_hand_targets(left, right)
        return left, right

    def get_ee_state(self):
        """Return (left_q[7], right_q[7]) as lists.

        State comes from the sim articulation, not from commanded policy targets.
        """
        if self.env is None:
            raise RuntimeError("SimDex3Controller requires an env before reading state")

        left_q, right_q = self.env.get_hand_state()
        left = list(left_q) if self.left_hand else [0.0] * Dex3_Num_Motors
        right = list(right_q) if self.right_hand else [0.0] * Dex3_Num_Motors
        return left, right

    def get_ee_sensor(self):
        """Return (left_sensor[108], right_sensor[108]) — all zeros in sim."""
        return self._SENSOR_ZEROS, self._SENSOR_ZEROS
