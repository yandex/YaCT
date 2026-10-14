# The low-level command loop is derived from Unitree RL Gym deploy_real
# (https://github.com/unitreerobotics/unitree_rl_gym), Copyright (c) 2016-2023
# HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics"), BSD 3-Clause. See NOTICES.
import os
import time
from typing import Optional

from lib.vendor.env import in_docker
if in_docker('g1_sim'):
    from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
    from unitree_sdk2py.utils.crc import CRC
    from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

from lib.deploy.control.util import create_damping_cmd, create_zero_cmd, init_cmd_hg, MotorMode
from lib.deploy.control.remote_controller import RemoteController, KeyMap
from lib.deploy.control.config import Config


class Controller:
    def __init__(self, config: Config, network: Optional[str] = None, disconnection_timeout_s: float = 0.1) -> None:

        self.network = network
        self.config = config

        self.remote_controller = RemoteController()
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.mode_pr_ = MotorMode.PR # Pitch Roll ankle controll (default, recommended)
        self.mode_machine_ = 0

        self.last_state_time = -1
        self.disconnection_timeout_s = disconnection_timeout_s

        self._init_factory()
        self._init_pubs()
        if self.network != 'sim':
            self._init_msc()
            self._wait_for_low_state()


        init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        print("init finished!")

    def _init_factory(self):
        print('channel factory init')
        if self.network == 'sim':
            ChannelFactoryInitialize(1)
        else:
            ChannelFactoryInitialize(0, self.network)


    def _init_msc(self):
        print("switching to Release Mode...")
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()
        status, result = self.msc.CheckMode()
        while result['name']:
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)

    def _init_pubs(self):
        print('initialize cmd publisher ...')
        self.lowcmd_publisher_ = ChannelPublisher(self.config.lowcmd_topic, LowCmdHG)
        self.lowcmd_publisher_.Init()

        print('initialize subscriber ...')
        self.lowstate_subscriber = ChannelSubscriber(self.config.lowstate_topic, LowStateHG)
        self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

    def _wait_for_low_state(self):
        print('waiting for the first read ...')
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    @property
    def state(self):
        return self.low_state

    @property
    def command(self):
        return self.low_cmd

    def get_remote_controller(self) -> RemoteController:
        return self.remote_controller

    def LowStateHgHandler(self, msg):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)
        self.last_state_time = time.time()

    def send_cmd(self, cmd):
        self.low_cmd = cmd
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def zero_torque_mode(self):
        create_zero_cmd(self.low_cmd)
        self.send_cmd(self.low_cmd)

    def damping_mode(self):
        create_damping_cmd(self.low_cmd)
        self.send_cmd(self.low_cmd)

    @property
    def disconnected(self):
        return time.time() - self.last_state_time > self.disconnection_timeout_s
