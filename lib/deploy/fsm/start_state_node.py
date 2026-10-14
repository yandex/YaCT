import numpy as np
import torch
import typing as tp
import threading

from lib.deploy.fsm.common import Node, DisabledState, ActiveState, BusyState
from lib.deploy.control.sleeper import create_sleeper_from_g1_env
from lib.deploy.control.base_env import BaseEnv
from lib.deploy.debug.logs import DeployConsole, emit_sleep_warnings
from lib.deploy.deploy_utils.networks import MockNetworkType, is_mock_network, is_sync_sim_network

from lib.deploy.configs.joint_mappings import (
    REAL_ORDER,
    MOTOR_ID_2_ACTION_ID,
)


class StartStateNode(Node):

    def __init__(
        self,
        env: BaseEnv,
        start_joint_pos_sim: tp.Optional[np.ndarray] = None,
        start_joint_pos_real: tp.Optional[np.ndarray] = None,
        *,
        console: tp.Optional[DeployConsole] = None,
    ):
        super().__init__()
        self.env = env
        self.console = console if console is not None else DeployConsole(self.name)
        self.sleeper = create_sleeper_from_g1_env(
            env,
            enable_logging=False,
        )
        self._thread = None
        self._stop_event = threading.Event()

        if self.env.network == MockNetworkType.SIM:
            from lib.deploy.deploy_utils.g1_sim import init_reset_topic
            self._sim_reset = init_reset_topic(init_time_printer=False)
        elif is_sync_sim_network(self.env.network):
            self._sim_reset = lambda: self.env.reset_sim()

        assert not ((start_joint_pos_sim is not None)
                    and (start_joint_pos_real is not None))

        if start_joint_pos_sim is not None:
            assert start_joint_pos_real is None

            joint_mapping_from_sim = torch.zeros(len(REAL_ORDER)).long()

            if isinstance(start_joint_pos_sim, torch.Tensor):
                start_joint_pos_sim = start_joint_pos_sim.detach().clone().float()
            else:
                start_joint_pos_sim = torch.as_tensor(start_joint_pos_sim, dtype=torch.float32)
            start_joint_pos_real = torch.zeros_like(start_joint_pos_sim)
            for motor_id, sim_id in MOTOR_ID_2_ACTION_ID.items():
                joint_mapping_from_sim[motor_id] = sim_id

            torch.index_select(start_joint_pos_sim, 0,
                               joint_mapping_from_sim, out=start_joint_pos_real)
            start_joint_pos_real = start_joint_pos_real.detach().cpu().numpy()

        if start_joint_pos_real is None:
            start_joint_pos_real = self.env.config.default_position
        self.start_joint_pos_real = start_joint_pos_real

    def activate(self):
        """Start the damping thread"""
        self.console.info(f"Activating {self.name}")
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
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

    def _node_loop(self):
        """Main damping loop that runs in a separate thread"""

        if is_mock_network(self.env.network):
            self._sim_reset()

        self.state = BusyState()
        if not is_mock_network(self.env.network):
            if self.start_joint_pos_real is None:
                self.env.go_to_initial_state(ask=False, cancel_event=self._stop_event)
            else:
                self.env.go_to_state(self.start_joint_pos_real, 2, cancel_event=self._stop_event)

        if self._stop_event.is_set():
            return
        self.state = ActiveState()

        while not self._stop_event.is_set():
            if not is_mock_network(self.env.network):
                self.env.act(self.start_joint_pos_real)
            time_slept = self.sleeper.sleep()
            emit_sleep_warnings(
                self.console,
                time_slept,
                self.sleeper.last_oversleep_time,
            )
