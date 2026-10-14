import threading

from lib.deploy.control.sleeper import create_sleeper_from_g1_env
from typing import Optional

from lib.deploy.control.base_env import BaseEnv
from lib.deploy.debug.logs import DeployConsole, emit_sleep_warnings

from lib.deploy.fsm.common import StartableNode, DisabledState, ActiveState


class DampNode(StartableNode):

    def __init__(self, env: BaseEnv, console: Optional[DeployConsole] = None):
        super().__init__()
        self.env = env
        self.console = console if console is not None else DeployConsole(self.name)
        self.sleeper = create_sleeper_from_g1_env(
            env,
            enable_logging=False,
        )
        self._thread = None
        self._stop_event = threading.Event()

    def activate(self):
        """Start the damping thread"""
        self.console.info(f"Activating {self.name}")
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._node_loop, daemon=True)
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
        while not self._stop_event.is_set():
            self.env.damping_mode()
            time_slept = self.sleeper.sleep()
            emit_sleep_warnings(
                self.console,
                time_slept,
                self.sleeper.last_oversleep_time,
            )
