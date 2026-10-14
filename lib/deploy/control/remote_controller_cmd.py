import time
import threading
from typing import Callable, Dict, Tuple, Union
from lib.deploy.control.remote_controller import Joystick, RemoteController, KeyMap


class RemoteControllerCommands:
    def __init__(self, remote_controller: RemoteController):
        self._remote_controller = remote_controller
        self._stop_event = threading.Event()
        self._commands: Dict[Union[KeyMap, Joystick], Callable] = {}
        self._prev_keys_pressed = []

    def add_command(self, key: KeyMap, command: Callable):
        self._commands[key] = command

    def add_joystick_command(self, joystick: Joystick, command: Callable[[Tuple[float, float]], None]):
        self._commands[joystick] = command

    def start(self):
        self._thread = threading.Thread(target=self._read_worker, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()

    def _read_worker(self):
        while not self._stop_event.is_set():
            keys = self._remote_controller.get_keys_pressed()
            for key in keys:
                if key in self._commands.keys():
                    if key not in self._prev_keys_pressed:
                        print(f"Key '{key.name}' pressed")
                        self._commands[key]()

            for joystick in Joystick:
                if joystick in self._commands.keys():
                    self._commands[joystick](self._remote_controller.get_joystick(joystick))

            self._prev_keys_pressed = keys

            time.sleep(0.01)
