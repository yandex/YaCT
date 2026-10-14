import copy
import numpy as np
import typing as tp
from lib.deploy.control.remote_controller import Joystick, RemoteController, KeyMap


class Commander:
    def get_command(self) -> np.ndarray:
        return ...

    def get_vertical_command(self) -> float:
        """Return vertical velocity command (positive = up, negative = down)."""
        return 0.0


class KeyboardCommander:
    # Vertical speed for up/down movement (m/s)
    VERTICAL_SPEED = 0.3

    def __init__(self, ranges, additional_commands: tp.Dict[str, tp.Callable[[], None]] = {}, max_abs: float = 1.):
        self.zero_command = np.zeros(3, dtype=np.float32)
        self.current_command = self.zero_command.copy()
        self.current_vertical_command = 0.0
        self.ranges = ranges
        self.max_abs = max_abs
        self.key_mappings = {
            'w': np.array([self.ranges.lin_vel_x_range[1], 0, 0], dtype=np.float32) * max_abs,   # Forward
            'a': np.array([0, self.ranges.lin_vel_y_range[1], 0], dtype=np.float32) * max_abs,  # Left
            's': np.array([self.ranges.lin_vel_x_range[0], 0, 0], dtype=np.float32) * max_abs,  # Backward
            'd': np.array([0, self.ranges.lin_vel_y_range[0], 0], dtype=np.float32) * max_abs,   # Right
            'e': np.array([0, 0, self.ranges.ang_vel_z_range[0]], dtype=np.float32) * max_abs,   # Rotate Right
            'q': np.array([0, 0, self.ranges.ang_vel_z_range[1]], dtype=np.float32) * max_abs,  # Rotate Left
            ' ': np.array([0, 0, 0], dtype=np.float32),   # Stop
        }
        # Vertical key mappings (up/down)
        self.vertical_key_mappings = {
            'o': self.VERTICAL_SPEED * max_abs,   # Up
            'l': -self.VERTICAL_SPEED * max_abs,  # Down
        }

        self.additional_commands = additional_commands

        self.stdin_commands = self._init_stdin_commands()

    def _init_stdin_commands(self):
        from lib.vendor.util.stdin_cmd import StdinCommands
        stdin_cmnds = StdinCommands()

        # Add commands for each key
        for key in self.key_mappings.keys():
            stdin_cmnds.add_command(key, lambda k=key: self._on_key_press(k))

        # Add vertical commands
        for key in self.vertical_key_mappings.keys():
            stdin_cmnds.add_command(key, lambda k=key: self._on_vertical_key_press(k))

        for key, fn in self.additional_commands.items():
            stdin_cmnds.add_command(key, fn)
        stdin_cmnds.start()
        return stdin_cmnds

    def _on_key_press(self, key):
        """Handle key press events"""
        if key in self.key_mappings:
            self.current_command = copy.deepcopy(self.key_mappings[key])
            self.current_vertical_command = 0.0  # Reset vertical when horizontal key pressed
            print(f"Key '{key}' pressed: {self.current_command}")
        return None  # Don't stop the command loop

    def _on_vertical_key_press(self, key):
        """Handle vertical key press events (up/down)"""
        if key in self.vertical_key_mappings:
            self.current_vertical_command = self.vertical_key_mappings[key]
            self.current_command = self.zero_command.copy()  # Reset horizontal when vertical key pressed
            print(f"Key '{key}' pressed: vertical={self.current_vertical_command}")
        return None

    def get_command(self):
        """Return the current command vector"""
        return copy.deepcopy(self.current_command)

    def get_vertical_command(self) -> float:
        """Return the current vertical command (positive = up, negative = down)"""
        return self.current_vertical_command

    def stop(self):
        """Stop the keyboard command listener"""
        if hasattr(self, 'stdin_commands'):
            self.stdin_commands.stop()


class RemoteControllerCommander:
    # Vertical speed for up/down movement (m/s)
    VERTICAL_SPEED = 0.3

    def __init__(self, remote_controller: RemoteController, ranges, additional_commands: tp.Dict[KeyMap, tp.Callable[[], None]] = {}):
        self.ranges = ranges
        self.zero_command = np.zeros(3, dtype=np.float32)
        self.current_command = self.zero_command.copy()
        self.current_vertical_command = 0.0

        self.remote_controller = remote_controller
        self._vertical_keys_pressed = set()  # Track which vertical keys are currently pressed

        self.additional_commands = additional_commands
        self.joystick_commands = {
            Joystick.left: self._on_left_joystick_move,
            Joystick.right: self._on_right_joystick_move
        }
        # Vertical commands (up/down d-pad)
        self.vertical_commands = {
            KeyMap.up: self._on_up_press,
            KeyMap.down: self._on_down_press,
        }

        self.remote_controller_commands = self._init_remote_controller_commands()

    def _on_left_joystick_move(self, pos):
        self.current_command[0] = pos[1] * (-self.ranges.lin_vel_x_range[0] if pos[1] < 0 else self.ranges.lin_vel_x_range[1])
        self.current_command[1] = -pos[0] * (-self.ranges.lin_vel_y_range[0] if pos[0] < 0 else self.ranges.lin_vel_y_range[1])

        # if abs(pos[0]) + abs(pos[1]) > 0.05:
        #     print(f"Left joystick moved: {self.current_command}")

    def _on_right_joystick_move(self, pos):
        self.current_command[2] = - pos[0] * (-self.ranges.ang_vel_z_range[0] if pos[0] < 0 else self.ranges.ang_vel_z_range[1])
        # if abs(pos[0]) + abs(pos[1]) > 0.05:
        #     print(f"Right joystick moved: {self.current_command}")

    def _on_up_press(self):
        """Handle up d-pad press"""
        self._vertical_keys_pressed.add(KeyMap.up)
        self.current_vertical_command = self.VERTICAL_SPEED
        # print(f"Up pressed: vertical={self.current_vertical_command}")

    def _on_down_press(self):
        """Handle down d-pad press"""
        self._vertical_keys_pressed.add(KeyMap.down)
        self.current_vertical_command = -self.VERTICAL_SPEED
        # print(f"Down pressed: vertical={self.current_vertical_command}")

    def _update_vertical_command(self):
        """Update vertical command based on currently pressed keys"""
        keys_pressed = self.remote_controller.get_keys_pressed()

        # Check if any vertical keys were released
        for key in [KeyMap.up, KeyMap.down]:
            if key in self._vertical_keys_pressed and key not in keys_pressed:
                self._vertical_keys_pressed.discard(key)
                # print(f"{key.name} released")

        # Update vertical command based on what's currently pressed
        if KeyMap.up not in keys_pressed and KeyMap.down not in keys_pressed:
            self.current_vertical_command = 0.0

    def _init_remote_controller_commands(self):
        from lib.deploy.control.remote_controller_cmd import RemoteControllerCommands
        remote_controller_commands = RemoteControllerCommands(self.remote_controller)

        for key, fn in self.additional_commands.items():
            remote_controller_commands.add_command(key, fn)
        # Add vertical commands
        for key, fn in self.vertical_commands.items():
            remote_controller_commands.add_command(key, fn)
        for joystick, fn in self.joystick_commands.items():
            remote_controller_commands.add_joystick_command(joystick, fn)
        remote_controller_commands.start()
        return remote_controller_commands

    def stop(self):
        """Stop the remote controller command listener"""
        if hasattr(self, 'remote_controller_commands'):
            self.remote_controller_commands.stop()

    def get_command(self):
        """Return the current command vector"""
        return copy.deepcopy(self.current_command)

    def get_vertical_command(self) -> float:
        """Return the current vertical command (positive = up, negative = down)"""
        self._update_vertical_command()
        return self.current_vertical_command


class ConstantCommander:
    def __init__(self, cmd: tp.Optional[np.ndarray] = None):
        if cmd is None:
            cmd = np.zeros(3, dtype=np.float32)
        self.cmd = cmd

    def set_cmd(self, cmd):
        self.cmd = cmd

    def get_command(self):
        return self.cmd.copy()


class MoveCommander(Commander):
    def __init__(self):
        self.command = np.zeros(3, dtype=np.float32)

    def Move(self, x, y, z):
        self.command[:] = (x, y, z)

    def get_command(self):
        return self.command
