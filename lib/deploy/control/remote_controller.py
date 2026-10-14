# Derived from Unitree RL Gym deploy_real
# (https://github.com/unitreerobotics/unitree_rl_gym), Copyright (c) 2016-2023
# HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics"), BSD 3-Clause. See NOTICES.
import struct
from enum import IntEnum, Enum
from typing import List, Tuple


class Joystick(Enum):
    left = 'left'
    right = 'right'


class KeyMap(IntEnum):
    R1 = 0
    L1 = 1
    start = 2
    select = 3
    R2 = 4
    L2 = 5
    F1 = 6
    F2 = 7
    A = 8
    B = 9
    X = 10
    Y = 11
    up = 12
    right = 13
    down = 14
    left = 15

class RemoteController:
    def __init__(self):
        self.lx = 0
        self.ly = 0
        self.rx = 0
        self.ry = 0
        self.button = [0] * 16

    def set(self, data):
        # wireless_remote
        keys = struct.unpack("H", data[2:4])[0]
        for i in range(16):
            self.button[i] = (keys & (1 << i)) >> i
        self.lx = struct.unpack("f", data[4:8])[0]
        self.rx = struct.unpack("f", data[8:12])[0]
        self.ry = struct.unpack("f", data[12:16])[0]
        self.ly = struct.unpack("f", data[20:24])[0]

    def get_joystick(self, joystick: Joystick) -> Tuple[float, float]:
        """
        Returns the joystick position in format (x, y).
        """

        if joystick == Joystick.left:
            return (self.lx, self.ly)
        elif joystick == Joystick.right:
            return (self.rx, self.ry)

    def get_keys_pressed(self) -> List[KeyMap]:
        """
        Returns a list of keys that are currently pressed.
        """

        keys_pressed = []
        for k in KeyMap:
            if self.button[k.value]:
                keys_pressed.append(k)
        return keys_pressed
