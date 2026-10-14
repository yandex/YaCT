"""Mocap / robot frame constants."""

from enum import Enum

FEET_LINKS = ["left_ankle_roll_link", "right_ankle_roll_link"]

HEAD_OFFSET = [0.0, 0.0, 0.4]
IMU_PELVIS_OFFSET = [0.045249998569488525, 0.0, -0.0833899974822998]


class HeadingFrameSource(Enum):
    PELVIS = "pelvis"
    HEAD = "head"
    HEAD_Z08 = "head_z08"
