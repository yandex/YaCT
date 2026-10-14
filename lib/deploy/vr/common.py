
from dataclasses import dataclass
from pydantic import BaseModel, Field, field_validator, field_serializer, ConfigDict
from typing import Annotated, Optional
import numpy as np
from numpy.typing import NDArray, ArrayLike

SpatialTransformMatrix = ArrayLike


@dataclass
class VRControllerState:
    pose: SpatialTransformMatrix
    btn_a: bool
    btn_b: bool
    trigger_front: float  # [0, 1]. 0 is not pressed, 1 is fully pressed.
    trigger_side: float   # [0, 1]. 0 is not pressed, 1 is fully pressed.
    btn_j: bool
    axe_j_x: float        # [-1,1] - X coordinate of joystick, negative - left, positive - right
    axe_j_y: float        # [-1,1] - Y coordinate of joystick, negative - top, positive - bottom
    timestamp: float


@dataclass
class VRHeadState:
    pose: SpatialTransformMatrix
    timestamp: float


@dataclass
class VRState:
    left: VRControllerState
    right: VRControllerState
    head: VRHeadState


class Pose(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    pos: Annotated[NDArray[np.float32], Field(description="Position of the robot")]
    quat: Annotated[NDArray[np.float32], Field(description="Quaternion (w,x,y,z) of the robot")]

    @field_validator('pos', mode='before')
    @classmethod
    def validate_and_convert_pos(cls, v):
        arr = np.asarray(v, dtype=np.float32)
        if arr.shape != (3,):
            raise ValueError(f'Position must be (3,), got {arr.shape}')
        return arr

    @field_validator('quat', mode='before')
    @classmethod
    def validate_and_convert_quat(cls, v):
        arr = np.asarray(v, dtype=np.float32)
        if arr.shape != (4,):
            raise ValueError(f'Quaternion must be (4,), got {arr.shape}')
        return arr

    @field_serializer('pos', 'quat')
    def serialize_array(self, value: NDArray[np.float32]) -> list:
        return value.tolist()
