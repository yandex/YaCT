"""
VR TCP Velocity Tracker interface.

Provides local-frame linear (m/s) and rotational (deg/s) velocities
for head, left wrist, and right wrist.
"""

from abc import ABC, abstractmethod
from typing import List, Any
import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, field_serializer


class Velocity(BaseModel):
    """Linear and rotational velocity of a single TCP."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    xyz: np.ndarray  # linear velocity, m/s, shape (3,)
    rotvec: np.ndarray  # rotational velocity, deg/s, shape (3,)

    @field_validator('xyz', 'rotvec', mode='before')
    @classmethod
    def validate_vec3(cls, v: Any) -> np.ndarray:
        if isinstance(v, np.ndarray):
            arr = v
        else:
            arr = np.array(v, dtype=float)
        if arr.shape != (3,):
            raise ValueError(f"Expected shape (3,), got {arr.shape}")
        return arr

    @field_serializer('xyz', 'rotvec')
    def serialize_vec3(self, v: np.ndarray, _info) -> List[float]:
        return v.tolist()


class VRTCPVelState(BaseModel):
    """Velocities for head, left wrist and right wrist in their local frames."""

    head: Velocity
    left_wrist: Velocity
    right_wrist: Velocity
    dt: float = 0.0  # time delta (seconds) over which velocities were computed


class VRTCPVelTracker(ABC):
    """Abstract base class for VR TCP velocity trackers."""

    @abstractmethod
    def get_state(self) -> VRTCPVelState:
        pass
