"""
VR pose types and helpers for teleoperation deploy.

Live teleop uses Vuer WebXR via ``VuerVRPoseTracker`` in ``vuer_vr_pose_tracker.py``.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Any
from pydantic import BaseModel, field_validator, field_serializer, ConfigDict
import numpy as np
from scipy.spatial.transform import Rotation as R
from lib.deploy.vr.common import VRControllerState


class VRPoseState(BaseModel):
    """
    Pydantic model containing head and wrist poses in global VR frame.

    Validates that poses are 4x4 matrices and handles serialization
    between numpy arrays and lists automatically.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _pose_fields = ['head_pose', 'left_wrist_pose', 'right_wrist_pose']

    head_pose: Optional[np.ndarray] = None
    left_wrist_pose: Optional[np.ndarray] = None
    right_wrist_pose: Optional[np.ndarray] = None

    head_timestamp: Optional[float] = None
    left_wrist_timestamp: Optional[float] = None
    right_wrist_timestamp: Optional[float] = None


    @field_validator(*_pose_fields, mode='before')
    @classmethod
    def validate_pose_matrix(cls, v: Any) -> Optional[np.ndarray]:
        """
        Validates input to ensure it is a (4,4) numpy array.
        Converts lists/lists-of-lists to numpy arrays automatically.
        """
        if v is None:
            return None

        if isinstance(v, np.ndarray):
            arr = v
        else:
            try:
                arr = np.array(v, dtype=float)
            except Exception as e:
                raise ValueError(f"Could not convert input to numpy array: {e}")

        if arr.shape != (4, 4):
            raise ValueError(f"Pose matrix must be shape (4,4), got {arr.shape}")

        return arr

    @field_serializer(*_pose_fields)
    def serialize_pose_matrix(self, v: Optional[np.ndarray], _info) -> Optional[List[List[float]]]:
        """
        Converts numpy arrays to lists when model_dump() is called.
        """
        if v is None:
            return None
        return v.tolist()


    def get_head_position(self) -> Optional[np.ndarray]:
        """Get head position as 3D vector."""
        if self.head_pose is not None:
            return self.head_pose[:3, 3]
        return None

    def get_head_rotation(self) -> Optional[R]:
        """Get head rotation as scipy Rotation object."""
        if self.head_pose is not None:
            return R.from_matrix(self.head_pose[:3, :3])
        return None

    def get_left_wrist_position(self) -> Optional[np.ndarray]:
        """Get left wrist position as 3D vector."""
        if self.left_wrist_pose is not None:
            return self.left_wrist_pose[:3, 3]
        return None

    def get_left_wrist_rotation(self) -> Optional[R]:
        """Get left wrist rotation as scipy Rotation object."""
        if self.left_wrist_pose is not None:
            return R.from_matrix(self.left_wrist_pose[:3, :3])
        return None

    def get_right_wrist_position(self) -> Optional[np.ndarray]:
        """Get right wrist position as 3D vector."""
        if self.right_wrist_pose is not None:
            return self.right_wrist_pose[:3, 3]
        return None

    def get_right_wrist_rotation(self) -> Optional[R]:
        """Get right wrist rotation as scipy Rotation object."""
        if self.right_wrist_pose is not None:
            return R.from_matrix(self.right_wrist_pose[:3, :3])
        return None

    def is_full(self) -> bool:
        return (
            (self.head_pose is not None)
            and (self.left_wrist_pose is not None)
            and (self.right_wrist_pose is not None)
        )

class BaseVRPoseTracker(ABC):
    """Abstract base class for VR pose trackers providing head and wrist poses."""

    @abstractmethod
    def get_state(self) -> VRPoseState:
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        pass

    @abstractmethod
    def get_controller_states(self) -> Tuple[Optional[VRControllerState], Optional[VRControllerState]]:
        pass
