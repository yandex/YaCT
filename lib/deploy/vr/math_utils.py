import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


def pose_matrix_to_pos_quat(pose: NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Convert a 4x4 transformation matrix to position (3,) and quaternion (4,) in (w, x, y, z) format.
    """
    pos = pose[:3, 3].astype(np.float32)
    rot = pose[:3, :3]
    # scipy's as_quat with scalar_first=True returns (w, x, y, z)
    quat = Rotation.from_matrix(rot).as_quat(scalar_first=True).astype(np.float32)
    return pos, quat

def pos_quat_to_pose_matrix(pos: NDArray[np.float32], quat: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Convert position (3,) and quaternion (4,) in (w, x, y, z) format to a 4x4 transformation matrix.
    """
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = pos
    pose[:3, :3] = Rotation.from_quat(quat, scalar_first=True).as_matrix()
    return pose

def quat_mul_np(quat1: NDArray[np.float32], quat2: NDArray[np.float32], scalar_first: bool = True) -> NDArray[np.float32]:
    """
    Multiply two quaternions.
    """
    return (Rotation.from_quat(quat1, scalar_first=scalar_first) * Rotation.from_quat(quat2, scalar_first=scalar_first)).as_quat(scalar_first=scalar_first)
