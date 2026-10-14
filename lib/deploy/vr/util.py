import torch
import lib.vendor.util.math as math_utils
import numpy as np


# NumPy port of isaaclab.utils.math.quat_mul, Copyright (c) 2022-2025 The Isaac Lab
# Project Developers, BSD 3-Clause. See NOTICES.
def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions together.

    Args:
        q1: The first quaternion in (w, x, y, z). Shape is (..., 4).
        q2: The second quaternion in (w, x, y, z). Shape is (..., 4).

    Returns:
        The product of the two quaternions in (w, x, y, z). Shape is (..., 4).

    Raises:
        ValueError: Input shapes of ``q1`` and ``q2`` are not matching.
    """
    # check input is correct
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    # reshape to (N, 4) for multiplication
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    # extract components from quaternions
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    # perform multiplication
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    return np.stack([w, x, y, z], axis=-1).reshape(shape)

def calc_heading_from_rot_mat(rot_mat):
    """
    Calculate heading (yaw) angle from rotation matrix.
    Extracts yaw from rotation matrix using standard ZYX Euler angle extraction.
    """
    # Extract heading from rotation matrix elements
    # For a rotation matrix, yaw = atan2(R[1,0], R[0,0])
    heading = np.arctan2(rot_mat[..., 1, 0], rot_mat[..., 0, 0])

    return heading


def rot_mat_from_heading(heading):
    """
    Create rotation matrix from heading angle (rotation around Z-axis).
    """
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)

    # Create rotation matrix for rotation around Z-axis
    batch_shape = heading.shape
    h_rot = np.zeros((*batch_shape, 3, 3), dtype=heading.dtype)

    h_rot[..., 0, 0] = cos_h
    h_rot[..., 0, 1] = -sin_h
    h_rot[..., 1, 0] = sin_h
    h_rot[..., 1, 1] = cos_h
    h_rot[..., 2, 2] = 1

    return h_rot

def calc_heading_from_rotation(rot_mat):
    """
    Extract heading (yaw-only rotation) from a rotation matrix.
    Projects rotation onto the horizontal plane.
    """
    heading = calc_heading_from_rot_mat(rot_mat)
    h_rot = rot_mat_from_heading(heading)
    return h_rot

def calc_heading(q):
    """
    Calculate heading (yaw) angle from quaternion.
    Assumes quaternion format is [w, x, y, z]
    """
    # Extract quaternion components [w, x, y, z]
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Calculate yaw angle from quaternion
    # yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    heading = np.arctan2(siny_cosp, cosy_cosp)

    return heading

def quat_from_angle_axis(angle, axis):
    """
    Create quaternion from angle and axis.
    Returns quaternion in [w, x, y, z] format
    """
    # Normalize axis
    axis_norm = axis / (np.linalg.norm(axis, axis=-1, keepdims=True) + 1e-8)

    # Create quaternion: q = [cos(angle/2), sin(angle/2) * axis]
    half_angle = angle * 0.5
    sin_half = np.sin(half_angle)
    cos_half = np.cos(half_angle)

    quat = np.zeros((*angle.shape, 4), dtype=angle.dtype)
    quat[..., 0] = cos_half                        # w
    quat[..., 1] = axis_norm[..., 0] * sin_half   # x
    quat[..., 2] = axis_norm[..., 1] * sin_half   # y
    quat[..., 3] = axis_norm[..., 2] * sin_half   # z

    return quat

def calc_heading_quat(q):
    """
    Calculate heading quaternion (yaw-only rotation) from full quaternion.
    """
    heading = calc_heading(q)
    axis = np.zeros_like(q[..., 1:4])  # [x, y, z] components
    axis[..., 2] = 1
    heading_q = quat_from_angle_axis(heading, axis)
    return heading_q

def quat_apply(quat, vec):
    """
    Apply quaternion rotation to a vector.
    Assumes quaternion format is [w, x, y, z]
    """
    # Extract quaternion components [w, x, y, z]
    quat_w = quat[..., 0:1]
    quat_xyz = quat[..., 1:4]

    # Compute rotation using quaternion formula: v' = v + 2 * r x (r x v + w * v)
    # where r is the vector part (xyz) and w is the scalar part
    t = 2 * np.cross(quat_xyz, vec)
    rotated = vec + quat_w * t + np.cross(quat_xyz, t)

    return rotated

def heading_frame_from_head(h_pos, h_quat, heading_frame: str = 'pelvis'):
    """
    Calculate heading frame from head position and quaternion.
    Assumes quaternion format is [w, x, y, z]
    """
    h_ori = calc_heading_quat(h_quat)
    if heading_frame == 'pelvis':
        offset_local = np.array([0.0695, -0.0000, -0.5176], dtype=h_pos.dtype)
        offset_local = np.broadcast_to(offset_local, h_pos.shape)
        offset_world = quat_apply(h_quat, offset_local)
    elif heading_frame == 'head':
        offset_world = np.zeros_like(h_pos)
    elif heading_frame == 'head_z08':
        offset_world = np.array([0.0, 0.0, -0.4], dtype=h_pos.dtype)
    return h_pos + offset_world, h_ori

def heading_frame_from_head_pose(h_pose):
    """
    Calculate heading frame from head pose matrix.

    Args:
        h_pose: 4x4 pose matrix or (..., 4, 4) batch of pose matrices

    Returns:
        4x4 pose matrix with heading frame
    """
    # Extract position (translation) from pose
    h_pos = h_pose[..., :3, 3]

    # Extract rotation matrix from pose
    h_rot = h_pose[..., :3, :3]

    # Calculate heading orientation
    h_ori_rot = calc_heading_from_rotation(h_rot)

    # Define local offset
    offset_local = np.array([0.0695, -0.0000, -0.5176], dtype=h_pose.dtype)

    # Apply rotation to offset: offset_world = R @ offset_local
    offset_world = np.einsum('...ij,j->...i', h_rot, offset_local)

    # New position
    new_pos = h_pos + offset_world

    # Construct output pose matrix
    out_pose = np.zeros_like(h_pose)
    out_pose[..., :3, :3] = h_ori_rot
    out_pose[..., :3, 3] = new_pos
    out_pose[..., 3, 3] = 1
    return out_pose


def random_quaternion_simple(size=None, dtype=np.float32):
    """
    Generate random quaternion(s) in [w, x, y, z] format.
    """
    if size is None:
        shape = (4,)
    elif isinstance(size, int):
        shape = (size, 4)
    else:
        shape = (*size, 4)
    # Generate random values and normalize
    quat = np.random.randn(*shape).astype(dtype)
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)

    return quat

from scipy.spatial.transform import Rotation

import numpy as np
from scipy.spatial.transform import Rotation as R
def frame2_in_frame1(pos1, quat1, pos2, quat2):
    """
    Transform frame 2's pose into frame 1's coordinate system.
    Supports batch operations.

    Parameters:
    -----------
    pos1 : array-like, shape (3,) or (N, 3)
        Position(s) of frame 1 in world coordinates [x, y, z]
    quat1 : array-like, shape (4,) or (N, 4)
        Quaternion(s) of frame 1 in world coordinates [w, x, y, z]
    pos2 : array-like, shape (3,) or (N, 3)
        Position(s) of frame 2 in world coordinates [x, y, z]
    quat2 : array-like, shape (4,) or (N, 4)
        Quaternion(s) of frame 2 in world coordinates [w, x, y, z]

    Returns:
    --------
    pos_rel : ndarray, shape (3,) or (N, 3)
        Position(s) of frame 2 in frame 1's coordinate system
    quat_rel : ndarray, shape (4,) or (N, 4)
        Quaternion(s) of frame 2 in frame 1's coordinate system [w, x, y, z]

    Notes:
    ------
    All inputs can be batched or broadcast together. For example:
    - Single frame1, multiple frame2: pos1=(3,), quat1=(4,), pos2=(N,3), quat2=(N,4)
    - Multiple frame1, single frame2: pos1=(N,3), quat1=(N,4), pos2=(3,), quat2=(4,)
    - Multiple both: all shapes (N, ...)
    """
    # Convert to numpy arrays
    pos1 = np.asarray(pos1)
    pos2 = np.asarray(pos2)
    quat1 = np.asarray(quat1)
    quat2 = np.asarray(quat2)

    # Convert quaternions to Rotation objects using scalar_first=True for [w,x,y,z]
    rot1 = R.from_quat(quat1, scalar_first=True)
    rot2 = R.from_quat(quat2, scalar_first=True)

    # Compute relative rotation: rot_rel = rot1^(-1) * rot2
    rot_rel = rot1.inv() * rot2
    quat_rel = rot_rel.as_quat(scalar_first=True)

    # Compute relative position: pos_rel = rot1^(-1).apply(pos2 - pos1)
    pos_rel = rot1.inv().apply(pos2 - pos1)

    return pos_rel, quat_rel


def apply_transform_from_new_base(pos, quat, old_base_pos, old_base_quat, new_base_pos, new_base_quat):
    """
    Transform a frame from one base coordinate system to another.
    Supports batch operations.

    Computes: frame_coord_2 = new_base + (frame - old_base)

    Parameters:
    -----------
    pos : array-like, shape (3,) or (N, 3)
        Position(s) of the frame in world coordinates [x, y, z]
    quat : array-like, shape (4,) or (N, 4)
        Quaternion(s) of the frame in world coordinates [w, x, y, z]
    old_base_pos : array-like, shape (3,) or (N, 3)
        Position(s) of the old base frame [x, y, z]
    old_base_quat : array-like, shape (4,) or (N, 4)
        Quaternion(s) of the old base frame [w, x, y, z]
    new_base_pos : array-like, shape (3,) or (N, 3)
        Position(s) of the new base frame [x, y, z]
    new_base_quat : array-like, shape (4,) or (N, 4)
        Quaternion(s) of the new base frame [w, x, y, z]

    Returns:
    --------
    pos_new : ndarray, shape (3,) or (N, 3)
        Position(s) of the frame relative to the new base
    quat_new : ndarray, shape (4,) or (N, 4)
        Quaternion(s) of the frame relative to the new base [w, x, y, z]
    """
    # Step 1: Get relative transform from old_base to frame
    pos_rel, quat_rel = frame2_in_frame1(old_base_pos, old_base_quat, pos, quat)

    # Step 2: Apply that relative transform to new_base
    new_base_quat = np.asarray(new_base_quat)
    new_base_pos = np.asarray(new_base_pos)

    rot_new_base = R.from_quat(new_base_quat, scalar_first=True)
    rot_rel = R.from_quat(quat_rel, scalar_first=True)

    # New rotation: rot_new = rot_new_base * rot_rel
    rot_new = rot_new_base * rot_rel
    quat_new = rot_new.as_quat(scalar_first=True)

    # New position: pos_new = new_base_pos + rot_new_base.apply(pos_rel)
    pos_new = new_base_pos + rot_new_base.apply(pos_rel)

    return pos_new, quat_new


def transform_velocity_from_new_base(lin_vel, ang_vel, old_base_quat, new_base_quat):
    """
    Transform linear and angular velocities from one base frame to another.
    Supports batch operations.

    Parameters:
    -----------
    lin_vel : array-like, shape (3,) or (N, 3)
        Linear velocity/velocities in old base frame
    ang_vel : array-like, shape (3,) or (N, 3)
        Angular velocity/velocities in old base frame
    old_base_quat : array-like, shape (4,) or (N, 4)
        Quaternion(s) of the old base frame [w, x, y, z]
    new_base_quat : array-like, shape (4,) or (N, 4)
        Quaternion(s) of the new base frame [w, x, y, z]

    Returns:
    --------
    lin_vel_new : ndarray, shape (3,) or (N, 3)
        Linear velocity/velocities in new base frame
    ang_vel_new : ndarray, shape (3,) or (N, 3)
        Angular velocity/velocities in new base frame

    Notes:
    ------
    Velocities are rotated by: R_new * R_old^(-1)
    This assumes the velocities are expressed in their respective body frames.
    """
    lin_vel = np.asarray(lin_vel)
    ang_vel = np.asarray(ang_vel)
    old_base_quat = np.asarray(old_base_quat)
    new_base_quat = np.asarray(new_base_quat)

    # Get the relative rotation from old base to new base
    rot_old = R.from_quat(old_base_quat, scalar_first=True)
    rot_new = R.from_quat(new_base_quat, scalar_first=True)

    # Rotation to transform velocities: R_new * R_old^(-1)
    rot_transform = rot_new * rot_old.inv()

    # Apply rotation to both linear and angular velocities
    lin_vel_new = rot_transform.apply(lin_vel)
    ang_vel_new = rot_transform.apply(ang_vel)

    return lin_vel_new, ang_vel_new


def test():
    print("=" * 80)
    print("Testing quat_mul")
    print("=" * 80)

    # Generate random quaternions
    np.random.seed(42)
    rand_q1 = random_quaternion_simple(10)
    rand_q2 = random_quaternion_simple(10)

    # NumPy implementation
    result_np = quat_mul(rand_q1, rand_q2)

    # PyTorch ground truth
    q1_torch = torch.tensor(rand_q1).float()
    q2_torch = torch.tensor(rand_q2).float()
    result_gt = math_utils.quat_mul(q1_torch, q2_torch)

    print(f"NumPy result (first 3):")
    print(f"  {result_np[:3]}")
    print(f"PyTorch GT result (first 3):")
    print(f"  {result_gt[:3].numpy()}")

    quat_diff = np.linalg.norm(result_np - result_gt.numpy())
    print(f"Quaternion difference (L2 norm): {quat_diff:.6e}")

    assert quat_diff < 1e-5, f"Quaternion mismatch: {quat_diff}"
    print("✓ quat_mul test passed!\n")

    # Test with different shapes
    print("Testing with batched shapes...")
    rand_q1_batched = random_quaternion_simple((2, 5))
    rand_q2_batched = random_quaternion_simple((2, 5))

    result_np_batched = quat_mul(rand_q1_batched, rand_q2_batched)
    result_gt_batched = math_utils.quat_mul(
        torch.tensor(rand_q1_batched).float(),
        torch.tensor(rand_q2_batched).float()
    )

    batched_diff = np.linalg.norm(result_np_batched - result_gt_batched.numpy())
    print(f"Batched quaternion difference (L2 norm): {batched_diff:.6e}")

    assert batched_diff < 1e-5, f"Batched quaternion mismatch: {batched_diff}"
    print("✓ Batched quat_mul test passed!\n")

    # Test shape mismatch error
    print("Testing shape mismatch error...")
    try:
        quat_mul(random_quaternion_simple(5), random_quaternion_simple(3))
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  Correctly raised ValueError: {e}")
    print("✓ Shape mismatch test passed!\n")


    rand_pos = np.random.normal(size=(3,))
    rand_quat = random_quaternion_simple()

    print("=" * 80)
    print("Testing heading_frame_from_head (quaternion version)")
    print("=" * 80)

    pos_h, quat_h = heading_frame_from_head(rand_pos, rand_quat)

    def calc_heading_quat(q):
        heading = calc_heading(q)
        axis = torch.zeros_like(q[..., 0:3])
        axis[..., 2] = 1
        heading_q = math_utils.quat_from_angle_axis(heading, axis)
        return heading_q

    def heading_frame_from_head_gt(h_pos, h_quat):
        h_ori = calc_heading_quat(h_quat)
        offset_local= torch.tensor([0.0695, -0.0000, -0.5176], device=h_pos.device, dtype=h_pos.dtype).expand(h_pos.shape)
        offset_world = math_utils.quat_apply(h_quat, offset_local)
        return h_pos + offset_world, h_ori

    gt_pos_h, gt_quat_h = heading_frame_from_head_gt(
        torch.tensor(rand_pos).float(), torch.tensor(rand_quat).float()
    )

    print(f"NumPy result:")
    print(f"  Position: {pos_h}")
    print(f"  Quaternion: {quat_h}")
    print(f"PyTorch GT result:")
    print(f"  Position: {gt_pos_h}")
    print(f"  Quaternion: {gt_quat_h}")

    pos_diff = np.linalg.norm(pos_h - gt_pos_h.numpy())
    quat_diff = np.linalg.norm(quat_h - gt_quat_h.numpy())
    print(f"Position difference: {pos_diff:.6e}")
    print(f"Quaternion difference: {quat_diff:.6e}")

    assert pos_diff < 1e-5, f"Position mismatch: {pos_diff}"
    assert quat_diff < 1e-5, f"Quaternion mismatch: {quat_diff}"
    print("✓ Quaternion version test passed!\n")

    print("=" * 80)
    print("Testing heading_frame_from_head_pose (pose matrix version)")
    print("=" * 80)

    # Convert quaternion [w, x, y, z] to rotation matrix using scipy
    # Scipy expects [x, y, z, w] format, so we need to reorder
    quat_scipy = np.array([rand_quat[1], rand_quat[2], rand_quat[3], rand_quat[0]])  # [w,x,y,z] -> [x,y,z,w]
    rot_matrix = Rotation.from_quat(quat_scipy).as_matrix()

    # Create 4x4 pose matrix (numpy)
    pose_np = np.eye(4, dtype=np.float32)
    pose_np[:3, :3] = rot_matrix.astype(np.float32)
    pose_np[:3, 3] = rand_pos

    # Create 4x4 pose matrix (torch)
    pose_torch = torch.eye(4, dtype=torch.float32)
    quat_torch = torch.tensor(rand_quat).float()
    pos_torch = torch.tensor(rand_pos).float()
    pose_torch[:3, :3] = math_utils.matrix_from_quat(quat_torch)
    pose_torch[:3, 3] = pos_torch

    print(f"Input pose matrix (NumPy):")
    print(pose_np)
    print(f"\nInput pose matrix (PyTorch):")
    print(pose_torch)

    # Test numpy version
    result_pose_np = heading_frame_from_head_pose(pose_np)

    # Create ground truth using torch
    def heading_frame_from_head_pose_gt(h_pose):
        h_pos = h_pose[:3, 3]
        h_rot = h_pose[:3, :3]

        # Convert rotation matrix to quaternion
        h_quat = math_utils.quat_from_matrix(h_rot)

        # Calculate heading quaternion
        h_ori_quat = calc_heading_quat(h_quat)

        # Convert back to rotation matrix
        h_ori_rot = math_utils.matrix_from_quat(h_ori_quat)

        # Apply offset
        offset_local = torch.tensor([0.0695, -0.0000, -0.5176], dtype=h_pose.dtype)
        offset_world = h_rot @ offset_local
        new_pos = h_pos + offset_world

        # Construct output pose
        out_pose = torch.eye(4, dtype=h_pose.dtype)
        out_pose[:3, :3] = h_ori_rot
        out_pose[:3, 3] = new_pos

        return out_pose

    result_pose_torch = heading_frame_from_head_pose_gt(pose_torch)

    print(f"\nNumPy result pose:")
    print(result_pose_np)
    print(f"\nPyTorch GT result pose:")
    print(result_pose_torch)

    # Compare positions
    pos_np = result_pose_np[:3, 3]
    pos_torch = result_pose_torch[:3, 3].numpy()
    pos_pose_diff = np.linalg.norm(pos_np - pos_torch)

    # Compare rotation matrices
    rot_np = result_pose_np[:3, :3]
    rot_torch = result_pose_torch[:3, :3].numpy()
    rot_pose_diff = np.linalg.norm(rot_np - rot_torch)

    print(f"\nPosition difference: {pos_pose_diff:.6e}")
    print(f"Rotation matrix difference: {rot_pose_diff:.6e}")

    assert pos_pose_diff < 1e-5, f"Pose position mismatch: {pos_pose_diff}"
    assert rot_pose_diff < 1e-5, f"Pose rotation mismatch: {rot_pose_diff}"
    print("✓ Pose matrix version test passed!\n")

    # Verify consistency between quaternion and pose versions
    print("=" * 80)
    print("Verifying consistency between quaternion and pose versions")
    print("=" * 80)

    consistency_pos_diff = np.linalg.norm(pos_h - pos_np)
    print(f"Position difference between methods: {consistency_pos_diff:.6e}")
    assert consistency_pos_diff < 1e-5, f"Position consistency mismatch: {consistency_pos_diff}"

    # Compare quaternion result with rotation matrix result
    # Convert quat_h [w, x, y, z] to rotation matrix
    quat_h_scipy = np.array([quat_h[1], quat_h[2], quat_h[3], quat_h[0]])  # [w,x,y,z] -> [x,y,z,w]
    quat_as_rot = Rotation.from_quat(quat_h_scipy).as_matrix()

    consistency_rot_diff = np.linalg.norm(quat_as_rot - rot_np)
    print(f"Rotation difference between methods: {consistency_rot_diff:.6e}")
    assert consistency_rot_diff < 1e-5, f"Rotation consistency mismatch: {consistency_rot_diff}"

    print("✓ Consistency check passed!")
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)


if __name__ == '__main__':
    test()
