"""Single source of default deploy paths and constants for YaCT OSS."""

from lib.vendor.mocap.constants import HeadingFrameSource

DEFAULT_CHECKPOINT = "checkpoints/default.pt"
DEFAULT_T_POSE_TRAJ = "trajectories/t_pose_traj.json"
DEFAULT_MUJOCO_XML = "mujoco/g1_29dof_rev_1_0_custom_actuators/g1_29dof_rev_1_0.xml"
DEFAULT_FK_URDF = "fk/g1_fk.urdf"

HEADING_FRAME_SOURCE = HeadingFrameSource.HEAD
