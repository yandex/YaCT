from typing import Dict, Sequence, Tuple, List
import pytorch_kinematics as pk
import numpy as np
from lib.vendor.mocap.constants import HEAD_OFFSET, IMU_PELVIS_OFFSET, HeadingFrameSource, FEET_LINKS
import torch
from lib.assets.paths import resolve_asset
import lib.vendor.util.math as math_utils
from pydantic import BaseModel, Field
from lib.deploy.configs.deploy_defaults import DEFAULT_FK_URDF


class KeypointConfig(BaseModel):
    anchor_link: str
    offset: Sequence[float] = Field(default_factory=lambda: [0, 0, 0])


class RobotModelConfig(BaseModel):
    pelvis_link: str = "pelvis"
    head_anchor_link: str = "torso_link"
    head_offset: Sequence[float] = Field(default_factory=lambda: HEAD_OFFSET)
    keypoints: List[KeypointConfig] = Field(default_factory=lambda: [
        KeypointConfig(anchor_link="left_wrist_yaw_link"),
        KeypointConfig(anchor_link="right_wrist_yaw_link"),
        KeypointConfig(anchor_link="torso_link", offset=HEAD_OFFSET),
    ])
    heading_frame_source: HeadingFrameSource = HeadingFrameSource.PELVIS
    robot_model_path: str = DEFAULT_FK_URDF
    device: str = "cuda:0"


class RobotModel:
    @torch.no_grad()
    def __init__(self, config: RobotModelConfig = RobotModelConfig()):
        self.config = config
        self.device = torch.device(config.device)
        self.robot_model_path = resolve_asset(config.robot_model_path)

        self.chain = pk.build_chain_from_urdf(open(self.robot_model_path).read()).to(device='cpu', dtype=torch.float32)

        self.link_names = sorted(self.chain.get_link_names())
        self.num_links = len(self.link_names)

        self.link_name_to_idx = {name: idx for idx, name in enumerate(self.link_names)}

        # Pre-calculate indices for feet links to avoid lookup during forward pass
        self.feet_link_ids = torch.tensor(
            [self.link_name_to_idx[name] for name in FEET_LINKS],
            device=self.device, dtype=torch.long
        )

        self.HEAD_OFFSET_T = torch.tensor(HEAD_OFFSET, device=self.device, dtype=torch.float32)
        self.IMU_PELVIS_OFFSET_T = torch.tensor(IMU_PELVIS_OFFSET, device=self.device, dtype=torch.float32)

        L = self.num_links

        self.keypoints = config.keypoints
        self.keypoint_anchor_link_ids = torch.tensor(
            [self.link_name_to_idx[kp.anchor_link] for kp in self.keypoints],
            device=self.device, dtype=torch.long
        )
        self.keypoint_offsets_l = torch.tensor([kp.offset for kp in self.keypoints], device=self.device, dtype=torch.float32)

        self.heading_quat_w = torch.tensor([1, 0, 0, 0], device=self.device, dtype=torch.float32)
        self.heading_pos_w = torch.zeros(3, device=self.device, dtype=torch.float32)
        self.pelvis_quat_w = torch.tensor([1, 0, 0, 0], device=self.device, dtype=torch.float32)
        self.pelvis_pos_w = torch.zeros(3, device=self.device, dtype=torch.float32)

        self.link_pos_b = torch.zeros(L, 3, device=self.device, dtype=torch.float32)
        self.link_quat_b = torch.tensor([1, 0, 0, 0], device=self.device, dtype=torch.float32).repeat(L, 1)
        self.link_pos_w = torch.zeros(L, 3, device=self.device, dtype=torch.float32)
        self.link_quat_w = torch.tensor([1, 0, 0, 0], device=self.device, dtype=torch.float32).repeat(L, 1)
        self.link_pos_h = torch.zeros(L, 3, device=self.device, dtype=torch.float32)
        self.link_quat_h = torch.tensor([1, 0, 0, 0], device=self.device, dtype=torch.float32).repeat(L, 1)

        self._cpu_pos_buf = np.zeros((L, 3), dtype=np.float32)
        self._cpu_rot_buf = np.zeros((L, 3, 3), dtype=np.float32)

        self.ground_z_w = None

    def get_link_index(self, link_name: str) -> int:
        return self.link_name_to_idx[link_name]

    @staticmethod
    def get_feet_sites_from_ankle(ankle_position_w, ankle_orientation_w):
        """Get foot sites from ankle position and orientation in world frame."""
        foot_l_pos = torch.tensor([0.04, 0, -0.029], device=ankle_position_w.device, dtype=ankle_position_w.dtype).expand(ankle_position_w.shape)
        foot_top_l_pos = torch.tensor([0.13, 0, -0.029], device=ankle_position_w.device, dtype=ankle_position_w.dtype).expand(ankle_position_w.shape)

        foot_w_pos = ankle_position_w + math_utils.quat_apply(ankle_orientation_w, foot_l_pos)
        foot_top_w_pos = ankle_position_w + math_utils.quat_apply(ankle_orientation_w, foot_top_l_pos)
        return (foot_w_pos, ankle_orientation_w), (foot_top_w_pos, ankle_orientation_w)

    @torch.no_grad()
    def get_heading_pos_quat_w(self):
        return self.heading_pos_w, self.heading_quat_w

    @torch.no_grad()
    def recompute_keypoint_pos_quat_h(self):
        keypoint_anchors_pos_h = self.link_pos_h[self.keypoint_anchor_link_ids]
        keypoint_anchors_quat_h = self.link_quat_h[self.keypoint_anchor_link_ids]
        keypoint_anchors_quat_w = self.link_quat_w[self.keypoint_anchor_link_ids]

        keypoint_offsets_w = math_utils.quat_apply(keypoint_anchors_quat_w, self.keypoint_offsets_l)
        keypoint_offsets_h = math_utils.quat_apply_inverse(self.heading_quat_w, keypoint_offsets_w)

        keypoint_pos_h = keypoint_anchors_pos_h + keypoint_offsets_h
        keypoint_quat_h = keypoint_anchors_quat_h

        self.keypoint_pos_h = keypoint_pos_h
        self.keypoint_quat_h = keypoint_quat_h

        return self.keypoint_pos_h.view(1, -1, 3), self.keypoint_quat_h.view(1, -1, 4)

    @torch.no_grad()
    def get_keypoint_pos_quat_h(self):
        return self.keypoint_pos_h.view(1, -1, 3), self.keypoint_quat_h.view(1, -1, 4)

    @torch.no_grad()
    def get_keypoint_pos_quat_w(self):
        keypoint_pos_h, keypoint_quat_h = self.get_keypoint_pos_quat_h()
        return math_utils.quat_apply(self.heading_quat_w, keypoint_pos_h) + self.heading_pos_w, \
            math_utils.quat_mul(self.heading_quat_w.expand_as(keypoint_quat_h), keypoint_quat_h)

    @torch.no_grad()
    def recompute_forward_kinematics(
        self,
        joint_positions: Dict[str, float | torch.Tensor] | torch.Tensor,
        head_quat_w: torch.Tensor,
        head_pos_w: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(joint_positions, torch.Tensor):
            fk_input = joint_positions.cpu().float()
        else:
            fk_input = joint_positions

        fk_result = self.chain.forward_kinematics(fk_input)

        L = self.num_links
        pos_buf = self._cpu_pos_buf  # shape (L,3)
        rot_buf = self._cpu_rot_buf  # shape (L,3,3)

        for i, link in enumerate(self.link_names):
            mat = fk_result[link].get_matrix()
            mat0 = mat[0]
            mat0_cpu = mat0.detach().cpu().numpy()

            pos_buf[i, :] = mat0_cpu[:3, 3]
            rot_buf[i, :, :] = mat0_cpu[:3, :3]

        rot_t = torch.from_numpy(rot_buf).to(device=self.device, dtype=torch.float32)  # (L,3,3)
        pos_t = torch.from_numpy(pos_buf).to(device=self.device, dtype=torch.float32)  # (L,3)

        quat_t = pk.matrix_to_quaternion(rot_t)

        self.link_pos_b.copy_(pos_t)
        self.link_quat_b.copy_(quat_t)

        head_quat_b = self.link_quat_b[self.link_name_to_idx[self.config.head_anchor_link]]
        pelvis_quat_w = math_utils.quat_mul(head_quat_w.view(4), math_utils.quat_inv(head_quat_b))
        self.head_pos_w = head_pos_w
        self.head_quat_w = head_quat_w
        head_anchor_pos_w = head_pos_w - math_utils.quat_apply(head_quat_w, self.HEAD_OFFSET_T)
        head_anchor_pelvis_offset_w = math_utils.quat_apply(
            pelvis_quat_w,
            self.link_pos_b[self.link_name_to_idx[self.config.head_anchor_link]] - self.link_pos_b[self.link_name_to_idx[self.config.pelvis_link]]
        )

        pelvis_pos_w = head_anchor_pos_w - head_anchor_pelvis_offset_w
        self.pelvis_pos_w = pelvis_pos_w
        self.pelvis_quat_w = pelvis_quat_w

        # CALC WORLD LINKS EARLY (Moved before heading logic)
        self.link_pos_w = math_utils.quat_apply(pelvis_quat_w, self.link_pos_b) + pelvis_pos_w
        self.link_quat_w = math_utils.quat_mul(pelvis_quat_w.expand_as(self.link_quat_b), self.link_quat_b)

        if self.config.heading_frame_source == HeadingFrameSource.PELVIS:
            self.heading_pos_w = pelvis_pos_w + math_utils.quat_apply(pelvis_quat_w, self.IMU_PELVIS_OFFSET_T)
            self.heading_quat_w = math_utils.calc_heading_quat(pelvis_quat_w)
        elif self.config.heading_frame_source == HeadingFrameSource.HEAD:
            self.heading_pos_w = head_pos_w
            self.heading_quat_w = math_utils.calc_heading_quat(head_quat_w)
        elif self.config.heading_frame_source == HeadingFrameSource.HEAD_Z08:
            # 1. Get raw feet link positions/orientations
            feet_pos_w = self.link_pos_w[self.feet_link_ids]
            feet_quat_w = self.link_quat_w[self.feet_link_ids]

            # 2. Derive feet sites (bottom/soles and top/toes)
            (foot_bottom_w, _), (foot_top_w, _) = self.get_feet_sites_from_ankle(feet_pos_w, feet_quat_w)

            # 3. Collect Z coordinates from: standard links, feet bottoms, feet tops
            z_candidates = torch.cat([
                self.link_pos_w[:, 2],  # All links
                foot_bottom_w[:, 2],    # Bottom of feet
                foot_top_w[:, 2]        # Top of feet (toes)
            ])

            # 4. Find ground (lowest point)
            ground_z = torch.min(z_candidates)
            self.ground_z_w = ground_z

            # 5. Set heading frame: Head X/Y, Ground Z + 0.8
            self.heading_pos_w = head_pos_w.clone()
            self.heading_pos_w[2] = ground_z + 0.8
            self.heading_quat_w = math_utils.calc_heading_quat(head_quat_w)
        else:
            raise ValueError(f"Unknown heading frame source: {self.config.heading_frame_source}")

        self.link_pos_h = math_utils.quat_apply_inverse(self.heading_quat_w, self.link_pos_w - self.heading_pos_w)
        self.link_quat_h = math_utils.quat_mul(math_utils.quat_inv(self.heading_quat_w).expand_as(self.link_quat_w), self.link_quat_w)

        self.recompute_keypoint_pos_quat_h()

        # get keypoints
        return self.get_keypoint_pos_quat_h()

    @torch.no_grad()
    def get_heading_frame_height(self, relative_to_ground: bool = False) -> torch.Tensor:
        """
        Get the z-coordinate (height) of the current heading frame.

        Args:
            relative_to_ground: If True, returns height relative to the lowest point (ground).
                            If False, returns absolute z-coordinate in world frame.
                            Note: relative_to_ground=True requires ground_z_w to be computed
                            (only available when using HeadingFrameSource.HEAD_Z08).

        Returns:
            torch.Tensor: Scalar tensor containing the height value.
        """
        if relative_to_ground:
            if self.ground_z_w is None:
                # Compute ground_z on demand if not already available
                feet_pos_w = self.link_pos_w[self.feet_link_ids]
                feet_quat_w = self.link_quat_w[self.feet_link_ids]
                (foot_bottom_w, _), (foot_top_w, _) = self.get_feet_sites_from_ankle(feet_pos_w, feet_quat_w)

                z_candidates = torch.cat([
                    self.link_pos_w[:, 2],
                    foot_bottom_w[:, 2],
                    foot_top_w[:, 2]
                ])
                ground_z = torch.min(z_candidates)
            else:
                ground_z = self.ground_z_w

            return self.heading_pos_w[2] - ground_z

        return self.heading_pos_w[2]
