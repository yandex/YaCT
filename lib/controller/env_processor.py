from collections import deque
from datetime import datetime
import logging
import shutil
from lib.deploy.debug.profiler import Profiler
from lib.deploy.deploy_utils.common import Timer, BaseEnvProcessor
from lib.deploy.deploy_utils.forward_kinematics import RobotModel, RobotModelConfig, HeadingFrameSource
import torch
import json
import os
import typing as tp
from lib.deploy.configs.joint_mappings import (
    REAL_ORDER,
    SIMULATION_ORDER,
    SIM_DEFAULT_POSITIONS,
)
from lib.deploy.vr.localization import LocalizationObservator
from lib.deploy.vr.vr_observations import VrHeadingObservator
import lib.vendor.util.math as math_utils

torch.set_printoptions(precision=4, sci_mode=False)

HISTORY = int(os.environ.get("HISTORY", "20"))
DIF_QUAT_MODE = os.environ.get("DIF_QUAT_MODE", "6d")
REF_QUAT_MODE = os.environ.get("REF_QUAT_MODE", "6d")

@torch.jit.script
def _quat_to_6d(q: torch.Tensor) -> torch.Tensor:
    """Convert (w, x, y, z) quaternion to 6D rotation representation."""
    w, x, y, z = q.unbind(-1)
    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return torch.stack([
        1 - 2 * (y2 + z2), 2 * (xy - wz),
        2 * (xy + wz), 1 - 2 * (x2 + z2),
        2 * (xz - wy), 2 * (yz + wx),
    ], dim=-1)


@torch.jit.script
def _quat_to_axis_angle(q: torch.Tensor) -> torch.Tensor:
    """Convert (w, x, y, z) quaternion to axis-angle representation."""
    xyz = q[..., 1:]
    sin_half = torch.norm(xyz, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, q[..., :1])
    scale = torch.where(sin_half > 1e-6, angle / sin_half, 2.0 * torch.ones_like(angle))
    return xyz * scale


@torch.jit.script
def _convert_rotation(q: torch.Tensor, mode: str) -> torch.Tensor:
    """Convert quaternion to specified rotation representation."""
    if mode == '6d':
        return _quat_to_6d(q)
    elif mode == 'aa':
        return _quat_to_axis_angle(q)
    return q


@torch.jit.script
def fused_calc_diffs(
    vr_pos_h: torch.Tensor,       # [3, 3] - Head, L_Wrist, R_Wrist
    vr_quat_h: torch.Tensor,      # [3, 4]
    vr_vel_lin: torch.Tensor,     # [3, 3]
    kp_pos_h: torch.Tensor,       # [3, 3] or [1, 3, 3] - L_Wrist, R_Wrist, Head
    kp_quat_h: torch.Tensor,      # [3, 4] or [1, 3, 4]
    dif_mode: str,
    ref_mode: str
) -> tp.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused kernel for computing pose differences and targets."""

    # VR reference: [Head, L_Wrist, R_Wrist]
    ref_head_pos, ref_wrists_pos = vr_pos_h[:1], vr_pos_h[1:]
    ref_head_quat, ref_wrists_quat = vr_quat_h[:1], vr_quat_h[1:]

    # Robot current: [L_Wrist, R_Wrist, Head] - squeeze batch dim if present
    kp_pos = kp_pos_h[0] if kp_pos_h.ndim == 3 else kp_pos_h
    kp_quat = kp_quat_h[0] if kp_quat_h.ndim == 3 else kp_quat_h
    curr_head_pos, curr_wrists_pos = kp_pos[2:], kp_pos[:2]
    curr_head_quat, curr_wrists_quat = kp_quat[2:], kp_quat[:2]

    # Pose differences: delta_q = q_curr^-1 * q_ref
    head_q_diff = math_utils.quat_mul(math_utils.quat_inv(curr_head_quat), ref_head_quat)
    wrists_q_diff = math_utils.quat_mul(math_utils.quat_inv(curr_wrists_quat), ref_wrists_quat)

    head_diff = torch.cat([ref_head_pos - curr_head_pos, _convert_rotation(head_q_diff, dif_mode)], dim=-1)
    wrists_diff = torch.cat([ref_wrists_pos - curr_wrists_pos, _convert_rotation(wrists_q_diff, dif_mode)], dim=-1)

    # Reference targets
    head_target = torch.cat([ref_head_pos, _convert_rotation(ref_head_quat, ref_mode)], dim=-1)
    wrists_target = torch.cat([ref_wrists_pos, _convert_rotation(ref_wrists_quat, ref_mode)], dim=-1)

    return (
        head_diff,                   # [1, D]
        wrists_diff.unsqueeze(0),    # [1, 2, D]
        head_target,                 # [1, D]
        wrists_target.unsqueeze(0),  # [1, 2, D]
        vr_vel_lin[:1],              # [1, 3]
        vr_vel_lin[1:].unsqueeze(0), # [1, 2, 3]
    )


class EnvProcessor(BaseEnvProcessor):
    def __init__(
            self,
            vr_observator: VrHeadingObservator,
            localization_observator: LocalizationObservator,
            robot_model: RobotModel,
            device: str = 'cuda:0',
            history_length: int = HISTORY,
            robot_heading_frame_source: HeadingFrameSource = HeadingFrameSource.PELVIS,
            log_level: tp.Optional[int] = None,
            log_dir: str = "./logs",
            log_file_prefix: str = "env_processor",
            profiler: tp.Optional[Profiler] = None,
            record_vr: bool = False,
            ):
        super().__init__(
            observation_keys=[
                'ang_vel',
                'gravity_orientation',
                'qj',
                'dqj',
                'last_motor_targets',
                'last_actions',
                'head_pos_quat_diff_h',
                'wrists_pos_quat_diff_h',
                'ref_head_pos_quat_vec_h',
                'ref_wrists_pos_quat_vec_h',
                'head_vel_h',
                'wrists_vel_h'
            ],
            device=device,
            history_length=history_length,
            log_level=log_level,
            log_dir=log_dir,
            log_file_prefix=log_file_prefix,
            profiler=profiler
        )
        self.is_debug = log_level == logging.DEBUG
        self.log_dir = log_dir
        self.record_vr = record_vr
        self.vr_observator = vr_observator
        self.localization_observator = localization_observator
        self.device = device
        self.history_length = history_length
        self.carry_state = None
        self.num_motors = len(REAL_ORDER)
        self.robot_model = robot_model

        # Cache for VR observation (populated once per step)
        self._cached_vr_obs = None
        self._cached_vr_tensors = None
        self.vr_states = deque(maxlen=10000)
        self.vr_jsonl_log_path = os.path.join(self.log_dir, f"vr_log_inc_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jsonl")

        self.kp_pos_h, self.kp_quat_h = None, None
        self._last_vr_obs = None
        self._last_motor_target = None

    def _log_vr_observations(self):
        if not self.is_debug and not self.record_vr:
            return
        if hasattr(self.vr_observator, 'observator'):
            state = self.vr_observator.observator.state
            self.vr_states.append(state.model_dump())
            with open(self.vr_jsonl_log_path, 'a') as f:
                f.write(json.dumps(state.model_dump()) + '\n')

    def save_vr_log(self, name: str):
        if not self.is_debug and not self.record_vr:
            return
        ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        vr_jsonl_log_path = os.path.join(self.log_dir, f"{name}_vr_log_inc_{ts}.jsonl")

        if os.path.exists(self.vr_jsonl_log_path):
            shutil.copyfile(self.vr_jsonl_log_path, vr_jsonl_log_path)
            os.remove(self.vr_jsonl_log_path)
        log = {
            "recorded_states": list(self.vr_states),
            "num_states": len(self.vr_states),
        }

        vr_log_path = os.path.join(self.log_dir, f"{name}_vr_{ts}.json")

        print(f"Saving VR log to {vr_log_path}")
        with open(vr_log_path, 'w') as f:
            json.dump(log, f)
        self.vr_states.clear()
        print(f"VR log saved to {vr_log_path}")

    def process_diff(self, quat_diff, quat_mode: str = DIF_QUAT_MODE):
        if quat_mode == 'q':
            return quat_diff
        if quat_mode == '6d':
            return math_utils.quaternion_to_6d(quat_diff)
        if quat_mode == 'aa':
            return math_utils.axis_angle_from_quat(quat_diff)
        raise ValueError(f"{quat_mode=}")

    def _clear_vr_observation_cache(self):
        """Clear the cached VR observation."""
        self._cached_vr_obs = None
        self._cached_vr_tensors = None

    def get_ref_pos_quat_vr_h_v2(self):
        # Use cached observation if available, otherwise fetch new one
        if self._cached_vr_obs is not None:
            obs = self._cached_vr_obs
        else:
            obs = self.vr_observator.get_observation()

        pose = obs.head_wrists_pose_h

        velocity = obs.head_wrists_vel_h
        target_pos_h = torch.cat([torch.tensor(pose.pos, device=self.device).view(-1, 3)])
        target_quat_h = torch.cat([torch.tensor(pose.quat, device=self.device).view(-1, 4)])

        target_pos_w = math_utils.quat_apply(self.robot_model.heading_quat_w, target_pos_h) + self.robot_model.heading_pos_w
        target_quat_w = math_utils.quat_mul(self.robot_model.heading_quat_w.expand_as(target_quat_h), target_quat_h)

        if self.is_debug:
            self.state_recorder.set_target_keypoint_pos_w(obs.head_wrists_pose_w.pos.reshape(-1, 3))
            self.state_recorder.set_target_keypoint_quat_w(obs.head_wrists_pose_w.quat.reshape(-1, 4))
        return torch.tensor(pose.pos, device=self.device), \
            torch.tensor(pose.quat, device=self.device), \
                torch.tensor(velocity.lin_vel, device=self.device)

    def vr_get_head_pos_quat_diff_h(self, quat_mode: str = DIF_QUAT_MODE):
        ref_wrists_hlr, ref_quat_hlr, _ = self.get_ref_pos_quat_vr_h_v2()

        ref_wrists_h = ref_wrists_hlr[:1]
        ref_quat_h = ref_quat_hlr[:1]

        keypoint_pos_h, keypoint_quat_h = self.robot_model.get_keypoint_pos_quat_h()
        wrists_h, quat_h = keypoint_pos_h[:, 2], keypoint_quat_h[:, 2]
        quat_diff = math_utils.quat_mul(
            math_utils.quat_inv(quat_h), ref_quat_h
        )
        quat_diff = self.process_diff(quat_diff, quat_mode)
        out = torch.cat([ref_wrists_h - wrists_h, quat_diff], dim=-1)

        if self.is_debug:
            self.state_recorder.set_diff_pos_head((ref_wrists_h - wrists_h).view(3).cpu().numpy())
            self.state_recorder.set_diff_quat_head((self.process_diff(math_utils.quat_mul(
                math_utils.quat_inv(quat_h), ref_quat_h
            ), 'aa')).view(3).cpu().numpy())
        self.debug_log(f"Head diff: {out.tolist()}")
        return out

    def vr_get_wrists_pos_quat_diff_h(self, quat_mode: str = DIF_QUAT_MODE):
        ref_wrists_hlr, ref_quat_hlr, _ = self.get_ref_pos_quat_vr_h_v2()
        ref_wrists_h = ref_wrists_hlr[None, 1:]
        ref_quat_h = ref_quat_hlr[None, 1:]

        keypoint_pos_h, keypoint_quat_h = self.robot_model.get_keypoint_pos_quat_h()
        wrists_h, quat_h = keypoint_pos_h[:, 0:2], keypoint_quat_h[:, 0:2]
        wrists_h, quat_h = wrists_h.to(self.device), quat_h.to(self.device)

        quat_diff = math_utils.quat_mul(
            math_utils.quat_inv(quat_h), ref_quat_h
        )
        quat_diff = self.process_diff(quat_diff, quat_mode)
        out = torch.cat([ref_wrists_h - wrists_h, quat_diff], dim=-1)
        if self.is_debug:
            self.state_recorder.set_diff_pos_wrists((ref_wrists_h - wrists_h).view(-1, 3).cpu().numpy())
            self.state_recorder.set_diff_quat_wrists((self.process_diff(math_utils.quat_mul(
                math_utils.quat_inv(quat_h), ref_quat_h
            ), 'aa')).view(-1, 3).cpu().numpy())

        self.debug_log(f"Wrists diff: {out.tolist()}")
        return out

    def vr_get_ref_wrists_pos_quat_vec_h(self, quat_mode: str = REF_QUAT_MODE):
        ref_wrists_hlr, ref_quat_hlr, _ = self.get_ref_pos_quat_vr_h_v2()
        ref_wrists_pos_h = ref_wrists_hlr[None, 1:]
        ref_wrists_quat_h = ref_quat_hlr[None, 1:]
        ref_wrists_quat_h = self.process_diff(ref_wrists_quat_h, quat_mode=quat_mode)
        out = torch.cat([ref_wrists_pos_h, ref_wrists_quat_h], dim=-1)
        self.debug_log(f"Wrists target: {out.tolist()}")
        return out

    def vr_get_ref_head_pos_quat_vec_h(self, quat_mode: str = REF_QUAT_MODE):
        ref_wrists_hlr, ref_quat_hlr, _ = self.get_ref_pos_quat_vr_h_v2()
        ref_head_pos_h = ref_wrists_hlr[:1]
        ref_head_quat_h = ref_quat_hlr[:1]
        ref_head_quat_h = self.process_diff(ref_head_quat_h, quat_mode=quat_mode)
        out = torch.cat([ref_head_pos_h, ref_head_quat_h], dim=-1)
        self.debug_log(f"Head target: {out.tolist()}")
        return out

    def quat_diff(self, q1, q2):
        q1_inv = math_utils.quat_inv(q1)
        mul = math_utils.quat_mul(q1_inv, q2)
        aa = math_utils.axis_angle_from_quat(mul)
        return aa

    def vr_get_head_vel_h(self):
        _, _, vels_hlr = self.get_ref_pos_quat_vr_h_v2()
        return vels_hlr[:1]

    def vr_get_wrists_vel_h(self):
        _, _, vels_hlr = self.get_ref_pos_quat_vr_h_v2()
        return vels_hlr[None, 1:]

    def _validate_math_logic(self, opt_head_diff, opt_wrists_diff, opt_ref_head, opt_ref_wrists, opt_head_vel, opt_wrists_vel):
        """
        Runs legacy functions, checks all tensors against optimized versions,
        prints a consolidated report of all mismatches, and raises an error if any fail.
        """
        legacy_head_diff = self.vr_get_head_pos_quat_diff_h()
        legacy_wrists_diff = self.vr_get_wrists_pos_quat_diff_h()
        legacy_ref_head = self.vr_get_ref_head_pos_quat_vec_h()
        legacy_ref_wrists = self.vr_get_ref_wrists_pos_quat_vec_h()
        legacy_head_vel = self.vr_get_head_vel_h()
        legacy_wrists_vel = self.vr_get_wrists_vel_h()

        checks = [
            (opt_head_diff, legacy_head_diff, "Head Diff"),
            (opt_wrists_diff, legacy_wrists_diff, "Wrist Diff"),
            (opt_ref_head, legacy_ref_head, "Ref Head Target"),
            (opt_ref_wrists, legacy_ref_wrists, "Ref Wrist Target"),
            (opt_head_vel, legacy_head_vel, "Head Velocity"),
            (opt_wrists_vel, legacy_wrists_vel, "Wrist Velocity"),
        ]

        errors = []

        print("\n=== MATH VALIDATION REPORT ===")
        for opt, leg, name in checks:
            try:
                torch.testing.assert_close(opt, leg, rtol=1e-4, atol=1e-4, msg=name)
                print(f"[PASS] {name}")
            except AssertionError as e:
                diff = (opt - leg).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()

                err_msg = (
                    f"\n[FAIL] {name}\n"
                    f"       Shapes    -> Opt: {list(opt.shape)}, Legacy: {list(leg.shape)}\n"
                    f"       Max Diff  -> {max_diff:.6f}\n"
                    f"       Mean Diff -> {mean_diff:.6f}\n"
                    f"       PyTorch Details:\n{str(e)}"
                )
                print(err_msg)
                errors.append(err_msg)

        print("==============================\n")

        if errors:
            final_msg = f"Found {len(errors)} math mismatches during validation!\n" + "\n".join(errors)
            logging.error(final_msg)
            raise AssertionError(final_msg)

    def process_obs(self, obs_dict, timestep, action_carry, observation_carry, dry_run: bool = False):
        with self.profiler.time("Process observation"):
            with self.profiler.time("Get last action"):
                    if action_carry.last_raw_action is None:
                        last_raw_action = torch.zeros((len(SIM_DEFAULT_POSITIONS)), dtype=torch.float16, device=self.device)
                        last_motor_target = torch.tensor(SIM_DEFAULT_POSITIONS, dtype=torch.float16, device=self.device) * 0
                    else:
                        last_raw_action = action_carry.last_raw_action
                        last_motor_target = action_carry.last_motor_target
                        self._last_motor_target = last_motor_target

            with self.profiler.time("Compute observation"):
                obs_dict_t = self._obs_dict_to_tensor(obs_dict, non_blocking=True)

                if self.is_debug:
                    self.state_recorder.set_joint_pos(obs_dict_t["qj"].cpu().numpy().reshape(-1))
                    self.state_recorder.set_pos_w(self.robot_model.pelvis_pos_w.cpu().numpy().reshape(3))
                    self.state_recorder.set_quat_w(self.robot_model.pelvis_quat_w.cpu().numpy().reshape(4))
                    keypoint_pos_w, keypoint_quat_w = self.robot_model.get_keypoint_pos_quat_w()
                    self.state_recorder.set_keypoint_pos_w(keypoint_pos_w.cpu().numpy().reshape(-1, 3))
                    self.state_recorder.set_keypoint_quat_w(keypoint_quat_w.cpu().numpy().reshape(-1, 4))

            with self.profiler.time("Fetch vr observation"):
                obs = self.vr_observator.get_observation()

                self._cached_vr_obs = obs

                pose_pos = torch.as_tensor(obs.head_wrists_pose_h.pos, device=self.device, dtype=torch.float32).reshape(-1, 3)
                pose_quat = torch.as_tensor(obs.head_wrists_pose_h.quat, device=self.device, dtype=torch.float32).reshape(-1, 4)
                vel_lin = torch.as_tensor(obs.head_wrists_vel_h.lin_vel, device=self.device, dtype=torch.float32).reshape(-1, 3)
                self._cached_vr_tensors = (pose_pos, pose_quat, vel_lin)

                if not dry_run:
                    self._log_vr_observations()

            with self.profiler.time("Fused Calc Diff"):
                # get current obs from here
                self.kp_pos_h, self.kp_quat_h = self.robot_model.get_keypoint_pos_quat_h()
                vr_pos, vr_quat, vr_vel = self._cached_vr_tensors
                self._last_vr_obs = obs

                (head_diff, wrists_diff,
                 ref_head, ref_wrists,
                 head_vel, wrists_vel) = fused_calc_diffs(
                    vr_pos, vr_quat, vr_vel,
                    self.kp_pos_h, self.kp_quat_h,
                    DIF_QUAT_MODE, REF_QUAT_MODE
                )

                # self._validate_math_logic(
                #     head_diff, wrists_diff,
                #     ref_head, ref_wrists,
                #     head_vel, wrists_vel
                # )
                if self.is_debug:
                    vr_pos_w = math_utils.quat_apply(self.robot_model.heading_quat_w, vr_pos) + self.robot_model.heading_pos_w
                    vr_quat_w = math_utils.quat_mul(self.robot_model.heading_quat_w.expand_as(vr_quat), vr_quat)
                    self.state_recorder.set_target_keypoint_pos_w(vr_pos_w.cpu().numpy().reshape(-1, 3))
                    self.state_recorder.set_target_keypoint_quat_w(vr_quat_w.cpu().numpy().reshape(-1, 4))

            with self.profiler.time("Append to buffers"):
                with self.profiler.time("Non blocking append"):
                    self.buffers['ang_vel'].append(obs_dict_t["ang_vel"].mul_(0.05), non_blocking=True)
                    self.buffers['gravity_orientation'].append(obs_dict_t["gravity_orientation"], non_blocking=True)
                    self.buffers['qj'].append(self.map_qj(obs_dict_t["qj"]), non_blocking=True)
                    self.buffers['dqj'].append(self.map_dqj(obs_dict_t["dqj"]).mul_(0.05), non_blocking=True)
                    self.buffers['last_actions'].append(last_raw_action, non_blocking=True)
                    self.buffers['last_motor_targets'].append(last_motor_target, non_blocking=True)

                    self.buffers['head_pos_quat_diff_h'].append(head_diff, non_blocking=True)
                    self.buffers['wrists_pos_quat_diff_h'].append(wrists_diff, non_blocking=True)
                    self.buffers['ref_wrists_pos_quat_vec_h'].append(ref_wrists, non_blocking=True)
                    self.buffers['ref_head_pos_quat_vec_h'].append(ref_head, non_blocking=True)
                    self.buffers['head_vel_h'].append(head_vel.mul_(0.05), non_blocking=True)
                    self.buffers['wrists_vel_h'].append(wrists_vel.mul_(0.05), non_blocking=True)

                with self.profiler.time("Clear cache"):
                    self._clear_vr_observation_cache()

                with self.profiler.time("Synchronize"):
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

            if self.is_debug:
                self.state_recorder.write_state()
            return self.get_obs_tensor(), observation_carry
    def _compose_tensors_into_obs(
            self,
            *tensors,
            non_blocking: bool = False
        ):
        if not hasattr(self, '_composed_obs'):
            self._composed_obs = torch.cat(
                [x.reshape(HISTORY, -1) for x in tensors],
                dim=-1,
            )[None, ...].float()
            self._lengths = [t.numel()//HISTORY for t in tensors]
        else:
            st = 0
            for l, t in zip(self._lengths, tensors):
                self._composed_obs[:, :, st:st + l].copy_(t.reshape(HISTORY, -1), non_blocking=non_blocking)
                st += l
        return self._composed_obs

    def get_info(self):
        kp_pos_h, kp_quat_h = self.robot_model.get_keypoint_pos_quat_h()
        keypoint_pos_w, keypoint_quat_w = self.robot_model.get_keypoint_pos_quat_w()
        heading_pos_w, heading_quat_w = self.robot_model.get_heading_pos_quat_w()

        localization_obs = self.localization_observator.get_observation()
        info = {
            'pelvis_height': (
                float(localization_obs.pose.pos[2]) -
                (self.robot_model.head_pos_w[2] - self.robot_model.pelvis_pos_w[2]).item()),
            'last_motor_target': torch.index_select(self._last_motor_target, 0, self.joint_mapping_from_sim).cpu().tolist() if self._last_motor_target is not None else [],
            'kp_pos_h': kp_pos_h.tolist()[0],
            'kp_angles_h': [_quat_to_axis_angle(kp).tolist() for kp in kp_quat_h][0],
            'kp_pos_w': keypoint_pos_w.tolist()[0],
            'kp_angles_w': [_quat_to_axis_angle(kp).tolist() for kp in keypoint_quat_w][0],
            'pelvis_pos_w': self.robot_model.pelvis_pos_w.squeeze().tolist(),
            'pelvis_angles_w': _quat_to_axis_angle(self.robot_model.pelvis_quat_w).tolist(),
            'heading_pos_w': heading_pos_w.tolist(),
            'heading_angles_w': _quat_to_axis_angle(heading_quat_w).tolist(),
            'localization_pos_w': localization_obs.pose.pos,
            'localization_angles_w': _quat_to_axis_angle(torch.tensor(localization_obs.pose.quat)).tolist(),
        }

        # VR reference (target) positions in world frame from cached observation
        # head_wrists_pose_w order: [Head, L_Wrist, R_Wrist]
        # Reorder to kp order: [L_Wrist, R_Wrist, Head]
        if self._last_vr_obs is not None:
            pose_w = self._last_vr_obs.head_wrists_pose_w
            ref_pos_w = pose_w.pos[[1, 2, 0]].tolist()
            ref_quat_w = pose_w.quat[[1, 2, 0]]
            info['ref_kp_pos_w'] = ref_pos_w
            info['ref_kp_angles_w'] = [_quat_to_axis_angle(torch.tensor(q)).tolist() for q in ref_quat_w]

            if self._last_vr_obs.head_wrists_pose_vrw is not None:
                pose_vrw = self._last_vr_obs.head_wrists_pose_vrw
                info['ref_kp_pos_vrw'] = pose_vrw.pos[[1, 2, 0]].tolist()
                info['ref_kp_angles_vrw'] = [_quat_to_axis_angle(torch.tensor(q)).tolist() for q in pose_vrw.quat[[1, 2, 0]]]

            if self._last_vr_obs.head_wrists_vel_vrw is not None:
                vel_vrw = self._last_vr_obs.head_wrists_vel_vrw
                info['ref_kp_lin_vel_vrw'] = vel_vrw.lin_vel[[1, 2, 0]].tolist()

        return info
