from abc import abstractmethod
import datetime
import logging
import time
from lib.deploy.debug.profiler import Profiler
import torch
import numpy as np
import os
import typing as tp
from contextlib import contextmanager
from dataclasses import dataclass
import statistics
from lib.deploy.configs.joint_mappings import (
    REAL_ORDER,
    G1_DEFAULT_POSITIONS,
    MOTOR_ID_2_ACTION_ID,
    SIM_DEFAULT_POSITIONS,
    SIM_SCALE,
)
import collections
from lib.deploy.deploy_utils.circular_buffer import CircularBuffer
from lib.deploy.deploy_utils.state_recorder import StateRecorder

torch.set_printoptions(precision=4, sci_mode=False)

@dataclass
class CarryFromAction:
    last_raw_action: torch.Tensor
    last_motor_target: torch.Tensor

@dataclass
class CarryFromObservation:
    current_joint_target: torch.Tensor


class BaseEnvProcessor:
    def __init__(
            self,
            *,
            observation_keys: tp.List[str],
            device: str = 'cuda:0',
            history_length: int = 10,
            log_level: tp.Optional[int] = None,
            log_dir: str = "./logs",
            log_file_prefix: str = "env_processor",
            profiler: tp.Optional[Profiler] = None
            ):
        self.device = device
        self.history_length = history_length
        self.carry_state = None
        self.num_motors = len(REAL_ORDER)
        self.observation_keys = observation_keys
        self.profiler = profiler if profiler is not None else Profiler(enabled=False)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._init_action_buffer()
        self._init_joint_buffer()
        self._init_circular_buffers()
        self._init_joint_mapping_tensor()


        if log_level is not None:
            os.makedirs(log_dir, exist_ok=True)

            state_log_file = os.path.join(log_dir, f"state_{timestamp}.log")
            self.state_recorder = StateRecorder(state_log_file)
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_file = os.path.join(log_dir, f"{log_file_prefix}_{timestamp}.log")

            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.setLevel(log_level)

            if not self.logger.handlers or True:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(logging.Formatter(
                    "[%(asctime)s.%(msecs)03d] [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S"
                ))

                file_handler = logging.FileHandler(log_file, mode='a')
                file_handler.setFormatter(logging.Formatter(
                    "[%(asctime)s.%(msecs)03d] [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S"
                ))
                self.logger.addHandler(console_handler)
                self.logger.addHandler(file_handler)

                self.logger.info(f"Logging to file: {log_file}")
        else:
            self.logger = None

    def debug_log(self, *args, **kwargs):
        if self.logger is not None:
            self.logger.debug(*args, **kwargs)

    def warning_log(self, *args, **kwargs):
        if self.logger is not None:
            self.logger.warning(*args, **kwargs)

    def error_log(self, *args, **kwargs):
        if self.logger is not None:
            self.logger.error(*args, **kwargs)

    def init_carries(self) -> tp.Tuple[CarryFromAction, CarryFromObservation]:
        action_carry = CarryFromAction(
            last_raw_action=None,
            last_motor_target=None,
        )
        observation_carry = CarryFromObservation(
            current_joint_target=None,
        )
        return action_carry, observation_carry

    def _init_action_buffer(self):
        self.action_buffer = torch.zeros(len(REAL_ORDER), dtype=torch.float32, device=self.device)
        self.sim_def_joint_pos = torch.tensor(SIM_DEFAULT_POSITIONS, device=self.device)
        self.sim_scale = SIM_SCALE

    def _init_joint_buffer(self):
        self.joint_buffer = torch.zeros(len(REAL_ORDER), dtype=torch.float32, device=self.device)

        for motor_id, default_pos in G1_DEFAULT_POSITIONS.items():
            sim_id = MOTOR_ID_2_ACTION_ID[motor_id]
            self.joint_buffer[sim_id] = default_pos

    def _init_circular_buffers(self):
        mk_buf = lambda: CircularBuffer(max_len=self.history_length)
        self.buffers = {
            k: mk_buf()
            for k in self.observation_keys
        }

    def _init_joint_mapping_tensor(self):
        self.joint_mapping_to_sim = torch.zeros_like(self.joint_buffer).long()
        self.joint_mapping_from_sim = torch.zeros_like(self.joint_buffer).long()
        for motor_id, sim_id in MOTOR_ID_2_ACTION_ID.items():
            self.joint_mapping_to_sim[sim_id] = motor_id
            self.joint_mapping_from_sim[motor_id] = sim_id

    def _obs_dict_to_tensor(self, obs_dict, non_blocking: bool = False):
        if not hasattr(self, 'obs_dict'):
            self.obs_dict = {k: torch.tensor(v, device=self.device) for k, v in obs_dict.items()}
            self.cpu_staging = {k: torch.empty_like(v, device='cpu', pin_memory=torch.cuda.is_available())
                           for k, v in self.obs_dict.items()}

        else:
            for k in self.obs_dict.keys():
                self.cpu_staging[k].numpy()[:] = obs_dict[k]
                self.obs_dict[k].copy_(self.cpu_staging[k], non_blocking=non_blocking)
        return self.obs_dict

    def _compose_tensors_into_obs(
            self,
            *tensors,
            non_blocking: bool = False
        ):
        if not hasattr(self, '_composed_obs'):
            self._composed_obs = torch.cat(
                [x.reshape(-1) for x in tensors],
                dim=0,
            )[None, ...].float()
            self._lengths = [t.numel() for t in tensors]
        else:
            st = 0
            for l, t in zip(self._lengths, tensors):
                self._composed_obs[:, st:st + l].copy_(t.view(-1), non_blocking=non_blocking)
                st += l
        return self._composed_obs

    def get_obs_tensor(self):
        tensors = [
            self.buffers[k].buffer
            for k in self.observation_keys
        ]
        obs_tensor = self._compose_tensors_into_obs(*tensors, non_blocking=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        return obs_tensor

    @abstractmethod
    def process_obs(self, obs_dict, timestep, action_carry: CarryFromAction, observation_carry: CarryFromObservation) -> tp.Tuple[torch.Tensor, CarryFromObservation]:
        ...

    def process_action(self, action_from_policy, action_carry: CarryFromAction, observation_carry: CarryFromObservation) -> tp.Tuple[torch.Tensor, CarryFromAction]:
        with self.profiler.time("Process action"):
            if not hasattr(self, '_process_action_buffer'):
                self._process_action_buffer = torch.empty_like(action_from_policy[0])
                self._action_from_policy_scaled_mapped = torch.empty_like(action_from_policy[0])

            if action_carry.last_raw_action is None:
                action_carry.last_raw_action = torch.empty_like(action_from_policy[0])
                action_carry.last_motor_target = torch.empty_like(action_from_policy[0])
            action_carry.last_raw_action.copy_(action_from_policy[0])

            self._process_action_buffer.copy_(action_from_policy[0])

            if action_carry.last_raw_action is None:
                action_carry.last_raw_action = torch.empty_like(action_from_policy[0])
                action_carry.last_motor_target = torch.empty_like(action_from_policy[0])


            action_carry.last_motor_target.copy_(self._process_action_buffer)

            torch.index_select(self._process_action_buffer, 0, self.joint_mapping_from_sim, out=self._action_from_policy_scaled_mapped)
            return self._action_from_policy_scaled_mapped, action_carry

    def map_qj_fk(self, qj):
        if not hasattr(self, '_qj_fk_mapped'):
            self._qj_fk_mapped = torch.empty(self.num_motors, device=qj.device, dtype=qj.dtype)
        torch.index_select(qj, 0, self.joint_mapping_to_sim, out=self._qj_fk_mapped[:self.num_motors])
        return self._qj_fk_mapped

    def map_qj(self, qj):
        if not hasattr(self, '_qj_mapped'):
            # self._qj_mapped = torch.empty(self.num_motors * 2, device=qj.device, dtype=qj.dtype)
            self._qj_mapped = torch.empty(self.num_motors, device=qj.device, dtype=qj.dtype)
        torch.index_select(qj, 0, self.joint_mapping_to_sim, out=self._qj_mapped[:self.num_motors])
        # self._qj_mapped[self.num_motors:].copy_(tgt_qj - self._qj_mapped[:self.num_motors])
        self._qj_mapped[:self.num_motors].sub_(self.sim_def_joint_pos)
        return self._qj_mapped


    def map_dqj(self, dqj):
        if not hasattr(self, '_dqj_mapped'):
            # self._dqj_mapped = torch.empty(self.num_motors * 2, device=dqj.device, dtype=dqj.dtype)
            self._dqj_mapped = torch.empty(self.num_motors, device=dqj.device, dtype=dqj.dtype)
        torch.index_select(dqj, 0, self.joint_mapping_to_sim, out=self._dqj_mapped[:self.num_motors])
        # self._dqj_mapped[self.num_motors:].copy_(tgt_dqj - self._dqj_mapped[:self.num_motors])
        return self._dqj_mapped


    def reset(self):
        for v in self.buffers.values():
            v.reset()


def check_robot_fall(gravity_orientation):
    gravity_z = gravity_orientation[2]
    tilt_angle = np.arccos(np.clip(abs(gravity_z), 0, 1)) * 180 / np.pi
    if tilt_angle > 45.0:
        return True
    return False


_timer_stats: tp.Dict[str, tp.List[float]] = {}

@contextmanager
def Timer(label='Code block', maxlen: int = 150):
    """Context manager to measure execution time and print cumulative statistics."""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        elapsed_time = (end_time - start_time) * 1000

        # Store the time for this label
        k = (label, maxlen)
        if k not in _timer_stats:
            _timer_stats[k] = collections.deque(maxlen=maxlen)
        _timer_stats[k].append(elapsed_time)

        # Print current time and statistics
        times = _timer_stats[k]
        count = len(times)

        # if count == 1:
        #     print(f"{label}: {elapsed_time:.2f} ms")
        # else:
        #     mean = statistics.mean(times)
        #     median = statistics.median(times)
        #     min_time = min(times)
        #     max_time = max(times)

        #     print(f"{label}: {elapsed_time:.2f} ms", end="\n")
        #     print(f" | Stats (n={count}): mean={mean:.2f}, median={median:.2f}, "
        #           f"min={min_time:.2f}, max={max_time:.2f}", end="\n")
