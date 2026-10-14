"""
Latency and discretization implemented without isaaclab dependencies.
"""

import numpy as np
from typing import Any, Callable, List, Optional, Tuple, Union
from enum import Enum
import copy


class InterpolationType(Enum):
    LINEAR = 0
    NONE = 1


class UpdateGuard:
    def __init__(self):
        self.last = None

    def __call__(self, env):
        step = env.common_step_counter

        if self.last is None:
            self.last = step
            return True

        if step == self.last:
            return False

        if step < self.last:
            # Reset detected
            self.last = step
            return True

        self.last = step
        return True

    def reset(self):
        self.last = None


class WithDiscretization:
    def __init__(
            self,
            term: Callable[[Any], Any],
            decimation: Union[int, Tuple[int, int]] = 1,
            augment_output_function: Optional[Callable] = None):
        self.term = term
        self.decimation = decimation
        self.update_guard = UpdateGuard()
        self.cached_val = None
        self.steps_until_update = 0
        self.augment_output_function = augment_output_function

    def __call__(self, env):
        if self.update_guard(env):
            self.steps_until_update -= 1

        update_needed = self.steps_until_update <= 0
        if self.cached_val is None:
            update_needed = True

        if update_needed:
            current = self.term(env)
            if self.augment_output_function is not None:
                current = self.augment_output_function(current)

            # Deep copy to ensure cache stability if term returns references
            self.cached_val = copy.deepcopy(current)

            dec_range = self.decimation if isinstance(
                self.decimation, (tuple, list)) else (self.decimation, self.decimation)
            self.steps_until_update = np.random.randint(
                dec_range[0], dec_range[1] + 1)

        return copy.deepcopy(self.cached_val)

    def reset(self):
        self.cached_val = None
        self.steps_until_update = 0
        if hasattr(self.term, "reset"):
            self.term.reset()  # type: ignore


class WithLatency:
    def __init__(self, term: Callable[[Any], Any], latency_range_ms: Tuple[int, int] = (0, 0), interpolation: InterpolationType = InterpolationType.NONE):
        self.term = term
        self.latency_range_ms = latency_range_ms
        self.interpolation = interpolation
        self.update_guard = UpdateGuard()

        self.buffer: List[Any] = []

        self.latency_ms = None
        self.step_dt_ms = None
        self.control_dt = None

    def __call__(self, env):
        # Init dt
        if self.step_dt_ms is None:
            if hasattr(env, 'config') and hasattr(env.config, 'control_dt'):
                self.control_dt = env.config.control_dt
                self.step_dt_ms = int(self.control_dt * 1000)
            else:
                # Default to 20ms if not found
                self.step_dt_ms = 20
                self.control_dt = 0.02

        if self.latency_ms is None:
            self.latency_ms = np.random.randint(
                self.latency_range_ms[0], self.latency_range_ms[1] + 1)

        if self.update_guard(env):
            val = self.term(env)
            self.buffer.append(copy.deepcopy(val))

            # Prune buffer - keep enough for max latency
            max_latency_steps = int(
                self.latency_range_ms[1] / self.step_dt_ms) + 5
            if len(self.buffer) > max_latency_steps + 10:
                self.buffer = self.buffer[-(max_latency_steps+5):]

        if len(self.buffer) == 0:
            return self.term(env)

        # Calculate delay in steps
        delay_steps = self.latency_ms / self.step_dt_ms

        # Indexing: buffer[-1] is delay 0
        idx_float = len(self.buffer) - 1 - delay_steps

        if self.interpolation == InterpolationType.NONE:
            idx = int(np.round(idx_float))
            idx = max(0, min(idx, len(self.buffer) - 1))
            return self.buffer[idx]

        elif self.interpolation == InterpolationType.LINEAR:
            idx_floor = int(np.floor(idx_float))
            idx_ceil = int(np.ceil(idx_float))

            idx_floor = max(0, min(idx_floor, len(self.buffer) - 1))
            idx_ceil = max(0, min(idx_ceil, len(self.buffer) - 1))

            if idx_floor == idx_ceil:
                return self.buffer[idx_floor]

            val_low = self.buffer[idx_floor]
            val_high = self.buffer[idx_ceil]

            # alpha is distance from floor
            alpha = idx_float - idx_floor

            return self._interpolate(val_low, val_high, alpha)

    def _interpolate(self, val_a, val_b, alpha):
        if isinstance(val_a, tuple):
            return tuple(self._interpolate(a, b, alpha) for a, b in zip(val_a, val_b))

        if isinstance(val_a, np.ndarray):
            return (1.0 - alpha) * val_a + alpha * val_b

        if isinstance(val_a, (int, float, np.number)):
            return (1.0 - alpha) * val_a + alpha * val_b

        return val_a

    def reset(self):
        self.buffer = []
        self.latency_ms = None
        if hasattr(self.term, "reset"):
            self.term.reset()  # type: ignore
