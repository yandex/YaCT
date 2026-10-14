from abc import ABC, abstractmethod
from typing import Dict, Union, Optional, List, Any
import numpy as np
import torch


class BaseEnv(ABC):
    @abstractmethod
    def step(self, action: Union[np.ndarray, torch.Tensor]) -> Dict[str, np.ndarray]:
        """
        Execute an action and return the observation.

        Args:
            action: Action array/tensor of shape (num_actions,)
                   Can be numpy array or torch tensor

        Returns:
            observation: Current observation dictionary
        """
        pass

    @abstractmethod
    def act(self, action: Union[np.ndarray, torch.Tensor]):
        """
        Execute action.

        Args:
            action: Action array/tensor
        """
        pass

    @abstractmethod
    def reset(self) -> Dict[str, np.ndarray]:
        """
        Reset environment and return initial observation.

        Returns:
            observation: Current observation dictionary
        """
        pass

    @abstractmethod
    def get_observation(self) -> Dict[str, np.ndarray]:
        """
        Get the current observation without taking any action.

        Returns:
            observation: Current observation dictionary
        """
        pass

    @abstractmethod
    def get_qj(self) -> List[float]:
        """Get current joint positions as list."""
        pass

    @abstractmethod
    def get_motor_temperatures(self) -> np.ndarray:
        """Get current motor temperatures"""
        pass

    @abstractmethod
    def get_current_motor_state(self) -> np.ndarray:
        """Get current motor (joint) positions."""
        pass

    @abstractmethod
    def get_env_time(self) -> float:
        """
        Get the current environment time.

        For simulation environments: returns common_step_counter * control_dt
        For real robot environments: returns time.time()

        Returns:
            Current time in seconds
        """
        pass

    @abstractmethod
    def go_to_initial_state(self, ask: bool = True, *, cancel_event=None):
        """Go to initial state."""
        pass

    @abstractmethod
    def go_to_state(self, target_positions, time_to_move: float = 4, *, cancel_event=None):
        """Move to target joints, stopping between commands if cancel_event is set."""
        pass

    @abstractmethod
    def zero_torque_mode(self):
        """Set robot to zero torque mode."""
        pass

    @abstractmethod
    def damping_mode(self):
        """Set robot to damping mode."""
        pass

    @abstractmethod
    def get_remote_controller(self) -> Any:
        """Get the remote controller object."""
        pass

    @property
    @abstractmethod
    def disconnected(self) -> bool:
        """Get whether robot is disconnected"""
        pass

    @property
    @abstractmethod
    def state(self) -> Any:
        """Get the current robot state."""
        pass

    @property
    @abstractmethod
    def command(self) -> Any:
        """Get the current command."""
        pass

    def close(self):
        pass
