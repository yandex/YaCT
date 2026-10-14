from lib.deploy.vr.vr_observations import VrHeadingObservator
from lib.deploy.control.base_env import BaseEnv
from abc import abstractmethod
from typing import Dict, Union, Optional, Tuple, List, Any
import numpy as np
import pydantic
import torch
from pydantic import Field

from lib.deploy.deploy_utils.forward_kinematics import RobotModel


class SimScenarioEvaluation(pydantic.BaseModel):
    is_success: bool = Field(description="Whether the scenario success predicate is satisfied.")
    should_end_episode: bool = Field(
        default=False,
        description="Whether the scenario requests episode termination.",
    )
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Scenario-specific scalar metrics. "
            "Opaque to the base layer; an absent key means the scenario did not report it."
        ),
    )


class BaseSimEnv(BaseEnv):
    """
    Base interface for simulation environments.
    Extends BaseEnv with simulation-specific methods.
    """

    @abstractmethod
    def reset_sim(self) -> Dict[str, np.ndarray]:
        """
        Hard reset - reset simulation to initial state.
        Differs from reset() which might only reset the episode/task state but keep physics state.
        Does not reset the scenario state.

        Returns:
            observation: Current observation dictionary
        """
        pass

    def reset_scenario(self) -> Dict[str, np.ndarray]:
        """Resets the sim scenario state, returns the observation after reset."""
        return self.get_observation()

    @abstractmethod
    def apply_external_force_torque_to_link(
        self,
        link_name: str,
        force: Union[np.ndarray, torch.Tensor],
        torque: Union[np.ndarray, torch.Tensor],
        duration_steps: int = 1
    ):
        """
        Apply external force/torque to a link.

        Args:
            link_name: Name of the link to apply force to
            force: Force vector (3,)
            torque: Torque vector (3,)
            duration_steps: Number of simulation steps to apply the force
        """
        pass

    @abstractmethod
    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment.

        Returns:
            image: Rendered image frame (H, W, C) or None
        """
        pass

    @abstractmethod
    def save_frame_maybe(self):
        """Save frame to buffer for video recording if conditions met."""
        pass

    @abstractmethod
    def save_video(self, path: Optional[str] = None):
        """
        Save buffered frames as video.

        Args:
            path: Output path for video file. If None, generates a default name.
        """
        pass

    @abstractmethod
    def update_fk_vis(self, robot_model: RobotModel):
        """
        Update forward kinematics visualization.

        Args:
            robot_model: RobotModel instance providing FK data
        """
        pass

    @abstractmethod
    def update_tgt_vis(self, vr_heading_observator: VrHeadingObservator):
        """
        Update target visualization.

        Args:
            vr_heading_observator: Observator providing target pose data
        """
        pass

    @abstractmethod
    def get_head_pos_quat_w(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get head position and quaternion in world frame (Ground Truth).

        Returns:
            pos: Position (3,)
            quat: Quaternion (4,) [w, x, y, z]
        """
        pass

    def update_cameras(self) -> None:
        """Update any camera-backed caches on the simulation thread.
        """
        return None

    def get_camera_data(self) -> Optional[Dict[str, Any]]:
        """Fetch the latest camera packet, if the env exposes one.
        """
        return None

    def evaluate_scenario(self) -> SimScenarioEvaluation:
        """Evaluate the attached sim-side scenario and return the result.
        """
        return SimScenarioEvaluation(is_success=False, should_end_episode=False)

    @property
    @abstractmethod
    def sim_env(self) -> Any:
        """Get the underlying simulation environment object."""
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        """Get the device (cpu/cuda) used for tensors."""
        pass
