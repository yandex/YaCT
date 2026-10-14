"""
VR Replay Wrapper for teleoperation experiments.

This module provides a wrapper that can replay recorded VR pose states
from a saved file, allowing for playback of previously recorded VR sessions.
"""

import json
import logging
from typing import List, Optional
from lib.deploy.vr.vr_pose_tracker import VRPoseState
import copy
import warnings

logger = logging.getLogger(__name__)


class VRReplayWrapper:
    """
    Wrapper that replays recorded VR pose states.

    This class loads recorded VR states from a file and provides them
    in the same interface as VRPoseTracker, allowing seamless playback
    of recorded VR sessions.
    """

    def __init__(self, recording_path: Optional[str] = None, loop: bool = False):
        """
        Initialize the VR replay wrapper.

        Args:
            recording_path: Optional path to load recording from immediately
        """
        self.recorded_states: List[VRPoseState] = []
        self.current_index = 0
        self.loop = loop

        if recording_path:
            self.load(recording_path)
        if loop:
            warnings.warn('loop=True does not supported in VRReplayWrapper now, since it is dangerous')

    def load(self, path: str) -> None:
        """
        Load recorded states from a file.

        Args:
            path: File path to load the recorded states from
        """
        with open(path, 'r', encoding="utf-8") as f:
            save_data = json.load(f)

        self.recorded_states = [VRPoseState(**state) for state in save_data['recorded_states']]
        self.current_index = 0
        logger.debug("Loaded %s recorded states from %s", len(self.recorded_states), path)

    def get_state(self) -> VRPoseState:
        """
        Get current VR pose state from replay.

        Returns:
            VRPoseState containing head and wrist poses from replay
        """
        if self.current_index >= len(self.recorded_states):
            self.recorded_states[-1] = self.add_time_offset_to_state(self.recorded_states[-1], 0.02)
            return copy.deepcopy(self.recorded_states[-1])


        out = copy.deepcopy(self.recorded_states[self.current_index])
        return out

    def add_time_offset_to_state(self, state: VRPoseState, offset: float):
        state.head_timestamp += offset
        state.right_wrist_timestamp += offset
        state.left_wrist_timestamp += offset
        return state


    def step(self):
        self.current_index += 1

    @property
    def finished(self) -> bool:
        return len(self.recorded_states) > 0 and self.current_index >= len(self.recorded_states)

    def is_connected(self) -> bool:
        """
        Check if replay system has data available.

        Returns:
            True if recording is loaded and has states
        """
        return len(self.recorded_states) > 0

    def get_controller_states(self):
        """
        Get raw controller states for advanced usage.
        Note: This returns None for replay as we don't have raw controller data.
        """
        return None, None

    def len(self):
        return len(self.recorded_states)
