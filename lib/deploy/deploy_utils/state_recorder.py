import time
import numpy as np
import json
from typing import Optional
from pathlib import Path


class StateRecorder:
    """Records state information including position, orientation, targets, and joint positions."""

    def __init__(self, output_file: str, mode: str = 'w'):
        """
        Initialize the state recorder.

        Args:
            output_file: Path to the output file
            mode: File mode ('w' for write/overwrite, 'a' for append)
        """
        self.output_file = Path(output_file)
        self.mode = mode

        # Current state storage
        self.pos_w: Optional[np.ndarray] = None
        self.quat_w: Optional[np.ndarray] = None

        self.keypoint_pos_w: Optional[np.ndarray] = None
        self.keypoint_quat_w: Optional[np.ndarray] = None
        self.target_keypoint_pos_w: Optional[np.ndarray] = None
        self.target_keypoint_quat_w: Optional[np.ndarray] = None
        self.joint_pos: Optional[np.ndarray] = None

        # Head differential (single values)
        self.diff_pos_head: Optional[np.ndarray] = None
        self.diff_quat_head: Optional[np.ndarray] = None

        # Wrists differential (2 wrists)
        self.diff_pos_wrists: Optional[np.ndarray] = None
        self.diff_quat_wrists: Optional[np.ndarray] = None

        # Create parent directory if it doesn't exist
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        # Initialize file (clear if mode is 'w')
        if mode == 'w':
            with open(self.output_file, 'w') as f:
                pass

    def set_diff_pos_head(self, diff_pos: np.ndarray):
        """
        Set differential position for head in hand/local frame.

        Args:
            diff_pos: Array of shape (3,) representing differential position [x, y, z]
        """
        diff_pos = np.asarray(diff_pos)
        assert diff_pos.shape == (3,), f"Expected shape (3,), got {diff_pos.shape}"
        self.diff_pos_head = diff_pos.copy()

    def set_diff_quat_head(self, diff_quat: np.ndarray):
        """
        Set differential quaternion for head in hand/local frame.

        Args:
            diff_quat: Array of shape (3,) representing diff axis angle
        """
        diff_quat = np.asarray(diff_quat)
        assert diff_quat.shape == (3,), f"Expected shape (3,), got {diff_quat.shape}"
        self.diff_quat_head = diff_quat.copy()

    def set_diff_pos_wrists(self, diff_pos: np.ndarray):
        """
        Set differential positions for wrists in hand/local frame.

        Args:
            diff_pos: Array of shape (2, 3) representing differential positions for 2 wrists
        """
        diff_pos = np.asarray(diff_pos)
        assert diff_pos.shape == (2, 3), f"Expected shape (2, 3), got {diff_pos.shape}"
        self.diff_pos_wrists = diff_pos.copy()

    def set_diff_quat_wrists(self, diff_quat: np.ndarray):
        """
        Set differential quaternions for wrists in hand/local frame.

        Args:
            diff_quat: Array of shape (2, 3) representing diff axis angle for 2 wrists
        """
        diff_quat = np.asarray(diff_quat)
        assert diff_quat.shape == (2, 3), f"Expected shape (2, 3), got {diff_quat.shape}"
        self.diff_quat_wrists = diff_quat.copy()

    def set_pos_w(self, pos: np.ndarray):
        """
        Set world position.

        Args:
            pos: Array of shape (3,) representing [x, y, z]
        """
        pos = np.asarray(pos)
        assert pos.shape == (3,), f"Expected shape (3,), got {pos.shape}"
        self.pos_w = pos.copy()

    def set_quat_w(self, quat: np.ndarray):
        """
        Set world orientation quaternion.

        Args:
            quat: Array of shape (4,) representing [w, x, y, z] or [x, y, z, w]
        """
        quat = np.asarray(quat)
        assert quat.shape == (4,), f"Expected shape (4,), got {quat.shape}"
        self.quat_w = quat.copy()

    def set_keypoint_pos_w(self, keypoint_pos: np.ndarray):
        """
        Set keypoint positions.

        Args:
            target_pos: Array of shape (num_keypoints, 3)
        """
        keypoint_pos = np.asarray(keypoint_pos)
        assert keypoint_pos.ndim == 2 and keypoint_pos.shape[1] == 3, \
            f"Expected shape (num_keypoints, 3), got {keypoint_pos.shape}"
        self.keypoint_pos_w = keypoint_pos.copy()

    def set_keypoint_quat_w(self, keypoint_quat: np.ndarray):
        """
        Set keypoint orientations.

        Args:
            keypoint_quat: Array of shape (num_keypoints, 4)
        """
        keypoint_quat = np.asarray(keypoint_quat)
        assert keypoint_quat.ndim == 2 and keypoint_quat.shape[1] == 4, \
            f"Expected shape (num_keypoints, 4), got {keypoint_quat.shape}"
        self.keypoint_quat_w = keypoint_quat.copy()

    def set_target_keypoint_pos_w(self, target_pos: np.ndarray):
        """
        Set target positions.

        Args:
            target_pos: Array of shape (num_targets, 3)
        """
        target_pos = np.asarray(target_pos)
        assert target_pos.ndim == 2 and target_pos.shape[1] == 3, \
            f"Expected shape (num_targets, 3), got {target_pos.shape}"
        self.target_keypoint_pos_w = target_pos.copy()

    def set_target_keypoint_quat_w(self, target_quat: np.ndarray):
        """
        Set target orientations.

        Args:
            target_quat: Array of shape (num_targets, 4)
        """
        target_quat = np.asarray(target_quat)
        assert target_quat.ndim == 2 and target_quat.shape[1] == 4, \
            f"Expected shape (num_targets, 4), got {target_quat.shape}"
        self.target_keypoint_quat_w = target_quat.copy()

    def set_joint_pos(self, joint_pos: np.ndarray):
        """
        Set joint positions.

        Args:
            joint_pos: Array of shape (num_joints,)
        """
        joint_pos = np.asarray(joint_pos)
        assert joint_pos.ndim == 1, f"Expected 1D array, got shape {joint_pos.shape}"
        self.joint_pos = joint_pos.copy()

    def write_state(self):
        """
        Write current state to file as a JSON line.

        Raises:
            ValueError: If any required state is not set
        """
        # Build state dictionary
        state = {}

        state['time'] = time.time()

        if self.pos_w is not None:
            state['pos_w'] = self.pos_w.tolist()

        if self.quat_w is not None:
            state['quat_w'] = self.quat_w.tolist()

        if self.keypoint_pos_w is not None:
            state['keypoint_pos_w'] = self.keypoint_pos_w.tolist()

        if self.keypoint_quat_w is not None:
            state['keypoint_quat_w'] = self.keypoint_quat_w.tolist()

        if self.target_keypoint_pos_w is not None:
            state['target_keypoint_pos_w'] = self.target_keypoint_pos_w.tolist()

        if self.target_keypoint_quat_w is not None:
            state['target_keypoint_quat_w'] = self.target_keypoint_quat_w.tolist()

        if self.joint_pos is not None:
            state['joint_pos'] = self.joint_pos.tolist()

        if self.diff_pos_head is not None:
            state['diff_pos_head'] = self.diff_pos_head.tolist()

        if self.diff_quat_head is not None:
            state['diff_quat_head'] = self.diff_quat_head.tolist()

        if self.diff_pos_wrists is not None:
            state['diff_pos_wrists'] = self.diff_pos_wrists.tolist()

        if self.diff_quat_wrists is not None:
            state['diff_quat_wrists'] = self.diff_quat_wrists.tolist()

        # Write to file
        with open(self.output_file, 'a') as f:
            json.dump(state, f)
            f.write('\n')

    def reset_state(self):
        """Clear all stored state values."""
        self.pos_w = None
        self.quat_w = None
        self.target_pos_w = None
        self.target_quat_w = None
        self.joint_pos = None
        self.diff_pos_head = None
        self.diff_quat_head = None
        self.diff_pos_wrists = None
        self.diff_quat_wrists = None


# Example usage
if __name__ == "__main__":
    # Create recorder
    recorder = StateRecorder("states/robot_states.jsonl")

    # Record first state
    recorder.set_pos_w(np.array([1.0, 2.0, 3.0]))
    recorder.set_quat_w(np.array([1.0, 0.0, 0.0, 0.0]))
    recorder.set_keypoint_pos_w(np.array([[0.5, 0.5, 1.0], [1.0, 1.0, 2.0]]))
    recorder.set_keypoint_quat_w(np.array([[1.0, 0.0, 0.0, 0.0], [0.707, 0.707, 0.0, 0.0]]))
    recorder.set_joint_pos(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))

    # Set head differential
    recorder.set_diff_pos_head(np.array([0.01, 0.02, 0.03]))
    recorder.set_diff_quat_head(np.array([1.0, 0.0, 0.0, 0.0]))

    # Set wrists differential (left and right wrist)
    recorder.set_diff_pos_wrists(np.array([[0.04, 0.05, 0.06],
                                            [0.07, 0.08, 0.09]]))
    recorder.set_diff_quat_wrists(np.array([[0.99, 0.01, 0.0, 0.0],
                                             [0.98, 0.02, 0.0, 0.0]]))
    recorder.write_state()

    # Record second state
    recorder.set_pos_w(np.array([1.1, 2.1, 3.1]))
    recorder.set_quat_w(np.array([0.99, 0.01, 0.0, 0.0]))
    recorder.set_target_keypoint_pos_w(np.array([[0.6, 0.6, 1.1], [1.1, 1.1, 2.1]]))
    recorder.set_target_keypoint_quat_w(np.array([[1.0, 0.0, 0.0, 0.0], [0.707, 0.707, 0.0, 0.0]]))
    recorder.set_joint_pos(np.array([0.11, 0.21, 0.31, 0.41, 0.51]))

    # Set head differential
    recorder.set_diff_pos_head(np.array([0.02, 0.03, 0.04]))
    recorder.set_diff_quat_head(np.array([0.999, 0.001, 0.0, 0.0]))

    # Set wrists differential
    recorder.set_diff_pos_wrists(np.array([[0.05, 0.06, 0.07],
                                            [0.08, 0.09, 0.10]]))
    recorder.set_diff_quat_wrists(np.array([[0.98, 0.02, 0.0, 0.0],
                                             [0.97, 0.03, 0.0, 0.0]]))
    recorder.write_state()

    print(f"States written to {recorder.output_file}")
