import numpy as np
from scipy.spatial.transform import Rotation as R
from collections import deque

class RealTimeVelocityEstimator:
    def __init__(self, window_seconds=0.5, max_linear_vel=10.0, max_angular_vel=20.0,
                 aggregation_method='mean'):
        """
        Real-time velocity estimator using a finite time window buffer

        Args:
            window_seconds: time window in seconds to keep velocity measurements
            max_linear_vel: velocity clipping threshold (m/s)
            max_angular_vel: angular velocity clipping threshold (rad/s)
            aggregation_method: 'mean', 'median', or 'weighted_mean'
        """
        self.window_seconds = window_seconds
        self.max_lin_vel = max_linear_vel
        self.max_ang_vel = max_angular_vel
        self.aggregation_method = aggregation_method

        self.prev_pose = None
        self.prev_time = None

        # Buffers to store (timestamp, linear_vel, angular_vel)
        self.velocity_buffer = deque()

        # Current velocity estimates
        self.linear_vel = np.zeros(3)
        self.angular_vel = np.zeros(3)

    def _clean_buffer(self, current_time):
        """Remove measurements older than window_seconds"""
        cutoff_time = current_time - self.window_seconds

        while self.velocity_buffer and self.velocity_buffer[0][0] < cutoff_time:
            self.velocity_buffer.popleft()

    def _aggregate_velocities(self, current_time):
        """Compute velocity estimate from buffer"""
        if not self.velocity_buffer:
            return np.zeros(3), np.zeros(3)

        # Extract velocities and timestamps
        timestamps = np.array([item[0] for item in self.velocity_buffer])
        linear_vels = np.array([item[1] for item in self.velocity_buffer])
        angular_vels = np.array([item[2] for item in self.velocity_buffer])

        if self.aggregation_method == 'mean':
            lin_vel = np.mean(linear_vels, axis=0)
            ang_vel = np.mean(angular_vels, axis=0)

        elif self.aggregation_method == 'median':
            lin_vel = np.median(linear_vels, axis=0)
            ang_vel = np.median(angular_vels, axis=0)

        elif self.aggregation_method == 'weighted_mean':
            # Weight more recent measurements higher
            ages = current_time - timestamps
            weights = np.exp(-ages / (self.window_seconds / 3))  # Decay over window/3
            weights = weights / np.sum(weights)  # Normalize

            lin_vel = np.average(linear_vels, axis=0, weights=weights)
            ang_vel = np.average(angular_vels, axis=0, weights=weights)

        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

        return lin_vel, ang_vel

    def update(self, pose, timestamp):
        """
        Update with new pose and return current velocity estimate

        Args:
            pose: (4, 4) transformation matrix
            timestamp: current timestamp in seconds

        Returns:
            (linear_velocity, angular_velocity): both (3,) arrays
        """
        # Clean old measurements from buffer
        self._clean_buffer(timestamp)

        if self.prev_pose is None:
            # First frame - no velocity estimate yet
            self.prev_pose = pose.copy()
            self.prev_time = timestamp
            return self.linear_vel.copy(), self.angular_vel.copy()

        dt = timestamp - self.prev_time
        if dt <= 0:
            return self.linear_vel.copy(), self.angular_vel.copy()

        # Compute raw velocities for this frame
        pos_curr = pose[:3, 3]
        pos_prev = self.prev_pose[:3, 3]
        raw_lin_vel = (pos_curr - pos_prev) / dt

        # Compute raw angular velocity
        rot_curr = R.from_matrix(pose[:3, :3])
        rot_prev = R.from_matrix(self.prev_pose[:3, :3])
        relative_rot = rot_curr * rot_prev.inv()
        raw_ang_vel = relative_rot.as_rotvec() / dt

        # Clip outliers
        raw_lin_speed = np.linalg.norm(raw_lin_vel)
        if raw_lin_speed > self.max_lin_vel:
            raw_lin_vel = raw_lin_vel / raw_lin_speed * self.max_lin_vel

        raw_ang_speed = np.linalg.norm(raw_ang_vel)
        if raw_ang_speed > self.max_ang_vel:
            raw_ang_vel = raw_ang_vel / raw_ang_speed * self.max_ang_vel

        # Add to buffer
        self.velocity_buffer.append((timestamp, raw_lin_vel.copy(), raw_ang_vel.copy()))

        # Compute aggregated velocities from buffer
        self.linear_vel, self.angular_vel = self._aggregate_velocities(timestamp)

        # Update state
        self.prev_pose = pose.copy()
        self.prev_time = timestamp

        return self.linear_vel.copy(), self.angular_vel.copy()

    def get_buffer_info(self):
        """Get information about current buffer state"""
        if not self.velocity_buffer:
            return {
                'buffer_size': 0,
                'time_span': 0.0,
                'oldest_measurement': None,
                'newest_measurement': None
            }

        oldest_time = self.velocity_buffer[0][0]
        newest_time = self.velocity_buffer[-1][0]

        return {
            'buffer_size': len(self.velocity_buffer),
            'time_span': newest_time - oldest_time,
            'oldest_measurement': oldest_time,
            'newest_measurement': newest_time
        }

    def reset(self):
        """Reset the estimator state"""
        self.prev_pose = None
        self.prev_time = None
        self.velocity_buffer.clear()
        self.linear_vel = np.zeros(3)
        self.angular_vel = np.zeros(3)

"""
Recommended Settings for VR:
Head tracking: window_seconds=0.5, method='mean'
Hand tracking: window_seconds=0.3, method='weighted_mean'
Very noisy data: window_seconds=0.4, method='median'
Low latency: window_seconds=0.2, method='weighted_mean'


# Usage examples:
# For smooth but responsive (good for hand tracking)
estimator = RealTimeVelocityEstimator(
    window_seconds=0.3,
    aggregation_method='weighted_mean'
)

# For maximum smoothing (good for head tracking)
estimator_smooth = RealTimeVelocityEstimator(
    window_seconds=0.5,
    aggregation_method='mean'
)

# For noise rejection (good for very noisy data)
estimator_robust = RealTimeVelocityEstimator(
    window_seconds=0.4,
    aggregation_method='median'
)

# In your VR loop:
linear_vel, angular_vel = estimator.update(current_pose, current_time)

# Optional: check buffer status
buffer_info = estimator.get_buffer_info()
print(f"Using {buffer_info['buffer_size']} measurements over {buffer_info['time_span']:.3f}s")

"""
