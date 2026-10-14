import json
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_
from functools import lru_cache

@lru_cache()
def init_reset_topic(init_time_printer=False):
    reset_pose_publisher = ChannelPublisher("rt/reset_pose/cmd", String_)
    reset_pose_publisher.Init()

    def publish_reset():
        category = 2 # full scene reset
        msg = String_(data=str(category))
        reset_pose_publisher.Write(msg)

    def sim_time_callback(msg):
        data = json.loads(msg.data)
        print(f"Sim Time: {data['simulation_time']:.3f}s | Step: {data['step_count']} | Real: {data['real_time']:.3f}s")

    if init_time_printer:
        time_subscriber = ChannelSubscriber("rt/sim_state", String_)
        time_subscriber.Init(sim_time_callback, 1)

    return publish_reset


class SimTimeSource:
    def __init__(self):
        self.sim_time = 0.0
        self.subscriber = None
        self._init_subscriber()

    def _init_subscriber(self):
        """Initialize the subscriber for simulation state."""
        try:
            self.subscriber = ChannelSubscriber("rt/sim_state", String_)
            self.subscriber.Init(self._sim_state_callback, 1)
        except Exception as e:
            print(f"Warning: Could not initialize sim state subscriber: {e}")
            print("Falling back to real time")

    def _sim_state_callback(self, msg):
        """Callback for simulation state messages."""
        try:
            data = json.loads(msg.data)
            self.sim_time = data.get('simulation_time', 0.0)
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    def __call__(self):
        """Return current simulation time."""
        return self.sim_time

    def close(self):
        """Clean up the subscriber."""
        if self.subscriber:
            self.subscriber.Close()
            self.subscriber = None


def create_sim_time_source():
    """
    Create a time source function that reads simulation time from rt/sim_state topic.
    Returns a callable that can be used with the Sleeper class.
    """
    return SimTimeSource()
