import time
import logging
from typing import Callable, Optional


class Sleeper:
    """
    A frequency-controlled sleeper that maintains loop timing using a configurable time source.
    """

    def __init__(
        self,
        frequency_hz: float,
        time_source: Optional[Callable[[], float]] = None,
        enable_logging: bool = True,
    ):
        """
        Initialize the sleeper.

        Args:
            frequency_hz: Target frequency in Hz for the loop
            time_source: Time function that returns current time in seconds. Defaults to time.time()
            enable_logging: Whether to emit deadline-miss warnings at all
        """
        self.frequency_hz = frequency_hz
        self.period = 1.0 / frequency_hz
        self.time_source = time_source or time.time
        self.last_time = self.time_source()

        self.cnt = 0
        self.miss_cnt = 0
        self.last_oversleep_time = None
        self.logger = None
        if enable_logging:
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.addHandler(logging.FileHandler("sleeper.log"))


    def sleep(self):
        """
        Sleep to maintain the target frequency.
        Returns the actual slack after waking up.
        """
        self.cnt += 1
        self.last_oversleep_time = None

        current_time = self.time_source()
        elapsed_before_sleep = current_time - self.last_time
        requested_sleep_time = self.period - elapsed_before_sleep

        target_time = current_time
        if requested_sleep_time > 0.001:
            # Sleep until the planned deadline, then measure how much slack remains.
            target_time = current_time + requested_sleep_time
            self._sleep_until_time(target_time)

        wake_time = self.time_source()
        elapsed_after_sleep = wake_time - self.last_time
        sleep_time = self.period - elapsed_after_sleep

        if sleep_time < 0:
            self.miss_cnt += 1
            if self.logger is not None and -sleep_time > 0.005:
                self.logger.warning(f"DEADLINE MISSED BY {-sleep_time * 1000:.1f}ms {self.miss_cnt/self.cnt}, out of {self.cnt}")

            if requested_sleep_time > 0.001:
                oversleep_time = wake_time - target_time
                self.last_oversleep_time = oversleep_time
                if self.logger is not None and oversleep_time > 0.005:
                    self.logger.warning(
                        f"SLEEP OVERSHOT DEADLINE BY {oversleep_time * 1000:.1f}ms "
                        f"(requested {requested_sleep_time * 1000:.1f}ms sleep)"
                    )

        self.last_time = wake_time

        return sleep_time

    def _sleep_until_time(self, target_time: float):
        while self.time_source() < target_time:
            time.sleep(0.0002)

    def set_frequency(self, frequency_hz: float):
        """Change the target frequency."""
        self.frequency_hz = frequency_hz
        self.period = 1.0 / frequency_hz
        self.last_time = self.time_source()

    def reset_timing(self):
        """Reset the timing reference point."""
        self.last_time = self.time_source()

def create_sleeper_from_g1_env(
    env: "G1Env",
    enable_logging: bool = True,
):
    time_source = time.time
    if env.network == 'sim':
        from lib.deploy.deploy_utils.g1_sim import create_sim_time_source
        time_source = create_sim_time_source()
    freq = int(1 / env.config.control_dt)
    return Sleeper(
        frequency_hz=freq,
        time_source=time_source,
        enable_logging=enable_logging,
    )
