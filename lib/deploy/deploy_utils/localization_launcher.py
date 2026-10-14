import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

from lib.deploy.deploy_utils.ssh_utils import RemoteServiceLauncher, SSHConfig, DockerConfig

try:
    from lib.deploy.vr.localization import OdometryData
    from unitree_sdk2py.core.channel import ChannelSubscriber
except ImportError:
    print("No unitree libs found!")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _resolve_jetson_credentials(
    host: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
) -> tuple[str, str, str]:
    resolved_host = host or os.environ.get("YACT_JETSON_HOST", "192.168.123.164")
    resolved_user = user or os.environ.get("YACT_JETSON_USER", "unitree")
    resolved_password = password or os.environ.get("YACT_JETSON_PASSWORD", "123")
    if not resolved_password:
        raise ValueError(
            "Jetson SSH password required: set YACT_JETSON_PASSWORD or pass password="
        )
    return resolved_host, resolved_user, resolved_password


@dataclass
class FastLioConfig:
    """Configuration for Fast-LIO on the Jetson lio container."""

    fast_lio_dir: str = "/opt/lio/scripts"
    fast_lio_script: str = "loco.sh"
    fast_lio_log_path: str = "/tmp/fast_lio.log"
    fast_lio_process_pattern: str = "fastlio_mapping"
    fast_lio_wait_time: float = 5.0

    ros_noetic_setup: str = "/opt/ros/noetic/setup.bash"
    ws_livox_setup: str = "/build/ws_livox/devel/setup.bash"
    ws_lio_setup: str = "/build/ws_lio/devel/setup.bash"

    def get_fast_lio_start_command(self) -> str:
        return f"cd {self.fast_lio_dir} && ./{self.fast_lio_script}"


class FastLioLauncher(RemoteServiceLauncher):
    """Starts Fast-LIO in the lio Docker container on the robot Jetson."""

    def __init__(
        self,
        ssh_config: SSHConfig,
        container_name: str,
        fast_lio_config: FastLioConfig,
        docker_config: Optional[DockerConfig] = None,
        restart_container: bool = True,
    ):
        super().__init__(
            ssh_config,
            container_name,
            docker_config,
            auto_start_container=True,
        )
        self.config = fast_lio_config
        self.restart_container = restart_container

    def get_process_pattern(self) -> str:
        return self.config.fast_lio_process_pattern

    def get_start_command(self) -> str:
        return self.config.get_fast_lio_start_command()

    def get_log_file(self) -> str:
        return self.config.fast_lio_log_path

    def pre_launch_checks(self) -> bool:
        logger.info("Checking ROS setup files...")
        setup_files = [
            self.config.ros_noetic_setup,
            self.config.ws_livox_setup,
            self.config.ws_lio_setup,
        ]
        for setup_file in setup_files:
            cmd = f"test -f {setup_file} && echo 'exists'"
            _, out, _ = self.container.exec_in_container(cmd)
            if "exists" not in out:
                logger.error(f"Setup file not found: {setup_file}")
                return False

        fast_lio_path = f"{self.config.fast_lio_dir}/{self.config.fast_lio_script}"
        cmd = f"test -f {fast_lio_path} && echo 'exists'"
        _, out, _ = self.container.exec_in_container(cmd)
        if "exists" not in out:
            logger.error(f"Fast-LIO script not found: {fast_lio_path}")
            return False

        logger.info("Pre-launch checks passed")
        return True

    def start(self) -> bool:
        command = self.get_start_command()
        log_file = self.get_log_file()
        logger.info(f"Starting Fast-LIO: {command}")
        result = self.container.start_process(
            command=command,
            log_file=log_file,
            verify_pattern=None,
            wait_time=0.5,
        )
        if not result:
            return False

        logger.info(
            f"Waiting {self.config.fast_lio_wait_time}s for Fast-LIO nodes to initialize..."
        )
        time.sleep(self.config.fast_lio_wait_time)
        return self.verify()

    def launch(self) -> bool:
        if self.restart_container:
            logger.info("Restarting lio container...")
            if not self.container.restart_container(wait_time=5):
                return False
        return super().launch()

    def stop_all(self) -> bool:
        logger.info("Stopping Fast-LIO (restarting container)...")
        return self.container.restart_container()

    def status(self) -> dict:
        fast_lio_pid = self.container.get_process_pid(self.config.fast_lio_process_pattern)
        container_running = self.container.is_container_running()
        status_dict = {
            "container_running": container_running,
            "container": self.container.container_name,
            "fast_lio": {
                "running": bool(fast_lio_pid),
                "pid": fast_lio_pid,
                "log_file": self.config.fast_lio_log_path,
                "process_pattern": self.config.fast_lio_process_pattern,
            },
        }
        logger.info(
            f"Fast-LIO ({self.config.fast_lio_process_pattern}): "
            f"{'running' if fast_lio_pid else 'stopped'} (PID: {fast_lio_pid})"
        )
        return status_dict

    def post_launch_actions(self) -> None:
        logger.info("Fast-LIO is ready (DDS topics: lidar/raw_odometry, lidar/odometry)")


def is_dds_localization_running(timeout_ms: int = 300) -> bool:
    """True if lidar/raw_odometry is publishing on the local DDS domain."""
    state = {"running": False}

    def handler(msg):
        state["running"] = True

    check_sub = ChannelSubscriber("lidar/raw_odometry", OdometryData)
    check_sub.Init(handler, 10)

    deadline = time.time() + timeout_ms / 1000.0
    while time.time() < deadline:
        if state["running"]:
            return True
        time.sleep(0.02)
    return False


def launch_fast_lio(
    host: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    container_name: str = "yact_lio",
    restart_container: bool = True,
) -> bool:
    """SSH to the Jetson, start the lio container, and run Fast-LIO."""
    host, user, password = _resolve_jetson_credentials(host, user, password)
    ssh_config = SSHConfig(host=host, user=user, password=password)
    docker_config = DockerConfig(
        container_name=container_name,
    )

    launcher = FastLioLauncher(
        ssh_config=ssh_config,
        container_name=container_name,
        fast_lio_config=FastLioConfig(),
        docker_config=docker_config,
        restart_container=restart_container,
    )
    return launcher.launch()


# Backward-compatible alias (bridge removed; Fast-LIO publishes DDS directly).
launch_localization_bridge = launch_fast_lio


def stop_all_services(
    host: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    container_name: str = "yact_lio",
) -> bool:
    host, user, password = _resolve_jetson_credentials(host, user, password)
    ssh_config = SSHConfig(host=host, user=user, password=password)
    launcher = FastLioLauncher(
        ssh_config=ssh_config,
        container_name=container_name,
        fast_lio_config=FastLioConfig(),
        docker_config=DockerConfig(container_name=container_name),
        restart_container=False,
    )
    return launcher.stop_all()


def get_status(
    host: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
    container_name: str = "yact_lio",
) -> dict:
    host, user, password = _resolve_jetson_credentials(host, user, password)
    ssh_config = SSHConfig(host=host, user=user, password=password)
    launcher = FastLioLauncher(
        ssh_config=ssh_config,
        container_name=container_name,
        fast_lio_config=FastLioConfig(),
        restart_container=False,
    )
    return launcher.status()


def main() -> None:
    if not launch_fast_lio(restart_container=True):
        raise RuntimeError("Failed to launch Fast-LIO")


if __name__ == "__main__":
    main()
