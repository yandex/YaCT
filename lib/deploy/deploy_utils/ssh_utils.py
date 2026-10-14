import logging
import os
import subprocess
import tempfile
import paramiko
import time
from dataclasses import dataclass
from typing import Optional, Callable
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class SSHConfig:
    """SSH connection configuration"""
    host: str
    user: str
    password: str
    timeout: int = 60


@dataclass
class DockerConfig:
    """Docker container startup configuration for remote Jetson."""

    image: str = None
    # Registry source used to stage `image` onto the robot when it is missing
    # (the robot may have no registry access; we pull here and ship over SSH).
    registry_image: str = None
    platform: str = "linux/arm64"
    container_name: str = "yact_lio"
    # Local Dockerfile + context used to build `image` when the registry pull
    # is unavailable (absolute paths). Optional; no build fallback if unset.
    dockerfile: str = None
    build_context: str = None
    env_vars: dict = None
    mounts: list = None

    def __post_init__(self):
        if self.image is None or self.registry_image is None:
            from lib.cli.paths import load_image_config, registry_image_ref

            cfg = load_image_config("lio")
            if self.image is None:
                self.image = f"yact-lio:{cfg['version']}"
            if self.registry_image is None:
                self.registry_image = registry_image_ref(cfg)
        if self.env_vars is None:
            self.env_vars = {
                "DISPLAY": "$DISPLAY",
                "ROS_HOSTNAME": "localhost",
                "ROS_MASTER_URI": "http://localhost:11311",
                "YACT_IN_DOCKER": "lio",
            }
        if self.mounts is None:
            self.mounts = [
                "$HOME/.Xauthority",
                "/tmp/.X11-unix",
                "/dev/bus/usb",
            ]

    def get_start_command(self) -> str:
        """Generate docker run command (executed on the remote host via SSH)."""
        env_flags = " ".join(f'-e "{k}={v}"' for k, v in self.env_vars.items())
        mount_flags = " ".join(f'-v {m}:{m}' for m in self.mounts)
        return (
            f"docker run -d --entrypoint \"\" --name {self.container_name} --net host --privileged "
            f"{env_flags} {mount_flags} {self.image} sleep infinity"
        )

    def get_stop_command(self) -> str:
        """Generate docker stop/rm command."""
        return f"docker rm -f {self.container_name}"

class SSHExecutor:
    """Handles SSH command execution"""

    def __init__(self, config: SSHConfig):
        self.config = config

    def run(self, command: str, timeout: Optional[int] = None, verbose: bool = True) -> tuple[int, str, str]:
        """Execute command via SSH and return (exit_code, stdout, stderr)"""
        timeout = timeout or self.config.timeout

        if verbose:
            logger.debug(f"SSH command: {command}")

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            ssh.connect(
                hostname=self.config.host,
                username=self.config.user,
                password=self.config.password,
                timeout=timeout,
                allow_agent=False,
                look_for_keys=False
            )

            stdin, stdout, stderr = ssh.exec_command(command)
            out = stdout.read().decode(errors="replace")
            err = stderr.read().decode(errors="replace")
            exit_status = stdout.channel.recv_exit_status()

            if verbose:
                if exit_status == 0:
                    logger.debug(f"SSH command succeeded (exit code: {exit_status})")
                    if out.strip():
                        logger.debug(f"stdout: {out.strip()}")
                else:
                    logger.error(f"SSH command failed (exit code: {exit_status})")
                    if out.strip():
                        logger.error(f"stdout: {out.strip()}")
                    if err.strip():
                        logger.error(f"stderr: {err.strip()}")

            return exit_status, out, err
        except Exception as e:
            logger.error(f"SSH connection/execution failed: {e}")
            logger.error(f"Host: {self.config.host}, User: {self.config.user}")
            return 1, "", str(e)
        finally:
            ssh.close()

    def put_file(self, local_path: str, remote_path: str, timeout: Optional[int] = None) -> bool:
        """Upload a local file to the remote host via SFTP."""
        timeout = timeout or self.config.timeout
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            ssh.connect(
                hostname=self.config.host,
                username=self.config.user,
                password=self.config.password,
                timeout=timeout,
                allow_agent=False,
                look_for_keys=False,
            )
            sftp = ssh.open_sftp()
            try:
                sftp.put(local_path, remote_path)
            finally:
                sftp.close()
            return True
        except Exception as e:
            logger.error(f"SFTP upload failed ({local_path} -> {remote_path}): {e}")
            return False
        finally:
            ssh.close()


def _run_local(cmd: list) -> bool:
    """Run a command on the local (deploy) host, logging it."""
    logger.info("local: %s", " ".join(cmd))
    return subprocess.run(cmd).returncode == 0


# Shared with scripts/build_and_push.py — a docker-container buildx builder that
# supports cross-arch builds (arm64 lio image) via QEMU.
_BUILDX_BUILDER = "yact-multiarch"


def _ensure_buildx_builder() -> bool:
    """Ensure the buildx builder exists; create it if missing.

    Returns False if docker buildx is unavailable (create fails).
    """
    exists = subprocess.run(
        ["docker", "buildx", "inspect", _BUILDX_BUILDER],
        capture_output=True, text=True,
    ).returncode == 0
    if exists:
        return True
    logger.info("Creating buildx builder '%s'...", _BUILDX_BUILDER)
    # network=host so the buildkit container can reach the base-image registry.
    return _run_local([
        "docker", "buildx", "create",
        "--name", _BUILDX_BUILDER,
        "--driver", "docker-container",
        "--driver-opt", "network=host",
    ])


class ContainerProcess:
    """Manages a process running inside a Docker container"""

    def __init__(
        self,
        ssh_executor: SSHExecutor,
        container_name: str,
        docker_config: Optional[DockerConfig] = None
    ):
        self.ssh = ssh_executor
        self.container_name = container_name
        self.docker_config = docker_config or DockerConfig()

    def exec_in_container(self, command: str, detached: bool = False, verbose: bool = True) -> tuple[int, str, str]:
        """Execute command inside container"""
        flag = "-d" if detached else ""
        escaped_command = command.replace("'", "'\"'\"'")
        cmd = f"docker exec {flag} {self.container_name} bash -c '{escaped_command}'"

        if verbose:
            logger.debug(f"Container exec: {command}")

        return self.ssh.run(cmd, verbose=verbose)

    def is_container_running(self) -> bool:
        """Check if container exists and is running"""
        status, out, _ = self.ssh.run(f"docker ps | grep {self.container_name}", verbose=False)
        return self.container_name in out

    def is_container_exists(self) -> bool:
        """Check if container exists (running or stopped)"""
        status, out, _ = self.ssh.run(f"docker ps -a | grep {self.container_name}", verbose=False)
        return self.container_name in out

    def _build_image_locally(self) -> bool:
        """Build ``docker_config.image`` from its Dockerfile, for the robot's arch.

        Fallback when the registry is unreachable. Uses ``docker buildx`` so it
        can cross-build for ``platform`` (arm64 for the Jetson) and ``--load``
        the result into the local image store to ship over SSH. On an amd64 host
        the arm64 build runs under QEMU — register binfmt once with
        ``docker run --privileged --rm tonistiigi/binfmt --install arm64``.
        """
        cfg = self.docker_config
        if not cfg.dockerfile or not cfg.build_context:
            logger.error("No Dockerfile/build context configured; cannot build '%s' locally", cfg.image)
            return False
        if not _ensure_buildx_builder():
            logger.error(
                "docker buildx is unavailable; cannot build the arm64 lio image locally. "
                "Install buildx (ships with modern Docker), or build the image on the robot."
            )
            return False
        logger.info("Building '%s' with buildx (%s) from %s ...", cfg.image, cfg.platform, cfg.dockerfile)
        return _run_local([
            "docker", "buildx", "build",
            "--builder", _BUILDX_BUILDER,
            "--platform", cfg.platform,
            "-f", cfg.dockerfile,
            "-t", cfg.image,
            "--load",
            cfg.build_context,
        ])

    def ensure_image_present(self) -> bool:
        """Ensure ``docker_config.image`` exists on the robot, staging it if not.

        The robot may have no registry access, so when the image is absent we
        pull it on this (deploy) host and ship it over SSH (``docker save`` ->
        SFTP -> ``docker load``). No-op if the image is already present.
        """
        cfg = self.docker_config
        status, out, _ = self.ssh.run(f"docker images -q {cfg.image}", verbose=False)
        if status == 0 and out.strip():
            logger.info(f"✓ Image '{cfg.image}' already present on robot")
            return True

        logger.info(
            f"Image '{cfg.image}' not on robot; pulling '{cfg.registry_image}' "
            f"locally and transferring over SSH..."
        )
        if _run_local(["docker", "pull", "--platform", cfg.platform, cfg.registry_image]):
            if cfg.image != cfg.registry_image and not _run_local(
                ["docker", "tag", cfg.registry_image, cfg.image]
            ):
                return False
        else:
            logger.warning(
                "Local pull of '%s' failed (no registry access?); "
                "falling back to a local build of '%s'.",
                cfg.registry_image, cfg.image,
            )
            if not self._build_image_locally():
                logger.error(
                    "Local pull and build both failed; cannot stage the image. "
                    "Give this host registry access, or build the lio image first."
                )
                return False

        with tempfile.NamedTemporaryFile(prefix="yact-lio-", suffix=".tar", delete=False) as tmp:
            tar_path = tmp.name
        try:
            if not _run_local(["docker", "save", "-o", tar_path, cfg.image]):
                return False
            remote_tar = f"/tmp/{os.path.basename(tar_path)}"
            size_mb = os.path.getsize(tar_path) // (1024 * 1024)
            logger.info(f"Uploading image ({size_mb} MB) to robot...")
            if not self.ssh.put_file(tar_path, remote_tar):
                return False
            status, out, err = self.ssh.run(f"docker load -i {remote_tar}", timeout=600)
            self.ssh.run(f"rm -f {remote_tar}", verbose=False)  # clean up regardless
            if status != 0:
                logger.error(f"Remote 'docker load' failed: {err or out}")
                return False
        finally:
            try:
                os.unlink(tar_path)
            except OSError:
                pass
        logger.info(f"✓ Image '{cfg.image}' transferred to robot")
        return True

    def start_container(self, wait_time: float = 5.0) -> bool:
        """
        Start the Docker container

        Args:
            wait_time: Time to wait for container to be ready

        Returns:
            True if container started successfully
        """
        logger.info(f"Starting container '{self.container_name}'...")

        start_cmd = self.docker_config.get_start_command()
        logger.info(f"Docker start command: {start_cmd}")

        status, out, err = self.ssh.run(start_cmd, timeout=120)

        if status != 0:
            logger.error(f"Failed to start container (exit code: {status})")
            logger.error(f"Command: {start_cmd}")
            if out:
                logger.error(f"stdout: {out}")
            if err:
                logger.error(f"stderr: {err}")
            return False

        logger.info(f"Container start command executed, waiting {wait_time}s for readiness...")
        time.sleep(wait_time)

        # Verify container is running
        if self.is_container_running():
            logger.info(f"✓ Container '{self.container_name}' started successfully")
            return True
        else:
            logger.error(f"✗ Container '{self.container_name}' failed to start")
            # Get container logs for debugging
            log_cmd = f"docker logs --tail 50 {self.container_name}"
            status, out, err = self.ssh.run(log_cmd, verbose=False)
            if out:
                logger.error(f"Container logs:\n{out}")
            return False

    def ensure_container_running(self, auto_start: bool = True) -> bool:
        """
        Ensure container is running, optionally start it if not

        Args:
            auto_start: Automatically start container if not running

        Returns:
            True if container is running
        """
        if self.is_container_running():
            logger.info(f"✓ Container '{self.container_name}' is already running")
            return True

        if not auto_start:
            logger.error(f"✗ Container '{self.container_name}' is not running")
            return False

        logger.warning(f"Container '{self.container_name}' is not running, attempting to start...")
        return self.start_container()

    def is_process_running(self, process_pattern: str) -> bool:
        """Check if a process matching pattern is running in container"""
        cmd = f"pgrep -f '{process_pattern}'"
        status, out, _ = self.exec_in_container(cmd, verbose=False)
        return status == 0 and out.strip()

    def get_process_pid(self, process_pattern: str) -> Optional[str]:
        """Get PID of process matching pattern"""
        cmd = f"pgrep -f '{process_pattern}'"
        status, out, _ = self.exec_in_container(cmd, verbose=False)
        return out.strip() if status == 0 else None

    def kill_process(self, process_pattern: str) -> bool:
        """Kill process matching pattern"""
        logger.debug(f"Attempting to kill process: {process_pattern}")
        cmd = f"pkill -f '{process_pattern}' || true"
        status, out, err = self.exec_in_container(cmd)
        time.sleep(0.5)

        # Verify process was killed
        if self.is_process_running(process_pattern):
            logger.warning(f"Process '{process_pattern}' still running after kill attempt")
        else:
            logger.debug(f"Process '{process_pattern}' successfully killed or was not running")

        return status == 0

    def start_process(
        self,
        command: str,
        log_file: Optional[str] = None,
        verify_pattern: Optional[str] = None,
        wait_time: float = 2.0
    ) -> bool:
        """
        Start a process in the container

        Args:
            command: Command to execute
            log_file: Optional log file path for stdout/stderr
            verify_pattern: Pattern to verify process started (default: uses command)
            wait_time: Time to wait before verification
        """
        full_command = command
        if log_file:
            full_command = f"{command} > {log_file} 2>&1"

        logger.info(f"Starting process in container...")
        logger.info(f"Command: {command}")
        if log_file:
            logger.info(f"Log file: {log_file}")

        status, out, err = self.exec_in_container(full_command, detached=True)

        if status != 0:
            logger.error(f"Failed to start process (exit code: {status})")
            logger.error(f"Full command: {full_command}")
            if out.strip():
                logger.error(f"stdout: {out}")
            if err.strip():
                logger.error(f"stderr: {err}")
            return False

        logger.info(f"Process start command executed, waiting {wait_time}s...")
        time.sleep(wait_time)

        # Verify if requested
        if verify_pattern is not None or verify_pattern is None:
            pattern = verify_pattern if verify_pattern else command.split()[0]
            logger.debug(f"Verifying process with pattern: {pattern}")

            is_running = self.is_process_running(pattern)
            if is_running:
                logger.info(f"✓ Process is running")
            else:
                logger.error(f"✗ Process is not running")
                logger.error(f"Pattern searched: {pattern}")

                # Show running processes for debugging
                status, out, _ = self.exec_in_container("ps aux", verbose=False)
                logger.error(f"Running processes in container:\n{out}")

                # Try to get logs if available
                if log_file:
                    logger.error(f"Attempting to retrieve logs from {log_file}...")
                    logs = self.get_logs(log_file, lines=50)
                    if logs and "No logs found" not in logs:
                        logger.error(f"Process logs:\n{logs}")
                    else:
                        logger.error(f"No logs found at {log_file}")

            return is_running

        return True

    def restart_container(self, wait_time: float = 5.0) -> bool:
        """
        Restart the Docker container

        Args:
            wait_time: Time to wait for container to be ready after restart

        Returns:
            True if container restarted successfully
        """
        logger.info(f"Restarting container '{self.container_name}'...")

        # Stop container
        stop_cmd = self.docker_config.get_stop_command()
        logger.info(f"Stopping container...")
        status, out, err = self.ssh.run(stop_cmd, timeout=30)

        if status != 0:
            logger.warning(f"Container stop returned non-zero exit code: {status}")
            # Continue anyway, container might not be running

        time.sleep(2)  # Brief wait after stop

        # Start container
        start_cmd = self.docker_config.get_start_command()
        logger.info(f"Starting container...")
        logger.debug(f"Start command: {start_cmd}")

        status, out, err = self.ssh.run(start_cmd, timeout=120)

        if status != 0:
            logger.error(f"Failed to restart container (exit code: {status})")
            if out:
                logger.error(f"stdout: {out}")
            if err:
                logger.error(f"stderr: {err}")
            return False

        logger.info(f"Waiting {wait_time}s for container to be ready...")
        time.sleep(wait_time)

        # Verify container is running
        if self.is_container_running():
            logger.info(f"✓ Container '{self.container_name}' restarted successfully")
            return True
        else:
            logger.error(f"✗ Container '{self.container_name}' failed to restart")
            return False

    def get_logs(self, log_file: str, lines: int = 50) -> Optional[str]:
        """Retrieve logs from container"""
        cmd = f"tail -{lines} {log_file} 2>/dev/null || echo 'No logs found'"
        status, out, _ = self.exec_in_container(cmd, verbose=False)
        return out if status == 0 else None


class RemoteServiceLauncher(ABC):
    """Abstract base class for service launchers"""

    def __init__(
        self,
        ssh_config: SSHConfig,
        container_name: str,
        docker_config: Optional[DockerConfig] = None,
        auto_start_container: bool = True
    ):
        self.ssh_executor = SSHExecutor(ssh_config)
        self.container = ContainerProcess(self.ssh_executor, container_name, docker_config)
        self.auto_start_container = auto_start_container

    @abstractmethod
    def get_process_pattern(self) -> str:
        """Return pattern to identify the process"""
        pass

    @abstractmethod
    def get_start_command(self) -> str:
        """Return command to start the service"""
        pass

    @abstractmethod
    def get_log_file(self) -> str:
        """Return log file path"""
        pass

    def pre_launch_checks(self) -> bool:
        """Override to add custom pre-launch checks"""
        return True

    def post_launch_actions(self) -> None:
        """Override to add custom post-launch actions"""
        pass

    def check_container(self) -> bool:
        """Verify container is running, start it if needed"""
        logger.info(f"Checking container '{self.container.container_name}'...")
        result = self.container.ensure_container_running(auto_start=self.auto_start_container)
        if not result:
            logger.error("Container check/start failed - cannot proceed")
        return result

    def stop(self) -> bool:
        """Stop the service"""
        logger.info("Stopping existing process...")
        pattern = self.get_process_pattern()

        # Check if process is running first
        if self.container.is_process_running(pattern):
            logger.info(f"Found running process matching: {pattern}")
            result = self.container.kill_process(pattern)
            if result:
                logger.info("✓ Process stopped successfully")
            else:
                logger.warning("Process stop command executed with warnings")
            return result
        else:
            logger.info(f"No process matching '{pattern}' is currently running")
            return True

    def start(self) -> bool:
        """Start the service"""
        logger.info("="*60)
        logger.info("Starting service...")
        logger.info("="*60)

        command = self.get_start_command()
        log_file = self.get_log_file()
        pattern = self.get_process_pattern()

        logger.info(f"Start command: {command}")
        logger.info(f"Process pattern: {pattern}")
        logger.info(f"Log file: {log_file}")

        result = self.container.start_process(
            command=command,
            log_file=log_file,
            verify_pattern=pattern
        )

        if not result:
            logger.error("="*60)
            logger.error("FAILED TO START SERVICE")
            logger.error("="*60)
            logger.error(f"Command that failed: {command}")
            logger.error(f"Check logs at: {log_file}")
            logger.error("="*60)

        return result

    def verify(self) -> bool:
        """Verify service is running"""
        logger.info("Verifying service status...")
        pattern = self.get_process_pattern()
        pid = self.container.get_process_pid(pattern)

        if pid:
            logger.info(f"✓ Service is running (PID: {pid})")
            return True
        else:
            logger.error(f"✗ Service is NOT running")
            logger.error(f"Process pattern searched: {pattern}")
            self.print_logs()
            return False

    def print_logs(self, lines: int = 50) -> None:
        """Print recent logs"""
        logger.info(f"Retrieving last {lines} lines of logs...")
        logs = self.container.get_logs(self.get_log_file(), lines)
        if logs and "No logs found" not in logs:
            logger.error("="*60)
            logger.error("SERVICE LOGS:")
            logger.error("="*60)
            logger.error(logs)
            logger.error("="*60)
        else:
            logger.error(f"No logs found at {self.get_log_file()}")

    def launch(self) -> bool:
        """Main launch sequence"""
        logger.info("="*60)
        logger.info(f"Launching {self.__class__.__name__}")
        logger.info("="*60)

        # Check/start container
        logger.info("Step 1/5: Container check")
        if not self.check_container():
            logger.error("Step 1/5: FAILED - Container check/start failed")
            return False
        logger.info("Step 1/5: SUCCESS - Container is running")

        # Pre-checks
        logger.info("Step 2/5: Pre-launch checks")
        if not self.pre_launch_checks():
            logger.error("Step 2/5: FAILED - Pre-launch checks failed")
            return False
        logger.info("Step 2/5: SUCCESS - Pre-launch checks passed")

        # Stop existing
        logger.info("Step 3/5: Stop existing process")
        self.stop()
        logger.info("Step 3/5: SUCCESS - Existing process stopped")

        # Start service
        logger.info("Step 4/5: Start service")
        if not self.start():
            logger.error("Step 4/5: FAILED - Service failed to start")
            return False
        logger.info("Step 4/5: SUCCESS - Service started")

        # Verify
        logger.info("Step 5/5: Verify service")
        if not self.verify():
            logger.error("Step 5/5: FAILED - Service verification failed")
            return False
        logger.info("Step 5/5: SUCCESS - Service verified")

        # Post actions
        self.post_launch_actions()

        logger.info("="*60)
        logger.info(f"✓ {self.__class__.__name__} launched successfully")
        logger.info(f"  Container: {self.container.container_name}")
        logger.info(f"  Process pattern: {self.get_process_pattern()}")
        logger.info(f"  Logs: {self.get_log_file()}")
        logger.info("="*60)
        return True

    def restart(self) -> bool:
        """Restart the service"""
        logger.info("Restarting service...")
        self.stop()
        return self.start() and self.verify()

    def status(self) -> dict:
        """Get service status"""
        pid = self.container.get_process_pid(self.get_process_pattern())
        container_running = self.container.is_container_running()

        status_dict = {
            "running": bool(pid),
            "pid": pid,
            "container": self.container.container_name,
            "container_running": container_running,
            "log_file": self.get_log_file()
        }

        logger.info("Service status:")
        for key, value in status_dict.items():
            logger.info(f"  {key}: {value}")

        return status_dict
