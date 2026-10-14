from datetime import datetime
import logging
import os
from typing import Callable, Optional

from lib.deploy.control.audio_alert_player import AudioAlertPlayer


def get_log_path(name: str, extension: str, logs_dir: str = "./logs"):
    return os.path.join(logs_dir, f"{name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.{extension}")


def _display_fsm_node(name: str) -> str:
    """Drop auto-suffixes like ``-0`` for readable terminal output."""
    if name.endswith("-0"):
        return name[:-2]
    return name


def format_fsm_graph(graph: dict, current: Optional[str] = None, *, color: Optional[bool] = None) -> str:
    """Render an FSM graph (from ``TeleopController.fsm_graph``) as a one-glance chain
    with the current node highlighted and the trigger hints noted."""
    import sys

    if color is None:
        color = bool(getattr(sys.stdout, "isatty", lambda: False)())
    green, bold, dim, reset = ("\033[32m", "\033[1m", "\033[2m", "\033[0m") if color else ("",) * 4

    names = [n["name"] for n in graph["nodes"]]
    paused = [n["name"] for n in graph["nodes"] if n.get("paused")]

    def fmt(name: str) -> str:
        return f"{green}{bold}▸{name}◂{reset}" if name == current else name

    # Forward chain; the last two nodes (DoTeleop <-> ReturnMiddle) form a loop.
    chain = " → ".join(fmt(n) for n in names[:-1])
    if len(names) >= 2:
        chain += f" ⇄ {fmt(names[-1])}"

    lines = [
        f"{bold}FSM{reset}  —  advance: number key or VR-B   ·   emergency damp: x or VR-A",
        "",
        "  " + chain,
    ]
    if paused:
        lines.append(
            f"  {dim}⏸ {', '.join(paused)} pause the sim (no physics, static view) — "
            f"advance with 1 or VR-B.{reset}"
        )
    return "\n".join(lines)


class DeployConsole:
    _CONFIGURED_LOGGERS = set()
    _terminal_writer: Optional[Callable[[str], None]] = None

    @classmethod
    def set_terminal_writer(cls, writer: Optional[Callable[[str], None]]) -> None:
        """Route terminal lines through ``writer`` (e.g. ``tqdm.write``) to avoid bar corruption."""
        cls._terminal_writer = writer

    @classmethod
    def _write_terminal(cls, message: str) -> None:
        if cls._terminal_writer is not None:
            cls._terminal_writer(message)
        else:
            print(message, flush=True)

    def __init__(
        self,
        name: str,
        *,
        level: int = logging.INFO,
        log_file: Optional[str] = None,
        audio_alert_player: Optional[AudioAlertPlayer] = None,
        quiet: bool = False,
    ):
        self.quiet = quiet
        try:
            from rich.console import Console
            from rich.text import Text
        except ImportError:
            self._rich = False
            self.console = None
            self._Text = None
        else:
            self._rich = True
            self._Text = Text
            self.console = Console(force_terminal=True)
        self.logger = None
        self._audio_alert_player = audio_alert_player

        if log_file is not None:
            self.logger = logging.getLogger(f"{name}.deploy_console_file")
            self.logger.setLevel(level)
            self.logger.propagate = False

            if name not in self._CONFIGURED_LOGGERS:
                os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
                file_handler = logging.FileHandler(log_file, mode="a")
                file_handler.setFormatter(
                    logging.Formatter(
                        "[%(asctime)s.%(msecs)03d] [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                    )
                )
                self.logger.addHandler(file_handler)
                self._CONFIGURED_LOGGERS.add(name)

    def _emit(
        self,
        level: int,
        message: str,
        style: str,
        *,
        alert_id: Optional[str] = None,
        force_terminal: bool = False,
    ) -> None:
        if self.quiet and level <= logging.INFO and not force_terminal:
            if self.logger is not None:
                self.logger.log(level, message)
            return
        if self._terminal_writer is not None:
            self._write_terminal(message)
        elif self._rich:
            assert self.console is not None and self._Text is not None
            self.console.print(self._Text(message, style=style))
        else:
            print(message, flush=True)
        if self.logger is not None:
            self.logger.log(level, message)
        if alert_id is not None and self._audio_alert_player is not None:
            self._audio_alert_player.play(alert_id)

    def info(self, message: str, *, alert_id: Optional[str] = None) -> None:
        self._emit(logging.INFO, message, "cyan", alert_id=alert_id)

    def highlight(self, message: str, *, alert_id: Optional[str] = None) -> None:
        self._emit(logging.INFO, message, "bold bright_cyan", alert_id=alert_id)

    def warning(self, message: str, *, alert_id: Optional[str] = None) -> None:
        self._emit(logging.WARNING, message, "yellow", alert_id=alert_id)

    def critical(self, message: str, *, alert_id: Optional[str] = None) -> None:
        self._emit(logging.CRITICAL, message, "bold red", alert_id=alert_id)

    def stop_audio(self) -> None:
        if self._audio_alert_player is not None:
            self._audio_alert_player.stop()

    def play_alert(self, alert_id: str) -> None:
        if self._audio_alert_player is not None:
            self._audio_alert_player.play(alert_id)

    def transition(
        self,
        source_name: str,
        target_name: str,
        *,
        cause: Optional[str] = None,
    ) -> None:
        """Compact one-line FSM transition for release-style output."""
        src = _display_fsm_node(source_name)
        dst = _display_fsm_node(target_name)
        message = f"FSM: {src} → {dst}"
        if cause and cause != "Outer loop":
            message = f"{message} ({cause})"
        if self.quiet:
            self._emit(logging.INFO, message, "cyan", force_terminal=True)
            return
        self.highlight(f"Moving along edge: {source_name} -> {target_name}")
        if cause is not None:
            self.highlight(f"Cause: {cause}")


def emit_sleep_warnings(
    console: DeployConsole,
    time_slept: float,
    oversleep_time: Optional[float],
) -> None:
    if console.quiet:
        return
    if time_slept < -0.005:
        console.warning(f"DEADLINE MISSED BY {-time_slept * 1000:.1f}ms")

    if oversleep_time is not None and oversleep_time > 0.005:
        console.warning(
            f"SLEEP OVERSHOT DEADLINE BY {oversleep_time * 1000:.1f}ms",
        )
