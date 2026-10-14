"""Optional deploy WAV alerts with per-alert wall-clock cooldowns."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Dict, Optional

from lib.deploy.control.config import AudioAlertSpec, AudioAlertsCfg

if TYPE_CHECKING:
    from lib.deploy.control.g1_audio_player import G1AudioPlayer


class AudioAlertPlayer:
    """Wraps :class:`G1AudioPlayer` and :class:`AudioAlertsCfg` for one-liner call sites.

    ``alert_id`` is the string key under YAML ``audio_alerts`` (see :class:`AudioAlertKey` for built-in names).
    """

    def __init__(
        self,
        network: Optional[str],
        alerts: Optional[AudioAlertsCfg] = None,
    ) -> None:
        self._alerts = alerts
        self._last_play_wall_s: Dict[str, float] = {}
        self._g1: Optional[G1AudioPlayer] = None
        if alerts is not None and alerts.player_needed():
            from lib.deploy.control.g1_audio_player import G1AudioPlayer

            self._g1 = G1AudioPlayer(network=network)

    def play(self, alert_id: str) -> None:
        if self._g1 is None or self._alerts is None:
            return
        spec = self._alerts.get(alert_id)
        if spec is None:
            return
        now = time.time()
        last = self._last_play_wall_s.get(alert_id, 0.0)
        if now - last < spec.cooldown_s:
            return
        self._last_play_wall_s[alert_id] = now
        self._g1.play(spec.wav_path, volume=spec.volume)

    def stop(self) -> None:
        if self._g1 is not None:
            self._g1.stop()
