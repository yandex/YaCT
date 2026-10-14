"""Browser keyboard bridge for the live view → InputReader."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from lib.controller.utils.input import InputReader


class BrowserInputBridge:
    """Translate browser keydown/keyup events into InputReader commands."""

    def __init__(self, reader: InputReader, *, motion_enabled: bool = True):
        self._reader = reader
        self._motion_enabled = motion_enabled
        self._held_vel_keys: set[str] = set()

    def set_motion_enabled(self, enabled: bool) -> None:
        self._motion_enabled = enabled
        if not enabled:
            self._held_vel_keys.clear()
            self._reader.set_velocity(np.zeros(3, dtype=np.float32))

    def handle_key(self, key: str, action: str) -> None:
        action = action.lower()
        if action == "clear":
            self._held_vel_keys.clear()
            self._reader.set_velocity(np.zeros(3, dtype=np.float32))
            return

        if key.lower() == "space":
            key = " "
        elif len(key) == 1:
            key = key.lower()

        vel_map = self._reader.vel_map

        if action == "keydown":
            if key in vel_map:
                if not self._motion_enabled:
                    return
                if key == " ":
                    self._held_vel_keys.clear()
                    self._reader.set_velocity(np.zeros(3, dtype=np.float32))
                else:
                    self._held_vel_keys.add(key)
                    self._reader.set_velocity(self._compute_held_velocity(vel_map))
            elif key == "x":
                self._reader.push_fsm(
                    self._reader.damp_node_name,
                    cause="Browser key X pressed",
                )
            elif len(key) == 1 and key.isdigit():
                self._reader.push_fsm(int(key), cause=f"Browser key {key} pressed")
            elif key == "r":
                self._reader.reset_sim()
        elif action == "keyup":
            if key in vel_map and self._motion_enabled:
                self._held_vel_keys.discard(key)
                if self._held_vel_keys:
                    self._reader.set_velocity(self._compute_held_velocity(vel_map))
                else:
                    self._reader.set_velocity(np.zeros(3, dtype=np.float32))

    def _compute_held_velocity(self, vel_map: dict) -> np.ndarray:
        vel = np.zeros(3, dtype=np.float32)
        for held_key in self._held_vel_keys:
            vel += vel_map[held_key]
        return vel
