import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional

import numpy as np
import yaml


@dataclass
class MotorTemperatureCfg:
    warning_threshold: float
    critical_threshold: float


@dataclass
class AudioAlertSpec:
    """One G1 deploy alert sound (16 kHz mono PCM WAV path)."""

    wav_path: str
    cooldown_s: float = 20.0
    #: Linear amplitude multiplier (0 = silent, 1 = as in file). ``None`` means 1.0.
    volume: Optional[float] = None


class AudioAlertKey(str, Enum):
    """Built-in alert ids used from Python (values match YAML ``audio_alerts`` keys)."""

    MOTOR_TEMP_WARNING = "motor_temp_warning"


@dataclass
class AudioAlertsCfg:
    """WAV alerts keyed by name (same keys as under YAML ``audio_alerts``)."""

    specs: Mapping[str, AudioAlertSpec] = field(default_factory=dict)

    def get(self, alert_id: str) -> Optional[AudioAlertSpec]:
        return self.specs.get(alert_id)

    def player_needed(self) -> bool:
        return bool(self.specs)


def _parse_audio_alerts_raw(raw: Any) -> Optional[AudioAlertsCfg]:
    """Build :class:`AudioAlertsCfg` from YAML ``audio_alerts`` mapping.

    Each key is an alert id; each value is ``null`` or a mapping with ``wav_path`` and optional
    ``cooldown_s``, ``volume`` (linear gain, 0–1 typical; ``None`` / omitted = full scale).
    New alert types only require YAML (and a :class:`AudioAlertKey` constant if triggered from code).
    """
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise TypeError(f"audio_alerts must be a mapping, got {type(raw).__name__}")
    specs: Dict[str, AudioAlertSpec] = {}
    for name, entry in raw.items():
        if entry is None:
            continue
        if not isinstance(entry, dict):
            raise TypeError(
                f"audio_alerts[{name!r}] must be a mapping or null, got {type(entry).__name__}"
            )
        if "wav_path" not in entry:
            raise KeyError(f"audio_alerts[{name!r}]: missing required key 'wav_path'")
        vol_raw = entry.get("volume")
        volume: Optional[float]
        if vol_raw is None:
            volume = None
        else:
            volume = float(vol_raw)
            if volume < 0.0:
                raise ValueError(f"audio_alerts[{name!r}]: volume must be >= 0, got {volume}")
        specs[str(name)] = AudioAlertSpec(
            wav_path=entry["wav_path"],
            cooldown_s=float(entry.get("cooldown_s", 20.0)),
            volume=volume,
        )
    return AudioAlertsCfg(specs=specs) if specs else None


class Config:
    def __init__(self, file_path) -> None:
        self.div = float(os.environ.get("DIV", 1.))
        assert self.div >= 1.

        with open(file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            self.control_dt = config["control_dt"]

            self.imu_type = config["imu_type"]

            self.lowcmd_topic = config["lowcmd_topic"]
            self.lowstate_topic = config["lowstate_topic"]

            self.kps = np.array(config["kps"]) / self.div
            self.kds = np.array(config["kds"]) / self.div

            self.joint_limits = np.array(config["joint_limits"])

            self.default_position = np.array(config["default_position"], dtype=np.float32)
            self.action_mask = np.array(config["action_mask"], dtype=bool)

            assert len(self.kps) == len(self.kds) == len(self.default_position) == len(self.action_mask)
            self.num_joints = len(self.kps)

            motor_temp = config.get("motor_temperature")
            self.motor_temperature: Optional[MotorTemperatureCfg] = (
                MotorTemperatureCfg(
                    warning_threshold=motor_temp["warning_threshold"],
                    critical_threshold=motor_temp["critical_threshold"],
                )
                if motor_temp is not None else None
            )

            self.audio_alerts = _parse_audio_alerts_raw(config.get("audio_alerts"))
