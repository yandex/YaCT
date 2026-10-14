#!/usr/bin/env python3
"""Generate the forward-walk VR replay trajectory.

Drives the CommandedVRPoseTracker used by `--wasd` with a scripted command
(stand, walk straight ahead, stand) on a 50 Hz simulated clock and saves the
poses in the VR replay format. Run inside the sim image from the repository root:

    python3 scripts/generate_walk_trajectory.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lib.deploy.vr.commanded_pose_tracker import CommandedVRPoseTracker

CONTROL_DT = 0.02  # state timestamp step, as in recorded VR replays (50 Hz)
SMOOTHING_ALPHA = 0.12  # same as the --wasd tracker in lib/controller/scripts/deploy.py
DEFAULT_OUTPUT = _ROOT / "assets" / "trajectories" / "walk_forward.json"


class ScriptedWalk:
    """Clock and commander for the tracker: stand, walk forward at `speed`, stand."""

    def __init__(self, stand_s: float, walk_s: float, speed: float) -> None:
        self.stand_s, self.walk_s, self.speed = stand_s, walk_s, speed
        self.time = 0.0

    def get_env_time(self) -> float:
        return self.time

    def get_command(self) -> list[float]:
        walking = self.stand_s <= self.time < self.stand_s + self.walk_s
        return [self.speed if walking else 0.0, 0.0, 0.0]


def generate(stand_s: float, walk_s: float, speed: float) -> dict:
    script = ScriptedWalk(stand_s, walk_s, speed)
    tracker = CommandedVRPoseTracker(
        env=script, control_dt=CONTROL_DT, smoothing_alpha=SMOOTHING_ALPHA
    )
    tracker.set_commander(script)
    states = []
    for step in range(round((2 * stand_s + walk_s) / CONTROL_DT)):
        script.time = step * CONTROL_DT
        states.append(tracker.get_state().model_dump())
    return {"recorded_states": states, "num_states": len(states)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the forward-walk VR replay trajectory")
    parser.add_argument("--speed", type=float, default=0.3, help="forward speed, m/s")
    parser.add_argument("--walk-seconds", type=float, default=6.0)
    parser.add_argument(
        "--stand-seconds", type=float, default=1.0, help="standing time before and after the walk"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    data = generate(args.stand_seconds, args.walk_seconds, args.speed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data), encoding="utf-8")
    print(f"wrote {data['num_states']} states to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
