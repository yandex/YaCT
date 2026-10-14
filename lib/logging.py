"""Run logging: time-chunked JSONL with disk retention, plus FSM transition and controller snapshots.

Examples
--------

Generic event log (rotating files under ``log_dir``; each line is JSON with
``wall_time_utc``, ``monotonic_time``, ``event``, ``payload``)::

    from lib.logging import ChunkedJsonlLog, ChunkedJsonlLogConfig

    log = ChunkedJsonlLog(
        name="my_run",
        config=ChunkedJsonlLogConfig(
            log_dir="/tmp/yact_logs",
            chunk_duration_s=60.0,
            keep_last_n_chunks=5,
        ),
    )
    log.write_event("custom_event", {"temperature_c": 42})
    log.close()

FSM tick logger: call once per control cycle; pass it in the controller callback
list so transitions and optional snapshots are recorded (same pattern as
controller startup)::

    from lib.logging import (
        ChunkedJsonlLog,
        ChunkedJsonlLogConfig,
        ControllerFsmLogger,
    )

    chunked_log = ChunkedJsonlLog(
        name="yact_controller",
        config=ChunkedJsonlLogConfig(log_dir="/tmp/yact_controller_logs"),
    )
    fsm_logger = ControllerFsmLogger(controller, chunked_log, snapshot_every_n=2)
    fsm_logger()  # each cycle
    fsm_logger.close()
"""

import json
import os
import threading
import time
import typing as tp
from dataclasses import dataclass
from datetime import datetime, timezone


def _utc_now() -> datetime:
    """Return the current instant in UTC (timezone-aware)."""
    return datetime.now(timezone.utc)


def _utc_stamp(dt: datetime) -> str:
    """Format ``dt`` as a compact UTC timestamp for chunk filenames (second resolution)."""
    return dt.strftime("%Y%m%dT%H%M%S")


def _utc_session_stamp(dt: datetime) -> str:
    """Format ``dt`` for session IDs, including microseconds for uniqueness."""
    return dt.strftime("%Y%m%dT%H%M%S_%f")


@dataclass(frozen=True)
class ChunkedJsonlLogConfig:
    """Settings for :class:`ChunkedJsonlLog`: where to write, chunk size, and retention."""

    log_dir: str
    chunk_duration_s: float = 10.0
    keep_last_n_chunks: int = 2
    previous_session_prune_min_age_s: float = 0.0


@dataclass(frozen=True)
class ChunkedLogPathInfo:
    """One on-disk JSONL chunk and its ``.meta.json`` sidecar, after parsing."""

    path: str
    meta_path: str
    session_id: str
    chunk_index: int
    chunk_stamp: str


class ChunkedJsonlLog:
    """Append-only JSONL writer that rotates files on a wall-clock chunk interval.

    Each session gets a unique ``session_id``; chunks are named and paired with a small
    ``.meta.json`` file. Beyond ``keep_last_n_chunks`` eligible chunks are deleted;
    chunks from prior sessions are only pruned after ``previous_session_prune_min_age_s``.
    Thread-safe via an internal lock.
    """

    def __init__(self, name: str, config: ChunkedJsonlLogConfig):
        """Open ``log_dir``, start a new session, and create the first chunk file."""
        if config.chunk_duration_s <= 0:
            raise ValueError("chunk_duration_s must be > 0")
        if config.keep_last_n_chunks <= 0:
            raise ValueError("keep_last_n_chunks must be > 0")
        if config.previous_session_prune_min_age_s < 0:
            raise ValueError("previous_session_prune_min_age_s must be >= 0")

        self._name = name
        self._config = config
        self._lock = threading.Lock()
        self._fp = None
        self._chunk_index = 0

        session_started_at = _utc_now()
        self._session_id = f"{name}_{_utc_session_stamp(session_started_at)}"
        self._current_path = ""
        self._current_chunk_started_wall_time = session_started_at
        self._current_chunk_started_mono = time.monotonic()

        os.makedirs(self._config.log_dir, exist_ok=True)

        with self._lock:
            self._open_next_chunk_locked(
                chunk_started_wall_time=session_started_at,
                chunk_started_mono=self._current_chunk_started_mono,
            )

    @property
    def path(self) -> str:
        """Filesystem path of the chunk file currently being written."""
        return self._current_path

    @property
    def log_dir(self) -> str:
        """Directory containing chunk ``.jsonl`` and ``.meta.json`` files."""
        return self._config.log_dir

    @property
    def session_id(self) -> str:
        """Unique id for this writer instance's session (embedded in chunk names)."""
        return self._session_id

    @property
    def chunk_duration_s(self) -> float:
        """Seconds of monotonic time before rotating to the next chunk file."""
        return self._config.chunk_duration_s

    @property
    def keep_last_n_chunks(self) -> int:
        """Maximum number of prune-eligible chunk files to keep on disk."""
        return self._config.keep_last_n_chunks

    def write_event(self, event: str, payload: tp.Dict[str, tp.Any]):
        """Append one JSON line: UTC wall time, monotonic time, ``event``, and ``payload``."""
        row = {
            "wall_time_utc": _utc_now().isoformat(timespec="microseconds").replace("+00:00", "Z"),
            "monotonic_time": time.monotonic(),
            "event": event,
            "payload": payload,
        }
        line = json.dumps(row, ensure_ascii=True)
        with self._lock:
            self._rotate_if_needed_locked(row["monotonic_time"])
            self._fp.write(line + "\n")

    def close(self):
        """Close the current chunk file if open. Idempotent."""
        with self._lock:
            if self._fp is None:
                return
            self._fp.close()
            self._fp = None

    def _rotate_if_needed_locked(self, now_mono: float):
        """Start a new chunk when ``chunk_duration_s`` has elapsed (caller holds lock)."""
        if now_mono - self._current_chunk_started_mono < self._config.chunk_duration_s:
            return
        self._open_next_chunk_locked(
            chunk_started_wall_time=_utc_now(),
            chunk_started_mono=now_mono,
        )

    def _open_next_chunk_locked(self, chunk_started_wall_time: datetime, chunk_started_mono: float):
        """Close the previous chunk, open the next ``.jsonl``, write metadata, prune (caller holds lock)."""
        if self._fp is not None:
            self._fp.close()

        self._chunk_index += 1
        self._current_chunk_started_wall_time = chunk_started_wall_time
        self._current_chunk_started_mono = chunk_started_mono
        chunk_name = (
            f"{self._session_id}_chunk{self._chunk_index:04d}_"
            f"{_utc_stamp(chunk_started_wall_time)}.jsonl"
        )
        self._current_path = os.path.join(self._config.log_dir, chunk_name)
        self._fp = open(self._current_path, "a", encoding="utf-8", buffering=1)
        self._write_chunk_meta_locked(chunk_started_wall_time)

        self._write_row_locked(
            {
                "wall_time_utc": chunk_started_wall_time.isoformat(timespec="microseconds").replace("+00:00", "Z"),
                "monotonic_time": chunk_started_mono,
                "event": "log_chunk_started",
                "payload": {
                    "session_id": self._session_id,
                    "chunk_index": self._chunk_index,
                    "chunk_path": self._current_path,
                    "chunk_duration_s": self._config.chunk_duration_s,
                    "keep_last_n_chunks": self._config.keep_last_n_chunks,
                    "previous_session_prune_min_age_s": self._config.previous_session_prune_min_age_s,
                },
            }
        )
        self._prune_old_chunks_locked()

    def _write_row_locked(self, row: tp.Dict[str, tp.Any]):
        """Serialize ``row`` as one JSONL line to the current chunk (caller holds lock)."""
        self._fp.write(json.dumps(row, ensure_ascii=True) + "\n")

    def _write_chunk_meta_locked(self, chunk_started_wall_time: datetime):
        """Write or overwrite the sidecar ``.meta.json`` for the current chunk path."""
        meta = {
            "log_name": self._name,
            "session_id": self._session_id,
            "chunk_index": self._chunk_index,
            "chunk_stamp": _utc_stamp(chunk_started_wall_time),
            "chunk_path": self._current_path,
            "chunk_duration_s": self._config.chunk_duration_s,
            "keep_last_n_chunks": self._config.keep_last_n_chunks,
            "previous_session_prune_min_age_s": self._config.previous_session_prune_min_age_s,
        }
        meta_path = self._meta_path_for_chunk_path(self._current_path)
        with open(meta_path, "w", encoding="utf-8") as meta_fp:
            json.dump(meta, meta_fp, ensure_ascii=True, indent=2)
            meta_fp.write("\n")

    def _prune_old_chunks_locked(self):
        """Delete oldest eligible chunks (and meta) beyond ``keep_last_n_chunks``."""
        infos = sorted(
            self._iter_chunk_infos(),
            key=lambda info: (info.session_id, info.chunk_index, info.chunk_stamp),
        )
        eligible_infos = [info for info in infos if self._is_prune_eligible(info)]
        excess = len(eligible_infos) - self._config.keep_last_n_chunks
        if excess <= 0:
            return

        deletion_candidates = [
            info for info in eligible_infos
            if info.path != self._current_path
        ]
        for info in deletion_candidates[:excess]:
            os.remove(info.path)
            if os.path.exists(info.meta_path):
                os.remove(info.meta_path)

    def _iter_chunk_infos(self) -> tp.Iterator[ChunkedLogPathInfo]:
        """Yield parsed chunk info for ``.jsonl`` files in ``log_dir`` that match this log name."""
        for name in os.listdir(self._config.log_dir):
            if not name.endswith(".jsonl"):
                continue
            info = self._load_chunk_info(os.path.join(self._config.log_dir, name))
            if info is not None:
                yield info

    def _load_chunk_info(self, path: str) -> tp.Optional[ChunkedLogPathInfo]:
        """Parse chunk identity from ``path`` if a valid sidecar meta exists for this log."""
        meta_path = self._meta_path_for_chunk_path(path)
        if not os.path.exists(meta_path):
            return None
        return self._load_chunk_info_from_meta(path, meta_path)

    def _load_chunk_info_from_meta(self, path: str, meta_path: str) -> tp.Optional[ChunkedLogPathInfo]:
        """Build :class:`ChunkedLogPathInfo` from meta JSON, or ``None`` if it is not this log's chunk."""
        with open(meta_path, "r", encoding="utf-8") as meta_fp:
            meta = json.load(meta_fp)

        if meta.get("log_name") != self._name:
            return None

        session_id = meta.get("session_id")
        chunk_index = meta.get("chunk_index")
        chunk_stamp = meta.get("chunk_stamp")
        chunk_path = meta.get("chunk_path")
        if (
            not isinstance(session_id, str)
            or not isinstance(chunk_index, int)
            or not isinstance(chunk_stamp, str)
            or chunk_path != path
        ):
            return None

        return ChunkedLogPathInfo(
            path=path,
            meta_path=meta_path,
            session_id=session_id,
            chunk_index=chunk_index,
            chunk_stamp=chunk_stamp,
        )

    def _meta_path_for_chunk_path(self, path: str) -> str:
        """Sidecar path convention: ``chunk.jsonl`` → ``chunk.jsonl.meta.json``."""
        return f"{path}.meta.json"

    def _is_prune_eligible(self, info: ChunkedLogPathInfo) -> bool:
        """Whether ``info`` may be deleted (current session always eligible; old sessions age-gated)."""
        if not self._is_previous_session(info):
            return True

        age_s = time.time() - os.path.getmtime(info.path)
        return age_s >= self._config.previous_session_prune_min_age_s

    def _is_previous_session(self, info: ChunkedLogPathInfo) -> bool:
        """True if ``info`` belongs to a different session than the live writer."""
        return info.session_id != self._session_id


class FsmTransitionCursor:
    """Cursor over an append-only FSM transition history: only "new" items are returned each time."""

    def __init__(self):
        self._last_transition_idx = 0

    def pop_new(self, history: tp.Sequence[tp.Any]) -> tp.List[tp.Any]:
        """Return transitions appended since the last call and advance the internal index."""
        new_items = list(history[self._last_transition_idx:])
        self._last_transition_idx = len(history)
        return new_items


class ControllerSnapshotBuilder:
    """Extract JSON-friendly controller state (FSM node, env time, action, motor, IMU) for log events."""

    def __init__(
        self,
        controller: tp.Any,
    ):
        """``controller`` must expose ``_fsm``, ``env``, and related attributes used by the builders below."""
        self._controller = controller

    def build_snapshot_payload(self) -> tp.Dict[str, tp.Any]:
        """Single dict suitable for an ``fsm_snapshot`` payload (keys match logging consumers)."""
        payload: tp.Dict[str, tp.Any] = {
            "node": self.current_node_name(),
            "env_time": self.env_time(),
            "action": self.latest_action(),
            "motor_state_format": self.motor_state_format(),
            "motor_state": self.motor_state(),
            "imu_state_format": self.imu_state_format(),
            "imu_state": self.imu_state(),
        }
        last_state_time = self.low_level_last_state_time()
        if last_state_time is not None:
            payload["last_state_time"] = last_state_time
        return payload

    def low_level_last_state_time(self) -> tp.Optional[float]:
        """Wall time of the last low-level state callback, if ``env.controller`` exposes ``last_state_time``."""
        env = self._controller.env
        if not hasattr(env, "controller") or env.controller is None:
            return None
        lc = env.controller
        if not hasattr(lc, "last_state_time"):
            return None
        try:
            return float(lc.last_state_time)
        except (TypeError, ValueError):
            return None

    def current_node_name(self) -> str:
        """Name of the FSM node currently active on the controller."""
        return self._controller._fsm._current_node.name

    def env_time(self) -> float:
        """Simulation or environment clock from ``env.get_env_time()``."""
        return float(self._controller.env.get_env_time())

    def latest_action(self) -> tp.Any:
        """Last action vector (or structure), coerced via :meth:`_to_jsonable`."""
        return self._to_jsonable(self._controller.env.action)

    def motor_state_format(self) -> str:
        """Label for :meth:`motor_state` serialization strategy."""
        if self.has_unitree_motor_state():
            return "unitree_low_state"
        return "fallback"

    def motor_state(self) -> tp.Any:
        """Motor telemetry: full Unitree low-level state if present, else minimal q/temp per joint."""
        if self.has_unitree_motor_state():
            motor_states = self._controller.env.controller.state.motor_state
            return [self._serialize_motor_state(i, motor_state) for i, motor_state in enumerate(motor_states)]

        return self._serialize_fallback_motor_state(self._controller.env)

    def has_unitree_motor_state(self) -> bool:
        """True if ``env.controller.state.motor_state`` is present."""
        if not hasattr(self._controller.env, "controller") or self._controller.env.controller is None:
            return False
        return self._controller.env.controller.state.motor_state is not None

    def imu_state_format(self) -> str:
        """Label for :meth:`imu_state` serialization strategy."""
        if self.has_unitree_imu_state():
            return "unitree_low_state"
        return "observation_fallback"

    def imu_state(self) -> tp.Dict[str, tp.Any]:
        """IMU fields from Unitree low-level state if available, else from env observation dict."""
        if self.has_unitree_imu_state():
            imu_state = self._controller.env.controller.state.imu_state
            return {
                "quaternion": [float(item) for item in imu_state.quaternion],
                "gyroscope": [float(item) for item in imu_state.gyroscope],
                "accelerometer": [float(item) for item in imu_state.accelerometer],
                "rpy": [float(item) for item in imu_state.rpy],
                "temperature": int(imu_state.temperature),
            }

        obs = self._controller.env.get_observation()
        return {
            "quaternion": [float(item) for item in obs["imu_quat"]],
            "gyroscope": [float(item) for item in obs["ang_vel"]],
            "gravity_orientation": [float(item) for item in obs["gravity_orientation"]],
        }

    def has_unitree_imu_state(self) -> bool:
        """True if ``env.controller.state.imu_state`` is present."""
        if not hasattr(self._controller.env, "controller") or self._controller.env.controller is None:
            return False
        return self._controller.env.controller.state.imu_state is not None

    def _serialize_motor_state(self, motor_id: int, motor_state: tp.Any) -> tp.Dict[str, tp.Any]:
        """One motor's Unitree low-level fields as plain scalars and lists."""
        return {
            "motor_id": motor_id,
            "mode": int(motor_state.mode),
            "q": float(motor_state.q),
            "dq": float(motor_state.dq),
            "ddq": float(motor_state.ddq),
            "tau_est": float(motor_state.tau_est),
            "temperature": [int(item) for item in motor_state.temperature],
            "vol": float(motor_state.vol),
            "sensor": [int(item) for item in motor_state.sensor],
            "motorstate": int(motor_state.motorstate),
            "reserve": [int(item) for item in motor_state.reserve],
        }

    def _serialize_fallback_motor_state(self, env: tp.Any) -> tp.List[tp.Dict[str, tp.Any]]:
        """Per-joint ``q`` and temperature when Unitree motor_state is unavailable."""
        qj = self._to_jsonable(env.get_current_motor_state())
        temps = self._to_jsonable(env.get_motor_temperatures())

        size = max(len(qj), len(temps))
        result = []
        for motor_id in range(size):
            entry = {"motor_id": motor_id}
            if motor_id < len(qj):
                entry["q"] = float(qj[motor_id])
            if motor_id < len(temps):
                entry["temperature"] = [float(temps[motor_id])]
            result.append(entry)
        return result

    @classmethod
    def _to_jsonable(cls, value: tp.Any) -> tp.Any:
        """Recursively convert numpy arrays, dicts, and sequences; fall back to ``repr`` for other types."""
        if value is None:
            return None
        if hasattr(value, "tolist"):
            return value.tolist()
        if isinstance(value, dict):
            return {str(k): cls._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._to_jsonable(item) for item in value]
        if isinstance(value, (str, int, float, bool)):
            return value
        return repr(value)


class ControllerFsmLogger:
    """Callable hook: logs new FSM transitions every tick and periodic ``fsm_snapshot`` rows to JSONL."""

    def __init__(
        self,
        controller: tp.Any,
        chunked_log: ChunkedJsonlLog,
        snapshot_every_n: int = 1,
    ):
        """``snapshot_every_n`` controls how often (in ``__call__`` invocations) a snapshot is written."""
        self._controller = controller
        self._chunked_log = chunked_log
        self._snapshot_every_n = max(int(snapshot_every_n), 1)
        self._tick = 0
        self._transitions = FsmTransitionCursor()
        self._snapshots = ControllerSnapshotBuilder(controller)

    @property
    def path(self) -> str:
        """Current chunk path of the underlying :class:`ChunkedJsonlLog`."""
        return self._chunked_log.path

    @property
    def log_dir(self) -> str:
        """Log directory of the underlying :class:`ChunkedJsonlLog`."""
        return self._chunked_log.log_dir

    @property
    def session_id(self) -> str:
        """Session id of the underlying :class:`ChunkedJsonlLog`."""
        return self._chunked_log.session_id

    def __call__(self):
        """Log new transitions; on every ``snapshot_every_n``-th tick, write an ``fsm_snapshot`` event."""
        self._log_new_transitions()
        if self._tick % self._snapshot_every_n == 0:
            self._write_event("fsm_snapshot", **self._snapshots.build_snapshot_payload())
        self._tick += 1

    def close(self):
        """Close the chunked log file handle."""
        self._chunked_log.close()

    def _log_new_transitions(self):
        """Emit ``fsm_transition`` events for any entries not yet consumed from FSM history."""
        history = self._controller._fsm.transition_history
        for transition in self._transitions.pop_new(history):
            self._write_event(
                "fsm_transition",
                transition_mono=transition.timestamp,
                source_node=transition.source_node_name,
                target_node=transition.target_node_name,
                cause=transition.cause,
            )

    def _write_event(self, event: str, **payload):
        """Forward a structured row to :meth:`ChunkedJsonlLog.write_event`."""
        self._chunked_log.write_event(event=event, payload=payload)
