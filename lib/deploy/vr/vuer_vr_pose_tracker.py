"""Vuer WebXR adapter implementing ``BaseVRPoseTracker`` for deploy.

Live teleop over the network with no headset SDK: Vuer serves a WebXR page over
HTTPS/WSS; the operator opens the printed URL in the headset browser, enters
Passthrough, and the headset streams head + controller poses back over a
WebSocket.

The Vuer server runs in a **separate process** (not a thread): its asyncio/aiohttp
event loop is CPU-busy and would otherwise contend for the GIL with the realtime
(SCHED_RR) policy control loop, causing multi-hundred-ms stalls ("massive
overtime"). Poses are handed back to the parent through a fixed-size shared-memory
array; ``get_state()`` reads the latest snapshot, non-blocking. The public
interface is unchanged, so the rest of the deploy pipeline is unaffected.

Poses arrive in the WebXR/OpenXR reference frame (right-handed, Y-up, meters) and
are converted to the robot frame by a similarity transform (``_openxr_to_robot``).
Hand tracking is not supported: only 6-DoF motion controllers (Quest/Touch).
"""

from __future__ import annotations

import atexit
import multiprocessing as mp
import os
import socket
import sys
import time
from importlib.util import find_spec
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from lib.deploy.vr.vr_pose_tracker import BaseVRPoseTracker, VRPoseState
from lib.deploy.vr.common import VRControllerState

# OpenXR (WebXR) -> robot frame: right-handed Y-up -> Z-up basis change, applied
# as the similarity transform ``T_ROBOT_OPENXR @ M @ T_OPENXR_ROBOT``.
_T_ROBOT_OPENXR = np.array(
    [[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float32
)
_T_OPENXR_ROBOT = np.array(
    [[0, -1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float32
)


def _webxr_matrix(value: dict) -> Optional[np.ndarray]:
    """Build the raw WebXR 4x4 pose from a payload ({matrix:[16]} or {position,rotation}).

    Returns None for the zero/degenerate poses streamed before tracking locks.
    """
    matrix = value.get("matrix")
    if matrix is not None and len(matrix) == 16:
        m = np.asarray(matrix, dtype=np.float32).reshape(4, 4).T  # three.js col-major
        if not np.isfinite(m).all() or abs(float(np.linalg.det(m[:3, :3]))) < 1e-6:
            return None
        return m

    pos, quat = value.get("position"), value.get("rotation")
    if pos is None or quat is None:
        return None
    pos = np.asarray(pos, dtype=np.float32).reshape(3)
    quat = np.asarray(quat, dtype=np.float32).reshape(4)
    if not (np.isfinite(pos).all() and np.isfinite(quat).all()) or np.linalg.norm(quat) < 1e-6:
        return None
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = R.from_quat(quat).as_matrix()
    m[:3, 3] = pos
    return m


def _openxr_to_robot(m_xr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Convert a WebXR/OpenXR 4x4 pose to the robot frame (None passes through)."""
    if m_xr is None:
        return None
    return _T_ROBOT_OPENXR @ m_xr @ _T_OPENXR_ROBOT


def _controller_state(pose_data, state, ts: float) -> Optional[VRControllerState]:
    """Parse a CONTROLLER_MOVE side into a VRControllerState (None if untracked).

    When tracked, the pose (``left``/``right``) is a 16-float column-major 4x4
    matrix (same as HEAD_MOVE); untracked it is an empty msgpack ExtType, which
    _webxr_matrix rejects. Buttons arrive in a separate named-field dict
    (``leftState``/``rightState``).
    """
    if isinstance(pose_data, (list, tuple)) and len(pose_data) == 16:
        pose = _openxr_to_robot(_webxr_matrix({"matrix": pose_data}))
    elif isinstance(pose_data, dict):
        pose = _openxr_to_robot(_webxr_matrix(pose_data))
    else:
        pose = None
    if pose is None:
        return None

    state = state if isinstance(state, dict) else {}

    def flag(*names) -> bool:
        return any(bool(state.get(n)) for n in names)

    def value(*names) -> float:
        for n in names:
            if n in state:
                v = state[n]
                return 1.0 if v is True else 0.0 if v is False else float(v)
        return 0.0

    return VRControllerState(
        pose=pose,
        btn_a=flag("aButton", "a"),
        btn_b=flag("bButton", "b"),
        trigger_front=value("triggerValue", "trigger"),
        trigger_side=value("squeezeValue", "squeeze"),
        btn_j=flag("thumbstick", "thumbstickClick"),
        axe_j_x=value("thumbstickX"),
        axe_j_y=value("thumbstickY"),
        timestamp=ts,
    )


class _QuietVuer:
    """stdout wrapper that drops Vuer's startup banner and internal chatter.

    Vuer prints an unconditional ``Vuer Server / Local: / Network:`` banner (with
    the wrong localhost/container URLs) since we print our own URL, plus some
    internal worker noise. Connect/disconnect messages are kept (useful status);
    everything else passes through.
    """

    _DROP = (
        "Vuer Server", "Local:", "Network:", "Workspace:", "vuer.ai/workspace",
        "default socket worker is up", "Uplink task running",
    )

    def __init__(self, wrapped):
        self._wrapped = wrapped

    def write(self, s: str) -> int:
        if any(marker in s for marker in self._DROP):
            return len(s)
        return self._wrapped.write(s)

    def __getattr__(self, name):
        return getattr(self._wrapped, name)


def _detect_lan_ip() -> str:
    """The IP to advertise to the headset.

    Honors ``YACT_VR_HOST`` first: inside a container (or behind NAT / a tunnel)
    the auto-detected address is the container/VM IP, which the headset cannot
    reach — set ``YACT_VR_HOST`` to the host's real LAN IP in that case. Otherwise
    best-effort primary LAN IPv4 (no packets sent; falls back to 127.0.0.1).
    """
    override = os.environ.get("YACT_VR_HOST")
    if override:
        return override

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def ensure_self_signed_cert(cert_dir: str, host_ip: str) -> Tuple[str, str]:
    """Generate (once) a self-signed cert/key for HTTPS/WSS; return their paths.

    WebXR requires a secure context, so the Vuer page must be served over HTTPS
    and the headset connects over WSS. We autogenerate a long-lived self-signed
    cert covering the LAN IP + localhost so no external tooling (ngrok) is needed;
    the operator accepts the browser warning once.
    """
    os.makedirs(cert_dir, exist_ok=True)
    cert_path = os.path.join(cert_dir, "vuer_cert.pem")
    key_path = os.path.join(cert_dir, "vuer_key.pem")
    if os.path.exists(cert_path) and os.path.exists(key_path):
        return cert_path, key_path

    import datetime
    import ipaddress

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "yact-vuer")])

    sans = [x509.DNSName("localhost")]
    for candidate in {host_ip, "127.0.0.1", "0.0.0.0"}:
        try:
            sans.append(x509.IPAddress(ipaddress.ip_address(candidate)))
        except ValueError:
            pass

    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(x509.SubjectAlternativeName(sans), critical=False)
        .sign(key, hashes.SHA256())
    )

    with open(key_path, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    return cert_path, key_path


# --- shared-memory pose channel (child server -> parent) -------------------
#
# Flat float64 layout, guarded by the Array's lock. Each entity has a `valid`
# flag (0/1), a timestamp, a 16-float 4x4 pose, and (controllers) 7 button/axis
# values. Values persist once written (latest-value semantics), matching the
# old in-thread caching.
_HEAD_OFF = 0                     # valid, ts, mat[16]                    -> 18
_LEFT_OFF = 18                    # valid, ts, mat[16], btnA,btnB,trigF,  -> 25
_RIGHT_OFF = 43                   # trigS, btnJ, axX, axY                 -> 25
_SHARED_LEN = 68


class _SharedVRState:
    """Reader/writer over the shared float64 array (works in either process)."""

    def __init__(self, arr):
        self._arr = arr  # multiprocessing.Array('d', _SHARED_LEN)

    def _buf(self) -> np.ndarray:
        # O(1) view over the shared block; built per-call so the object stays
        # picklable across the process boundary (no cached ndarray attribute).
        return np.frombuffer(self._arr.get_obj(), dtype=np.float64)

    # writer side (child) -------------------------------------------------
    def set_head(self, pose: np.ndarray, ts: float) -> None:
        with self._arr.get_lock():
            buf = self._buf()
            buf[_HEAD_OFF] = 1.0
            buf[_HEAD_OFF + 1] = ts
            buf[_HEAD_OFF + 2:_HEAD_OFF + 18] = np.asarray(pose, dtype=np.float64).ravel()

    def set_controller(self, off: int, s: VRControllerState) -> None:
        with self._arr.get_lock():
            buf = self._buf()
            buf[off] = 1.0
            buf[off + 1] = s.timestamp
            buf[off + 2:off + 18] = np.asarray(s.pose, dtype=np.float64).ravel()
            buf[off + 18] = 1.0 if s.btn_a else 0.0
            buf[off + 19] = 1.0 if s.btn_b else 0.0
            buf[off + 20] = float(s.trigger_front)
            buf[off + 21] = float(s.trigger_side)
            buf[off + 22] = 1.0 if s.btn_j else 0.0
            buf[off + 23] = float(s.axe_j_x)
            buf[off + 24] = float(s.axe_j_y)

    # reader side (parent) ------------------------------------------------
    def snapshot(self):
        """Return ((head_pose, head_ts), left_state, right_state), consistently."""
        with self._arr.get_lock():
            buf = self._buf()
            head = self._read_head(buf)
            left = self._read_controller(buf, _LEFT_OFF)
            right = self._read_controller(buf, _RIGHT_OFF)
        return head, left, right

    @staticmethod
    def _read_head(buf):
        if buf[_HEAD_OFF] < 0.5:
            return None, None
        pose = buf[_HEAD_OFF + 2:_HEAD_OFF + 18].reshape(4, 4).astype(np.float32)
        return pose, float(buf[_HEAD_OFF + 1])

    @staticmethod
    def _read_controller(buf, off) -> Optional[VRControllerState]:
        if buf[off] < 0.5:
            return None
        pose = buf[off + 2:off + 18].reshape(4, 4).astype(np.float32)
        return VRControllerState(
            pose=pose,
            btn_a=buf[off + 18] > 0.5,
            btn_b=buf[off + 19] > 0.5,
            trigger_front=float(buf[off + 20]),
            trigger_side=float(buf[off + 21]),
            btn_j=buf[off + 22] > 0.5,
            axe_j_x=float(buf[off + 23]),
            axe_j_y=float(buf[off + 24]),
            timestamp=float(buf[off + 1]),
        )


# --- child-process Vuer server ---------------------------------------------


class _VuerServer:
    """Runs inside the child process: parses WebXR events into the shared state."""

    def __init__(self, shared: _SharedVRState, debug: bool):
        self._shared = shared
        self._debug = debug
        self._logged: set = set()

    def register(self, app, DefaultScene, MotionControllers, Head) -> None:
        # start=False only registers the session handler; app.run() starts the
        # blocking server. Head streams the immersive headset pose via HEAD_MOVE
        # (CAMERA_MOVE is only the desktop orbit camera); MotionControllers needs
        # left/right enabled explicitly or the client never publishes poses.
        @app.spawn(start=False)
        async def _main(session):
            self._dlog("session connected")
            session.set @ DefaultScene(
                Head(key="head", stream=True, fps=30),
                MotionControllers(key="controllers", stream=True, fps=30, left=True, right=True),
            )
            await session.forever()

        @app.add_handler("HEAD_MOVE")
        async def _on_head(event, session):
            self._set_head(event.value or {})

        @app.add_handler("CAMERA_MOVE")  # desktop fallback; rarely fires in XR
        async def _on_camera(event, session):
            self._set_head(event.value or {})

        @app.add_handler("CONTROLLER_MOVE")
        async def _on_controller(event, session):
            value = event.value or {}
            ts = time.time()
            try:
                left = _controller_state(value.get("left"), value.get("leftState"), ts)
                right = _controller_state(value.get("right"), value.get("rightState"), ts)
            except Exception as exc:
                self._dlog_once("ctrl-err", f"CONTROLLER_MOVE parse error: {exc}")
                return
            if left is not None:
                self._shared.set_controller(_LEFT_OFF, left)
            if right is not None:
                self._shared.set_controller(_RIGHT_OFF, right)
            if left is not None or right is not None:
                self._dlog_once("ctrl", "controller poses valid")

    def _set_head(self, value: dict) -> None:
        try:
            pose = _openxr_to_robot(_webxr_matrix(value))
        except Exception as exc:
            self._dlog_once("head-err", f"head parse error: {exc}")
            return
        if pose is not None:
            self._shared.set_head(pose, time.time())
            self._dlog_once("head", "head pose valid")

    def _dlog(self, msg: str) -> None:
        if self._debug:
            print(f"[vuer] {msg}", flush=True)

    def _dlog_once(self, tag: str, msg: str) -> None:
        if self._debug and tag not in self._logged:
            self._logged.add(tag)
            print(f"[vuer] {msg}", flush=True)


def _run_vuer_server(shared_arr, port: int, cert_path: str, key_path: str, debug: bool) -> None:
    """Child-process entrypoint: build the Vuer app and serve forever."""
    import asyncio

    from vuer import Vuer
    from vuer.schemas import DefaultScene, Head, MotionControllers

    asyncio.set_event_loop(asyncio.new_event_loop())
    if not isinstance(sys.stdout, _QuietVuer):
        sys.stdout = _QuietVuer(sys.stdout)  # drop Vuer's banner / socket chatter

    server = _VuerServer(_SharedVRState(shared_arr), debug)
    # The `host` kwarg name has varied across vuer versions; fall back to the env
    # var + default bind if the constructor rejects it. Bind all interfaces so a
    # headset on the LAN can reach it (docker uses --network host).
    try:
        app = Vuer(host="0.0.0.0", port=port, cert=cert_path, key=key_path)
    except TypeError:
        os.environ.setdefault("VUER_HOST", "0.0.0.0")
        app = Vuer(port=port, cert=cert_path, key=key_path)
    server.register(app, DefaultScene, MotionControllers, Head)
    try:
        app.run()
    except Exception as exc:  # pragma: no cover - surfaced to the operator
        print(f"[vuer] server stopped: {exc}", flush=True)


class VuerVRPoseTracker(BaseVRPoseTracker):
    """Live VR via Vuer WebXR (headset browser + 6-DoF motion controllers).

    The WebXR server runs in a child process; poses come back via shared memory.
    """

    def __init__(self, *, port: int = 8012, cert_dir: Optional[str] = None):
        if find_spec("vuer") is None:  # fail fast in the parent
            raise SystemExit(
                "vuer is not installed; --vr needs it. Install with `pip install vuer` "
                "or rebuild the g1-sim image."
            )

        self._port = port
        self._host_ip = _detect_lan_ip()
        debug = os.environ.get("YACT_VR_DEBUG", "").lower() in ("1", "true")

        if cert_dir is None:
            from lib.vendor.env import LOCAL_ROOT
            cert_dir = os.path.join(LOCAL_ROOT, "tmp", "vuer_certs")
        cert_path, key_path = ensure_self_signed_cert(cert_dir, self._host_ip)

        # Separate process so the Vuer event loop's GIL never blocks the realtime
        # policy loop. `spawn` (not fork) gives a clean interpreter: no inherited
        # locks/threads and no dependence on being constructed before CUDA init.
        # The child re-imports the deploy module (a few seconds), but that overlaps
        # with the operator putting the headset on (see _wait_for_vr_connection).
        ctx = mp.get_context("spawn")
        self._arr = ctx.Array("d", _SHARED_LEN)
        self._shared = _SharedVRState(self._arr)
        self._proc = ctx.Process(
            target=_run_vuer_server,
            args=(self._arr, port, cert_path, key_path, debug),
            name="vuer-server",
            daemon=True,
        )
        self._proc.start()
        atexit.register(self.close)
        self._print_url()

    def close(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is not None and proc.is_alive():
            proc.terminate()
            proc.join(timeout=2)

    # --- URL ---------------------------------------------------------------

    def _print_url(self) -> None:
        from lib.vendor.env import print_path_lines

        print_path_lines("  VR (open on headset): ", self.url)
        print(
            "    self-signed cert — accept the browser security warning once, "
            "then click 'Passthrough'.",
            flush=True,
        )

    @property
    def url(self) -> str:
        # Pin the WebSocket target explicitly: the served client otherwise falls
        # back to ws://localhost:8012, which on the headset points at the headset
        # itself, so the connection never reaches this host and no poses flow.
        return f"https://{self._host_ip}:{self._port}?ws=wss://{self._host_ip}:{self._port}"

    # --- BaseVRPoseTracker -------------------------------------------------

    def get_state(self) -> VRPoseState:
        (head_pose, head_ts), left, right = self._shared.snapshot()
        return VRPoseState(
            head_pose=head_pose,
            left_wrist_pose=left.pose if left is not None else None,
            right_wrist_pose=right.pose if right is not None else None,
            head_timestamp=head_ts,
            left_wrist_timestamp=left.timestamp if left is not None else None,
            right_wrist_timestamp=right.timestamp if right is not None else None,
        )

    def is_connected(self) -> bool:
        return self.get_state().is_full()

    def get_controller_states(
        self,
    ) -> Tuple[Optional[VRControllerState], Optional[VRControllerState]]:
        _, left, right = self._shared.snapshot()
        return left, right
