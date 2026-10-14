"""Browser live preview for MuJoCo offscreen frames (MJPEG over HTTP)."""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

import cv2
import numpy as np

from lib.deploy.control.live_view_input import BrowserInputBridge
from lib.deploy.debug.logs import _display_fsm_node
from lib.vendor.env import print_path_lines

logger = logging.getLogger(__name__)

_WASD_HINTS_ACTIVE_WEB = [
    "Click the page, then use keyboard here",
    "W / S — forward / back",
    "A / D — strafe left / right",
    "Q / E — rotate left / right",
    "Space — stop movement",
    "x — emergency damp",
    "0–9 — jump to FSM node",
    "r — reset sim pose",
]

_WASD_HINTS_ACTIVE_STDIN = [
    "W / S — forward / back",
    "A / D — strafe left / right",
    "Q / E — rotate left / right",
    "Space — stop movement",
    "x — emergency damp",
    "0–9 — jump to FSM node",
    "r — reset sim pose",
    "Ctrl+C in terminal — quit",
    "Type keys in the terminal running ./yact sim",
]

_WASD_HINTS_UNAVAILABLE: List[str] = []

_REPLAY_HINTS_ACTIVE_WEB = [
    "Click the page for FSM / reset keys",
    "x — emergency damp",
    "0–9 — jump to FSM node",
    "r — reset sim pose",
]

_REPLAY_HINTS_ACTIVE_STDIN = [
    "x — emergency damp",
    "0–9 — jump to FSM node",
    "r — reset sim pose",
    "Ctrl+C in terminal — stop replay",
    "Type keys in the terminal running ./yact sim",
]

_REPLAY_HINTS_UNAVAILABLE = [
    "WASD / Q / E / Space — motion is driven by the recording",
]

_WASD_MODE_TOOLTIP = (
    "Keyboard teleop: velocity commands steer the policy."
)

_WASD_MODE_TOOLTIP_WEB = (
    "Click the page, then use WASD and other keys to steer the policy."
)

_VR_REPLAY_MODE_TOOLTIP = (
    "VR replay plays a recorded demo: headset and controller poses from a JSON "
    "trajectory are fed to the policy step-by-step. The robot follows that motion "
    "automatically — keyboard driving is disabled."
)

_REPLAY_PROGRESS_TOOLTIP = (
    "Current frame index in the recorded trajectory (updates as the demo plays)."
)

_INDEX_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>YaCT sim live view</title>
  <style>
    * { box-sizing: border-box; }
    html, body { margin: 0; height: 100%; background: #0d0d0d; color: #e8e8e8;
      font-family: ui-sans-serif, system-ui, sans-serif; overflow: hidden; }
    #stream { display: block; width: 100vw; height: 100vh; object-fit: contain;
      background: #000; }
    .panel { position: fixed; background: rgba(12, 12, 16, 0.82);
      border: 1px solid rgba(255,255,255,0.12); border-radius: 8px;
      backdrop-filter: blur(6px); padding: 10px 12px; font-size: 13px;
      line-height: 1.45; pointer-events: none; }
    #state-panel { top: 12px; left: 12px; min-width: 200px; }
    #state-panel h2 { margin: 0 0 8px; font-size: 11px; font-weight: 600;
      letter-spacing: 0.06em; text-transform: uppercase; color: #9aa0a6; }
    #state-panel dl { margin: 0; display: grid; grid-template-columns: auto 1fr;
      gap: 2px 10px; }
    #state-panel dt { color: #9aa0a6; }
    #state-panel dd { margin: 0; font-variant-numeric: tabular-nums; }
    #state-panel .has-tooltip { pointer-events: auto; cursor: help;
      border-bottom: 1px dotted rgba(255,255,255,0.35); width: fit-content; }
    #hints-panel { bottom: 12px; left: 12px; max-width: 360px; pointer-events: auto; }
    #hints-panel h2 { margin: 0 0 6px; font-size: 11px; font-weight: 600;
      letter-spacing: 0.06em; text-transform: uppercase; color: #9aa0a6; }
    .hints-section { margin: 0 0 6px; font-size: 10px; font-weight: 600;
      letter-spacing: 0.05em; text-transform: uppercase; color: #9aa0a6; }
    #hints-panel ul { margin: 0 0 8px; padding-left: 18px; color: #c8cdd2; }
    #hints-panel ul:last-child { margin-bottom: 0; }
    #hints-panel li { margin: 2px 0; }
    #hints-unavailable { color: #7a828a; }
    #hints-unavailable li { font-style: italic; }
    #start-overlay { position: fixed; inset: 0; display: flex; align-items: center;
      justify-content: center; background: rgba(0,0,0,0.45); pointer-events: auto; }
    #start-overlay.hidden { display: none; }
    #start-card { text-align: center; background: rgba(12, 12, 16, 0.92);
      border: 1px solid rgba(255,255,255,0.15); border-radius: 12px;
      padding: 28px 36px; max-width: 420px; }
    #start-card h1 { margin: 0 0 8px; font-size: 20px; font-weight: 600; }
    #start-card p { margin: 0 0 18px; color: #aeb4ba; font-size: 14px; }
    #start-btn { pointer-events: auto; cursor: pointer; border: none;
      border-radius: 8px; padding: 12px 28px; font-size: 15px; font-weight: 600;
      color: #fff; background: #2b6cb0; }
    #start-btn:hover { background: #3182ce; }
    #start-btn:active { background: #2c5282; }
    .status-waiting { color: #f6ad55; }
    .status-running { color: #68d391; }
    .status-finished { color: #9ae6b4; }
    #fsm-panel { top: 12px; left: 50%; transform: translateX(-50%);
      display: flex; flex-direction: column; align-items: center; max-width: 92vw; }
    #fsm-panel .fsm-legend { font-size: 10px; color: #9aa0a6; margin-bottom: 6px;
      letter-spacing: 0.03em; }
    #fsm-row { display: flex; align-items: center; gap: 4px; flex-wrap: wrap;
      justify-content: center; }
    .fsm-node { padding: 3px 8px; border-radius: 6px; font-size: 12px; white-space: nowrap;
      border: 1px solid rgba(255,255,255,0.15); color: #c8cdd2; background: rgba(255,255,255,0.04); }
    .fsm-node.paused { color: #7a828a; border-style: dashed; }
    .fsm-node.current { color: #0d0d0d; background: #68d391; border-color: #68d391; font-weight: 600; }
    .fsm-seg { display: flex; flex-direction: column; align-items: center; line-height: 1.1; }
    .fsm-key { font-size: 9px; color: #7a828a; font-weight: 600; white-space: nowrap; }
    .fsm-arrow { color: #6a6f75; font-size: 13px; }
    .fsm-seg.active .fsm-key { color: #68d391; }
    .fsm-seg.active .fsm-arrow { color: #68d391; }
    #paused-banner { position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
      background: rgba(12,12,16,0.9); border: 1px solid rgba(246,173,85,0.5);
      border-radius: 10px; padding: 16px 22px; text-align: center; max-width: 460px;
      pointer-events: none; }
    #paused-banner[hidden] { display: none; }
    #pb-title { font-size: 15px; font-weight: 600; color: #f6ad55; }
    #pb-sub { font-size: 13px; color: #aeb4ba; margin-top: 6px; line-height: 1.4; }
    body.focus-visible { outline: 1px dashed rgba(255,255,255,0.25); outline-offset: -4px; }
  </style>
</head>
<body tabindex="0">
  <img id="stream" src="/stream" alt="sim stream"/>
  <div id="state-panel" class="panel">
    <h2>State</h2>
    <dl>
      <dt>Status</dt><dd id="st-status">—</dd>
      <dt>Mode</dt><dd id="st-mode" class="has-tooltip">—</dd>
      <dt>Sim time</dt><dd id="st-time">—</dd>
      <dt>FSM</dt><dd id="st-fsm">—</dd>
      <dt>Velocity</dt><dd id="st-vel">—</dd>
      <dt>Replay</dt><dd id="st-replay" class="has-tooltip">—</dd>
    </dl>
  </div>
  <div id="fsm-panel" class="panel">
    <div class="fsm-legend">FSM — advance: number key / VR-B · emergency damp: x / VR-A</div>
    <div id="fsm-row"></div>
  </div>
  <div id="paused-banner" hidden>
    <div id="pb-title">Sim paused</div>
    <div id="pb-sub"></div>
  </div>
  <div id="hints-panel" class="panel">
    <h2>Controls</h2>
    <div id="hints-active-section" class="hints-section">In this mode</div>
    <ul id="hints-active"></ul>
    <div id="hints-unavailable-section" class="hints-section" hidden>Not available</div>
    <ul id="hints-unavailable" hidden></ul>
  </div>
  <div id="start-overlay">
    <div id="start-card">
      <h1>YaCT sim</h1>
      <p id="start-message">Click Start when the stream looks ready.</p>
      <button id="start-btn" type="button">Start</button>
    </div>
  </div>
  <script>
    const overlay = document.getElementById("start-overlay");
    const startBtn = document.getElementById("start-btn");
    const hintsActive = document.getElementById("hints-active");
    const hintsUnavailable = document.getElementById("hints-unavailable");
    const hintsUnavailableSection = document.getElementById("hints-unavailable-section");

    function setText(id, text) {
      document.getElementById(id).textContent = text;
    }

    function setTooltip(id, text) {
      const el = document.getElementById(id);
      if (text) {
        el.title = text;
        el.classList.add("has-tooltip");
      } else {
        el.removeAttribute("title");
        el.classList.remove("has-tooltip");
      }
    }

    function renderHintList(listEl, lines) {
      listEl.innerHTML = "";
      for (const line of lines || []) {
        const li = document.createElement("li");
        li.textContent = line;
        listEl.appendChild(li);
      }
    }

    function renderHints(data) {
      renderHintList(hintsActive, data.hints_active);
      const unavailable = data.hints_unavailable || [];
      const showUnavailable = unavailable.length > 0;
      hintsUnavailableSection.hidden = !showUnavailable;
      hintsUnavailable.hidden = !showUnavailable;
      if (showUnavailable) {
        renderHintList(hintsUnavailable, unavailable);
      }
    }

    function formatVelocity(v) {
      if (!v || v.length < 3) return "—";
      return `[${v[0].toFixed(2)}, ${v[1].toFixed(2)}, ${v[2].toFixed(2)}]`;
    }

    let fsmBuilt = false;
    function fsmKeys(edges, from, to) {
      const e = (edges || []).find((e) => e.from === from && e.to === to);
      return e ? e.keys.join(" / ") : "";
    }
    function renderFsm(data) {
      const g = data.fsm_graph;
      if (!g || !g.nodes) return;
      const row = document.getElementById("fsm-row");
      if (!fsmBuilt) {
        row.innerHTML = "";
        g.nodes.forEach((n, i) => {
          if (i > 0) {
            const prev = g.nodes[i - 1].name;
            const seg = document.createElement("span");
            seg.className = "fsm-seg";
            seg.dataset.from = prev;  // arrow leaving `prev`
            const arrow = (i === g.nodes.length - 1) ? "⇄" : "→";
            seg.innerHTML =
              `<span class="fsm-key">${fsmKeys(g.edges, prev, n.name)}</span>` +
              `<span class="fsm-arrow">${arrow}</span>`;
            row.appendChild(seg);
          }
          const chip = document.createElement("span");
          chip.className = "fsm-node" + (n.paused ? " paused" : "");
          chip.dataset.node = n.name;
          chip.textContent = n.name;
          row.appendChild(chip);
        });
        fsmBuilt = true;
      }
      const cur = data.fsm_node;
      row.querySelectorAll(".fsm-node").forEach((el) => {
        el.classList.toggle("current", el.dataset.node === cur);
      });
      row.querySelectorAll(".fsm-seg").forEach((el) => {
        el.classList.toggle("active", el.dataset.from === cur);
      });
      const curNode = g.nodes.find((n) => n.name === cur);
      const banner = document.getElementById("paused-banner");
      if (curNode && curNode.paused && data.status !== "waiting") {
        banner.hidden = false;
        document.getElementById("pb-title").textContent = "⏸ " + cur + " — sim paused";
        document.getElementById("pb-sub").textContent =
          "No physics or rendering in this state (the view is static). " +
          "Press 1 (or VR-B) to advance; x / VR-A for emergency damp.";
      } else {
        banner.hidden = true;
      }
    }

    async function pollState() {
      try {
        const res = await fetch("/state");
        if (!res.ok) return;
        const data = await res.json();
        renderHints(data);
        renderFsm(data);

        const status = data.status || "—";
        const statusEl = document.getElementById("st-status");
        statusEl.textContent = status;
        statusEl.className = "status-" + status;

        setText("st-mode", data.mode_label || data.mode || "—");
        setTooltip("st-mode", data.mode_tooltip || "");
        setText("st-time", data.sim_time != null ? data.sim_time.toFixed(1) + " s" : "—");
        setText("st-fsm", data.fsm_node || "—");
        setText("st-vel", formatVelocity(data.velocity));

        if (data.replay_total > 0) {
          setText("st-replay", `${data.replay_frame} / ${data.replay_total}`);
          setTooltip("st-replay", data.replay_tooltip || "");
        } else {
          setText("st-replay", "—");
          setTooltip("st-replay", "");
        }

        if (status === "waiting") {
          overlay.classList.remove("hidden");
        } else {
          overlay.classList.add("hidden");
          startBtn.disabled = false;
          startBtn.textContent = "Start";
        }

        if (data.start_message) {
          document.getElementById("start-message").textContent = data.start_message;
        }
      } catch (e) {
        /* server may still be booting */
      }
    }

    startBtn.addEventListener("click", async () => {
      startBtn.disabled = true;
      startBtn.textContent = "Starting…";
      try {
        const res = await fetch("/start", { method: "POST" });
        if (res.ok) {
          overlay.classList.add("hidden");
          startBtn.disabled = false;
          startBtn.textContent = "Start";
          document.body.focus();
        } else {
          startBtn.disabled = false;
          startBtn.textContent = "Start";
        }
      } catch (e) {
        startBtn.disabled = false;
        startBtn.textContent = "Start";
      }
    });

    const CONTROL_KEYS = new Set([
      "w", "a", "s", "d", "q", "e", " ", "x", "r",
      "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    ]);

    function normalizeKey(ev) {
      if (ev.key === " " || ev.code === "Space") return " ";
      if (ev.key.length === 1) return ev.key.toLowerCase();
      return ev.key;
    }

    function isControlKey(key) {
      return CONTROL_KEYS.has(key);
    }

    async function sendKey(key, action) {
      try {
        await fetch("/input", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ key, action }),
        });
      } catch (e) {
        /* server may still be booting */
      }
    }

    document.body.addEventListener("pointerdown", () => {
      if (overlay.classList.contains("hidden")) {
        document.body.focus();
      }
    });

    window.addEventListener("keydown", (ev) => {
      if (!overlay.classList.contains("hidden")) return;
      if (ev.repeat) return;
      const key = normalizeKey(ev);
      if (!isControlKey(key)) return;
      ev.preventDefault();
      sendKey(key, "keydown");
    });

    window.addEventListener("keyup", (ev) => {
      if (!overlay.classList.contains("hidden")) return;
      const key = normalizeKey(ev);
      if (!isControlKey(key)) return;
      ev.preventDefault();
      sendKey(key, "keyup");
    });

    function clearHeldKeys() {
      if (overlay.classList.contains("hidden")) {
        sendKey("", "clear");
      }
    }

    window.addEventListener("blur", clearHeldKeys);
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) clearHeldKeys();
    });

    pollState();
    setInterval(pollState, 250);
  </script>
</body>
</html>
"""


@dataclass
class ViewConfig:
    mode: str = "wasd"
    trajectory: Optional[str] = None
    hints_active: List[str] = field(default_factory=list)
    hints_unavailable: List[str] = field(default_factory=list)
    mode_tooltip: str = ""
    replay_tooltip: str = ""
    start_message: str = "Click Start when the stream looks ready."
    mode_label: str = "WASD teleop"


class LiveViewServer:
    """Serve the latest rendered frame as multipart MJPEG on HTTP."""

    def __init__(self, port: int = 8766, bind: str = "0.0.0.0"):
        self._port = port
        self._bind = bind
        self._jpeg_lock = threading.Lock()
        self._jpeg: Optional[bytes] = None
        self._last_encode_mono = 0.0
        self._min_encode_interval_s = 1.0 / 25.0
        self._httpd: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._state_lock = threading.Lock()
        self._config = ViewConfig()
        self._runtime_state: Dict[str, Any] = {"status": "waiting"}
        self._start_event = threading.Event()
        self._input_handler: Optional[Callable[[str, str], None]] = None

    def set_input_handler(self, handler: Optional[Callable[[str, str], None]]) -> None:
        self._input_handler = handler

    @property
    def port(self) -> int:
        return self._port

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._port}/"

    def configure(
        self,
        *,
        mode: str,
        trajectory: Optional[str] = None,
        web_keyboard: bool = False,
    ) -> None:
        if mode == "vr_replay":
            hints_active = list(
                _REPLAY_HINTS_ACTIVE_WEB if web_keyboard else _REPLAY_HINTS_ACTIVE_STDIN,
            )
            hints_unavailable = list(_REPLAY_HINTS_UNAVAILABLE)
            mode_label = "VR replay"
            mode_tooltip = _VR_REPLAY_MODE_TOOLTIP
            replay_tooltip = _REPLAY_PROGRESS_TOOLTIP
            if web_keyboard:
                start_message = "Click Start, then click the page for FSM keys."
            else:
                start_message = "Click Start to begin the trajectory replay."
            if trajectory:
                mode_label = f"VR replay ({trajectory})"
        elif mode == "vr":
            hints_active = list(
                _REPLAY_HINTS_ACTIVE_WEB if web_keyboard else _REPLAY_HINTS_ACTIVE_STDIN,
            )
            hints_unavailable = list(_REPLAY_HINTS_UNAVAILABLE)
            mode_label = "Vuer WebXR VR"
            mode_tooltip = (
                "Live VR teleop: headset and controller poses from Vuer WebXR "
                "drive the policy. Use VR buttons or keyboard for FSM."
            )
            replay_tooltip = ""
            if web_keyboard:
                start_message = "Click Start, then click the page for FSM keys."
            else:
                start_message = "Click Start when the headset is connected."
        else:
            hints_active = list(
                _WASD_HINTS_ACTIVE_WEB if web_keyboard else _WASD_HINTS_ACTIVE_STDIN,
            )
            hints_unavailable = list(_WASD_HINTS_UNAVAILABLE)
            mode_label = "WASD teleop"
            mode_tooltip = _WASD_MODE_TOOLTIP_WEB if web_keyboard else _WASD_MODE_TOOLTIP
            replay_tooltip = ""
            if web_keyboard:
                start_message = "Click Start, then click the page and use WASD."
            else:
                start_message = "Click Start, then use WASD in the terminal."

        config = ViewConfig(
            mode=mode,
            trajectory=trajectory,
            hints_active=hints_active,
            hints_unavailable=hints_unavailable,
            mode_tooltip=mode_tooltip,
            replay_tooltip=replay_tooltip,
            start_message=start_message,
            mode_label=mode_label,
        )
        with self._state_lock:
            self._config = config
            self._runtime_state = {
                "status": "waiting",
                "sim_time": 0.0,
                "fsm_node": "—",
                "velocity": [0.0, 0.0, 0.0],
                "replay_frame": 0,
                "replay_total": 0,
            }
        self._start_event.clear()

    def request_start(self) -> None:
        self._start_event.set()
        with self._state_lock:
            if self._runtime_state.get("status") == "waiting":
                self._runtime_state["status"] = "running"

    def wait_for_start(self, timeout: float) -> bool:
        return self._start_event.wait(timeout=timeout)

    def is_started(self) -> bool:
        return self._start_event.is_set()

    def set_state(self, **fields: Any) -> None:
        with self._state_lock:
            self._runtime_state.update(fields)

    def get_state_json(self) -> bytes:
        with self._state_lock:
            payload = {
                "mode": self._config.mode,
                "mode_label": self._config.mode_label,
                "mode_tooltip": self._config.mode_tooltip,
                "replay_tooltip": self._config.replay_tooltip,
                "hints_active": self._config.hints_active,
                "hints_unavailable": self._config.hints_unavailable,
                "start_message": self._config.start_message,
                **self._runtime_state,
            }
        return json.dumps(payload).encode("utf-8")

    def start(self) -> None:
        if self._thread is not None:
            return

        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                logger.debug(fmt, *args)

            def do_GET(self):
                path = urlparse(self.path).path
                if path in ("/", "/index.html"):
                    body = _INDEX_HTML.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if path == "/state":
                    body = server.get_state_json()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if path == "/stream":
                    self.send_response(200)
                    self.send_header(
                        "Content-Type",
                        "multipart/x-mixed-replace; boundary=frame",
                    )
                    self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                    self.send_header("Pragma", "no-cache")
                    self.end_headers()
                    while True:
                        frame = server.get_jpeg()
                        if frame is None:
                            time.sleep(0.05)
                            continue
                        try:
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(frame)}\r\n\r\n".encode())
                            self.wfile.write(frame)
                            self.wfile.write(b"\r\n")
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            break
                        time.sleep(0.04)
                    return

                self.send_error(404)

            def do_POST(self):
                path = urlparse(self.path).path
                if path == "/start":
                    server.request_start()
                    self.send_response(204)
                    self.end_headers()
                    return

                if path == "/input":
                    length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(length) if length else b""
                    try:
                        payload = json.loads(body.decode("utf-8"))
                        key = str(payload.get("key", ""))
                        action = str(payload.get("action", ""))
                        handler = server._input_handler
                        if handler is not None and action:
                            if action == "clear" or key:
                                handler(key, action)
                        self.send_response(204)
                        self.end_headers()
                    except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError):
                        self.send_error(400)
                    return

                self.send_error(404)

        try:
            self._httpd = ThreadingHTTPServer((self._bind, self._port), Handler)
        except OSError as exc:
            raise RuntimeError(
                f"Live view port {self._port} unavailable ({exc}). "
                f"Try --view-port with a different port."
            ) from exc
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="LiveViewServer",
            daemon=True,
        )
        self._thread.start()
        logger.info("Live view at %s", self.url)

    def stop(self) -> None:
        if self._httpd is None:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        self._httpd = None
        self._thread = None

    def update_frame(self, rgb: np.ndarray) -> None:
        if rgb is None:
            return
        now = time.monotonic()
        if now - self._last_encode_mono < self._min_encode_interval_s:
            return
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            logger.warning("Live view expects HxWx3 RGB, got shape %s", rgb.shape)
            return
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        self._last_encode_mono = now
        with self._jpeg_lock:
            self._jpeg = encoded.tobytes()

    def get_jpeg(self) -> Optional[bytes]:
        with self._jpeg_lock:
            return self._jpeg


class NullView:
    """Stand-in used when the live view is disabled, so callers never branch on it."""

    def wait_ready(self) -> None:
        pass

    def attach_input(self, *args, **kwargs) -> None:
        pass

    def update(self, *args, **kwargs) -> None:
        pass

    def hold_finished(self, *args, **kwargs) -> None:
        pass


class LiveView:
    """High-level wrapper around :class:`LiveViewServer` for a deploy/sim run.

    Build via :meth:`start`; when disabled it returns a :class:`NullView` whose
    methods all no-op, so the run loop calls the same methods either way.
    """

    def __init__(self, env, server, replay, *, no_wait: bool):
        self._env = env
        self._server = server
        self._replay = replay
        self._no_wait = no_wait

    @classmethod
    def start(cls, controller, *, enabled: bool, port: int, replay, replay_stem,
              vr: bool, no_wait: bool, verbose: bool, realtime: bool = True):
        if not enabled:
            return NullView()
        # Local import: mujoco_env imports this module, so importing it at top
        # level would be circular.
        from lib.deploy.control.mujoco_env import G1MujocoEnv

        env = controller.env
        if not isinstance(env, G1MujocoEnv):
            raise SystemExit("--view requires MuJoCo sim (sim_sync_mujoco)")
        try:
            if replay is not None:
                env.start_live_view(port=port, mode="vr_replay", trajectory=replay_stem, realtime=realtime)
            elif vr:
                env.start_live_view(port=port, mode="vr", realtime=realtime)
            else:
                env.start_live_view(port=port, mode="wasd", realtime=realtime)
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
        server = env.live_view_server
        if server is not None:
            server.set_state(fsm_graph=controller.fsm_graph())  # static; rendered in-browser
        if verbose:
            print_path_lines("  view:       ", env.live_view_url)
        if no_wait and server is not None:
            server.request_start()
        return cls(env, server, replay, no_wait=no_wait)

    def wait_ready(self) -> None:
        """Pause before the run loop until the browser Start button or Enter."""
        import sys
        from select import select

        if self._no_wait or self._server is None:
            return

        self._server.set_state(status="waiting")

        if not sys.stdin.isatty():
            print("Open the view URL and click Start in the browser.", flush=True)
        else:
            print(flush=True)
            print("Open the view URL and click Start in the browser.", flush=True)
            print("(Or press Enter here to start without the button.)", flush=True)

        while not self._server.is_started():
            if self._server.wait_for_start(timeout=0.2):
                break
            if sys.stdin.isatty():
                ready, _, _ = select([sys.stdin], [], [], 0.0)
                if ready:
                    sys.stdin.readline()
                    self._server.request_start()
                    break
        print(flush=True)

    def attach_input(self, controller, inp, *, motion_enabled: bool) -> None:
        if self._server is None:
            return
        # Local import: keeps this control module free of a load-time
        # dependency on the controller package.
        from lib.controller.utils.input import InputReader

        if isinstance(inp, InputReader):
            browser_input = BrowserInputBridge(inp, motion_enabled=motion_enabled)
            self._server.set_input_handler(browser_input.handle_key)
            self.update(controller, inp, status="running")
        else:
            self.update(controller, None, status="running")

    def update(self, controller, inp, *, status: str = "running") -> None:
        if self._server is None:
            return
        vel = inp.get_velocity() if inp is not None else np.zeros(3, dtype=np.float32)
        fields = {
            "status": status,
            "sim_time": float(controller.get_env_time()),
            "fsm_node": _display_fsm_node(controller.current_node_name),
            "velocity": [float(vel[0]), float(vel[1]), float(vel[2])],
        }
        if self._replay is not None:
            fields["replay_frame"] = int(self._replay.current_index)
            fields["replay_total"] = int(self._replay.len())
        else:
            fields["replay_frame"] = 0
            fields["replay_total"] = 0
        self._server.set_state(**fields)

    def hold_finished(self, controller, inp, *, hold_s: float = 4.0) -> None:
        """Keep the browser view alive briefly so the finished state shows."""
        self.update(controller, inp, status="finished")
        time.sleep(hold_s)
