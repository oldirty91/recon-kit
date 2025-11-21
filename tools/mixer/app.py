#!/usr/bin/env python3
import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Set

import numpy as np
from aiohttp import web, WSMsgType

# ---------------------------------------------------------------------------
# Configuration via environment
# ---------------------------------------------------------------------------

HTTP_PORT = int(os.getenv("HTTP_PORT", "8087"))

OUTPUT_RATE = int(os.getenv("OUTPUT_RATE", "48000"))  # Hz
FRAME_MS = int(os.getenv("FRAME_MS", "20"))           # ms per mix frame
FRAME_SEC = FRAME_MS / 1000.0
STALE_SEC = float(os.getenv("STALE_SEC", "5"))

ALSA_DEVICE = os.getenv("ALSA_DEVICE", "default")     # e.g., "plughw:2,0"

# Example: "23456:p25:8000,40000:fm:48000"
INITIAL_INPUTS = os.getenv("INITIAL_INPUTS", "").strip()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("recon-mixer")

# Global set of WebSocket clients for audio
WS_CLIENTS: Set[web.WebSocketResponse] = set()

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class InputStream:
    port: int
    name: str
    sample_rate: int
    mute: bool = False
    last_packet_time: float = 0.0
    buffer: bytearray = None

    def __post_init__(self):
        if self.buffer is None:
            self.buffer = bytearray()


class UdpAudioProtocol(asyncio.DatagramProtocol):
    """Receives UDP PCM16-LE audio packets for a specific port."""
    def __init__(self, mixer: "Mixer", port: int):
        self.mixer = mixer
        self.port = port

    def datagram_received(self, data: bytes, addr: Tuple[str, int]):
        self.mixer.handle_udp_packet(self.port, data)


class Mixer:
    """
    Manages multiple UDP audio inputs, mixes them, and feeds:
      - ALSA via aplay
      - Optional WebSocket stream for browser playback
    Also supports an internal test tone generator.
    """

    def __init__(self, output_rate: int, frame_sec: float, stale_sec: float, alsa_device: str):
        self.output_rate = output_rate
        self.frame_sec = frame_sec
        self.frame_samples = int(round(output_rate * frame_sec))
        self.stale_sec = stale_sec
        self.alsa_device = alsa_device

        self.inputs: Dict[int, InputStream] = {}
        self.transports: Dict[int, asyncio.DatagramTransport] = {}

        self.aplay_proc: Optional[asyncio.subprocess.Process] = None
        self._mix_task: Optional[asyncio.Task] = None

        # Test tone state
        self.test_tone_enabled: bool = False
        self.test_tone_freq: float = 1000.0   # Hz
        self.test_tone_level: float = 0.5     # 0..1
        self._test_phase: float = 0.0         # radians

        # Output mode: "alsa", "web", or "both"
        self.output_mode: str = "alsa"

    # ---------------------- Input Management ----------------------

    async def add_input(self, port: int, name: str, sample_rate: int):
        """
        Register a new UDP input. Supports:
          - sample_rate == OUTPUT_RATE (e.g., 48000)
          - sample_rate == 8000 (upsampled to OUTPUT_RATE)
        Any other value is coerced to 8000 (for now) with a warning.
        """
        allowed = (8000, self.output_rate)
        if sample_rate not in allowed:
            log.warning(
                "Input %s on port %d requested unsupported sample_rate=%d; "
                "coercing to 8000 Hz",
                name, port, sample_rate,
            )
            sample_rate = 8000

        if port in self.inputs:
            # Update metadata only
            s = self.inputs[port]
            s.name = name
            s.sample_rate = sample_rate
            log.info("Updated existing input: %s on UDP %d @ %d Hz", name, port, sample_rate)
            return

        loop = asyncio.get_running_loop()
        protocol = UdpAudioProtocol(self, port)
        transport, _ = await loop.create_datagram_endpoint(
            lambda: protocol, local_addr=("0.0.0.0", port)
        )

        self.transports[port] = transport
        self.inputs[port] = InputStream(port, name, sample_rate)

        log.info("Added input: %s on UDP %d @ %d Hz", name, port, sample_rate)

    async def remove_input(self, port: int):
        if port in self.inputs:
            self.inputs.pop(port)

        tr = self.transports.pop(port, None)
        if tr is not None:
            tr.close()

        log.info("Removed input on port %d", port)

    async def set_mute(self, port: int, mute: bool):
        if port not in self.inputs:
            raise KeyError(f"Port {port} not found")
        self.inputs[port].mute = mute
        log.info("Mute on port %d = %s", port, mute)

    def handle_udp_packet(self, port: int, data: bytes):
        """Receive raw PCM16-LE data from UDP."""
        stream = self.inputs.get(port)
        if not stream:
            return

        if len(data) % 2 != 0:
            data = data[:-1]

        if stream.last_packet_time == 0.0:
            log.info("First UDP packet on port %d: %d bytes", port, len(data))

        stream.buffer.extend(data)
        stream.last_packet_time = time.time()

    # ---------------------- Test Tone ----------------------

    async def set_test_tone(self, enable: bool, freq: Optional[float] = None, level: Optional[float] = None):
        """
        Enable/disable and configure the test tone.
        freq: Hz (e.g. 1000)
        level: 0..1 (fraction of full scale)
        """
        self.test_tone_enabled = bool(enable)
        if freq is not None:
            try:
                self.test_tone_freq = float(freq)
            except ValueError:
                pass
        if level is not None:
            try:
                self.test_tone_level = max(0.0, min(1.0, float(level)))
            except ValueError:
                pass

        if self.test_tone_enabled:
            log.info(
                "Test tone enabled: freq=%.1f Hz, level=%.2f",
                self.test_tone_freq, self.test_tone_level
            )
        else:
            log.info("Test tone disabled")

    def _tone_frame_int32(self) -> np.ndarray:
        """
        Generate one frame of test tone as int32, length = frame_samples.
        """
        if not self.test_tone_enabled or self.test_tone_level <= 0.0:
            return np.zeros(self.frame_samples, dtype=np.int32)

        t = np.arange(self.frame_samples, dtype=np.float64)
        angle = self._test_phase + 2.0 * np.pi * self.test_tone_freq * t / self.output_rate
        tone = np.sin(angle)

        # Update phase for next frame
        total_phase_advance = 2.0 * np.pi * self.test_tone_freq * self.frame_samples / self.output_rate
        self._test_phase = (self._test_phase + total_phase_advance) % (2.0 * np.pi)

        amp = 30000.0 * float(self.test_tone_level)
        samples = (tone * amp).astype(np.int32)
        return samples

    # ---------------------- Output Mode ----------------------

    async def set_output_mode(self, mode: str):
        mode = mode.lower().strip()
        if mode not in ("alsa", "web", "both"):
            raise ValueError("mode must be 'alsa', 'web', or 'both'")
        self.output_mode = mode
        log.info("Output mode set to %s", mode)

    # ---------------------- ALSA Sink ----------------------

    async def ensure_sink(self):
        """Ensure aplay is running as the sink when in ALSA or BOTH mode."""
        if self.output_mode not in ("alsa", "both"):
            # Not using ALSA, no need to spawn aplay
            return

        if self.aplay_proc and self.aplay_proc.returncode is None:
            return

        if self.aplay_proc:
            log.warning("Restarting ALSA sink (previous code=%s)", self.aplay_proc.returncode)

        cmd = [
            "aplay", "-q",
            "-t", "raw",
            "-f", "S16_LE",
            "-c", "1",
            "-r", str(self.output_rate),
        ]

        if self.alsa_device and self.alsa_device != "default":
            cmd.extend(["-D", self.alsa_device])

        log.info("Starting aplay sink: %s", " ".join(cmd))

        self.aplay_proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

    # ---------------------- Mixing Loop ----------------------

    async def mixing_loop(self):
        await self.ensure_sink()
        log.info(
            "Mixer started: output_rate=%d Hz frame_ms=%d frame_samples=%d",
            self.output_rate, FRAME_MS, self.frame_samples,
        )

        while True:
            await asyncio.sleep(self.frame_sec)

            # Ensure ALSA sink if in ALSA/BOTH mode
            if self.output_mode in ("alsa", "both"):
                await self.ensure_sink()

            now = time.time()
            frames = []

            # Collect frames from UDP inputs
            for port, stream in self.inputs.items():
                if stream.mute:
                    continue
                if not stream.last_packet_time or now - stream.last_packet_time > self.stale_sec:
                    continue

                if stream.sample_rate == self.output_rate:
                    in_samples = self.frame_samples
                else:
                    factor = self.output_rate // stream.sample_rate
                    in_samples = self.frame_samples // factor

                in_bytes = in_samples * 2
                if len(stream.buffer) < in_bytes:
                    continue

                raw = stream.buffer[:in_bytes]
                del stream.buffer[:in_bytes]

                arr = np.frombuffer(raw, dtype=np.int16)

                if stream.sample_rate != self.output_rate:
                    factor = self.output_rate // stream.sample_rate
                    arr = np.repeat(arr, factor)

                if len(arr) != self.frame_samples:
                    if len(arr) < self.frame_samples:
                        arr = np.pad(arr, (0, self.frame_samples - len(arr)))
                    else:
                        arr = arr[:self.frame_samples]

                frames.append(arr.astype(np.int32))

            # Test tone as another "frame"
            if self.test_tone_enabled:
                frames.append(self._tone_frame_int32())

            if not frames:
                mixed = np.zeros(self.frame_samples, dtype=np.int16)
            else:
                total = np.zeros(self.frame_samples, dtype=np.int32)
                for f in frames:
                    total += f
                total //= max(len(frames), 1)
                total = np.clip(total, -32768, 32767)
                mixed = total.astype(np.int16)

            data_bytes = mixed.tobytes()

            # ALSA output
            if self.output_mode in ("alsa", "both") and self.aplay_proc and self.aplay_proc.stdin:
                try:
                    self.aplay_proc.stdin.write(data_bytes)
                except BrokenPipeError:
                    log.error("Broken pipe to aplay; resetting sink")
                    self.aplay_proc = None

            # WebSocket broadcast
            if self.output_mode in ("web", "both") and WS_CLIENTS:
                dead = []
                for ws in list(WS_CLIENTS):
                    if ws.closed:
                        dead.append(ws)
                        continue
                    try:
                        await ws.send_bytes(data_bytes)
                    except Exception as e:
                        log.warning("WS send failed: %s", e)
                        dead.append(ws)
                for ws in dead:
                    WS_CLIENTS.discard(ws)

    # ---------------------- Introspection ----------------------

    def inputs_status(self):
        now = time.time()
        out = []
        for port, s in self.inputs.items():
            age = (now - s.last_packet_time) if s.last_packet_time else None
            out.append({
                "port": port,
                "name": s.name,
                "sample_rate": s.sample_rate,
                "mute": s.mute,
                "age_sec": age,
                "buffer_bytes": len(s.buffer),
            })
        return out

    def status(self):
        return {
            "http_port": HTTP_PORT,
            "output_rate": self.output_rate,
            "alsa_device": self.alsa_device,
            "frame_ms": FRAME_MS,
            "frame_samples": self.frame_samples,
            "inputs": self.inputs_status(),
            "aplay_running": bool(self.aplay_proc and self.aplay_proc.returncode is None),
            "time": time.time(),
            "test_tone_enabled": self.test_tone_enabled,
            "test_tone_freq": self.test_tone_freq,
            "test_tone_level": self.test_tone_level,
            "output_mode": self.output_mode,
            "ws_clients": len(WS_CLIENTS),
        }

    async def start(self):
        self._mix_task = asyncio.create_task(self.mixing_loop())

    async def shutdown(self):
        if self._mix_task:
            self._mix_task.cancel()
            try:
                await self._mix_task
            except asyncio.CancelledError:
                pass

        for t in self.transports.values():
            t.close()
        self.transports.clear()

        if self.aplay_proc:
            try:
                self.aplay_proc.stdin.close()
            except Exception:
                pass
            try:
                await self.aplay_proc.wait()
            except Exception:
                pass
            self.aplay_proc = None


# ---------------------------------------------------------------------------
# Web GUI HTML
# ---------------------------------------------------------------------------

HTML_PAGE = r"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Recon Mixer</title>
  <style>
    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0b1020;
      color: #f5f5f5;
      margin: 0;
      padding: 16px;
    }
    h1, h2 {
      margin: 0 0 8px 0;
    }
    .card {
      background: #141a33;
      border-radius: 8px;
      padding: 12px 16px;
      margin-bottom: 16px;
      box-shadow: 0 2px 6px rgba(0,0,0,0.4);
    }
    .row {
      display: flex;
      gap: 16px;
      flex-wrap: wrap;
    }
    .col {
      flex: 1 1 300px;
    }
    label {
      display: block;
      font-size: 0.85rem;
      margin-bottom: 4px;
    }
    input, select {
      width: 100%;
      padding: 6px 8px;
      border-radius: 4px;
      border: 1px solid #333;
      background: #0d1224;
      color: #f5f5f5;
      margin-bottom: 8px;
    }
    button {
      background: #2f6bff;
      border: none;
      border-radius: 4px;
      padding: 6px 10px;
      color: #fff;
      cursor: pointer;
      font-size: 0.85rem;
    }
    button:hover {
      background: #4f84ff;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
    }
    th, td {
      padding: 4px 6px;
      border-bottom: 1px solid #252b48;
      text-align: left;
    }
    th {
      background: #1b2340;
    }
    .tag {
      display: inline-block;
      padding: 2px 6px;
      border-radius: 999px;
      font-size: 0.7rem;
      background: #1f2b4f;
      color: #cbd5ff;
    }
    .tag.ok {
      background: #165f31;
      color: #c4f3d2;
    }
    .tag.bad {
      background: #7a1a1a;
      color: #f7d3d3;
    }
    pre {
      background: #0d1224;
      padding: 8px;
      border-radius: 4px;
      overflow-x: auto;
      font-size: 0.8rem;
    }
    .btn-small {
      font-size: 0.75rem;
      padding: 3px 6px;
      margin-right: 4px;
    }
    .tone-controls {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }
    .tone-controls input {
      width: 80px;
    }
    .output-controls {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
      margin-top: 8px;
      margin-bottom: 4px;
    }
  </style>
</head>
<body>
  <h1>Recon Mixer</h1>

  <div class="row">
    <div class="col">
      <div class="card">
        <h2>Status</h2>
        <div id="status-summary">
          Loading...
        </div>
        <div class="output-controls">
          <div>
            <label for="output_mode">Output Mode</label>
            <select id="output_mode">
              <option value="alsa">ALSA (Pi hardware)</option>
              <option value="web">Web (this browser)</option>
              <option value="both">Both</option>
            </select>
          </div>
          <div>
            <button id="btn-output-apply" type="button">Apply Output Mode</button>
          </div>
          <div>
            <button id="btn-web-audio" type="button">Enable Browser Audio</button>
          </div>
        </div>
      </div>

      <div class="card">
        <h2>Inputs</h2>
        <table>
          <thead>
            <tr>
              <th>Port</th>
              <th>Name</th>
              <th>Rate</th>
              <th>Mute</th>
              <th>Age (s)</th>
              <th>Buffer (bytes)</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody id="inputs-body">
            <tr><td colspan="7">Loading...</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <div class="col">
      <div class="card">
        <h2>Add / Update Input</h2>
        <form id="add-form">
          <label for="port">UDP Port</label>
          <input id="port" name="port" type="number" min="1" max="65535" required placeholder="e.g. 23456">

          <label for="name">Name</label>
          <input id="name" name="name" type="text" placeholder="e.g. p25">

          <label for="sample_rate">Sample Rate</label>
          <select id="sample_rate" name="sample_rate">
            <option value="8000">8000 Hz (OP25 etc.)</option>
            <option value="48000">48000 Hz</option>
          </select>

          <button type="submit">Add / Update</button>
        </form>
      </div>

      <div class="card">
        <h2>Test Tone</h2>
        <div class="tone-controls">
          <div>
            <label for="tone_freq">Frequency (Hz)</label>
            <input id="tone_freq" type="number" value="1000" min="50" max="20000">
          </div>
          <div>
            <label for="tone_level">Level (0–1)</label>
            <input id="tone_level" type="number" step="0.1" value="0.5" min="0" max="1">
          </div>
        </div>
        <p style="margin-top:8px;">
          <button id="btn-tone-on" type="button">Start Test Tone</button>
          <button id="btn-tone-off" type="button">Stop Test Tone</button>
          <span id="tone-status" class="tag" style="margin-left:8px;">unknown</span>
        </p>
      </div>

      <div class="card">
        <h2>Raw Status JSON</h2>
        <pre id="status-json">Loading...</pre>
      </div>
    </div>
  </div>

<script>
let audioCtx = null;
let ws = null;
let nextPlayTime = 0;

async function fetchJSON(url, options) {
  const res = await fetch(url, options || {});
  if (!res.ok) {
    throw new Error("HTTP " + res.status);
  }
  return res.json();
}

function fmtSeconds(sec) {
  if (sec === null || sec === undefined) return "-";
  return sec.toFixed(1);
}

function ensureWebAudio() {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || window.webkitAudioContext)({sampleRate: 48000});
  }
  if (audioCtx.state === "suspended") {
    audioCtx.resume();
  }
  if (nextPlayTime < audioCtx.currentTime) {
    nextPlayTime = audioCtx.currentTime;
  }
}

function connectWS() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
    return;
  }
  const proto = (location.protocol === "https:") ? "wss://" : "ws://";
  ws = new WebSocket(proto + location.host + "/ws/audio");
  ws.binaryType = "arraybuffer";

  ws.onopen = () => {
    console.log("WS audio connected");
  };
  ws.onclose = () => {
    console.log("WS audio closed");
    setTimeout(connectWS, 2000);
  };
  ws.onerror = (e) => {
    console.log("WS audio error", e);
  };
  ws.onmessage = (event) => {
    if (!audioCtx) return;
    const buf = event.data;
    const int16 = new Int16Array(buf);
    const audioBuf = audioCtx.createBuffer(1, int16.length, 48000);
    const ch = audioBuf.getChannelData(0);
    for (let i = 0; i < int16.length; i++) {
      ch[i] = int16[i] / 32768.0;
    }
    const src = audioCtx.createBufferSource();
    src.buffer = audioBuf;
    src.connect(audioCtx.destination);

    if (nextPlayTime < audioCtx.currentTime) {
      nextPlayTime = audioCtx.currentTime;
    }
    src.start(nextPlayTime);
    nextPlayTime += audioBuf.duration;
  };
}

async function refreshStatus() {
  try {
    const status = await fetchJSON("/status");
    const inputs = status.inputs || [];

    const statusSummary = document.getElementById("status-summary");
    const aplayTag = status.aplay_running
      ? '<span class="tag ok">aplay: running</span>'
      : '<span class="tag bad">aplay: stopped</span>';

    const toneTag = status.test_tone_enabled
      ? '<span class="tag ok">test tone: on</span>'
      : '<span class="tag bad">test tone: off</span>';

    const modeTag = `<span class="tag">${status.output_mode}</span>`;
    const wsTag = `<span class="tag">${status.ws_clients} WS clients</span>`;

    statusSummary.innerHTML = `
      <div>Output rate: <b>${status.output_rate}</b> Hz</div>
      <div>ALSA device: <b>${status.alsa_device}</b></div>
      <div>Frame: <b>${status.frame_ms}</b> ms (${status.frame_samples} samples)</div>
      <div>${aplayTag} &nbsp; Mode: ${modeTag} &nbsp; ${wsTag}</div>
      <div>${toneTag} (freq=${status.test_tone_freq} Hz, level=${status.test_tone_level})</div>
      <div>Inputs: <b>${inputs.length}</b></div>
    `;

    // Output mode select
    const sel = document.getElementById("output_mode");
    if (status.output_mode && ["alsa","web","both"].includes(status.output_mode)) {
      sel.value = status.output_mode;
    }

    // Tone controls display
    document.getElementById("tone_freq").value = status.test_tone_freq || 1000;
    document.getElementById("tone_level").value = status.test_tone_level || 0.5;
    const toneStatus = document.getElementById("tone-status");
    toneStatus.textContent = status.test_tone_enabled ? "ON" : "OFF";
    toneStatus.className = "tag " + (status.test_tone_enabled ? "ok" : "bad");

    // Inputs table
    const tbody = document.getElementById("inputs-body");
    if (!inputs.length) {
      tbody.innerHTML = '<tr><td colspan="7">No inputs configured</td></tr>';
    } else {
      tbody.innerHTML = "";
      for (const s of inputs) {
        const tr = document.createElement("tr");
        const muteLabel = s.mute ? "Muted" : "Active";
        const muteClass = s.mute ? "bad" : "ok";
        tr.innerHTML = `
          <td>${s.port}</td>
          <td>${s.name}</td>
          <td>${s.sample_rate}</td>
          <td><span class="tag ${muteClass}">${muteLabel}</span></td>
          <td>${fmtSeconds(s.age_sec)}</td>
          <td>${s.buffer_bytes}</td>
          <td>
            <button class="btn-small" data-action="toggle-mute" data-port="${s.port}">
              ${s.mute ? "Unmute" : "Mute"}
            </button>
            <button class="btn-small" data-action="remove" data-port="${s.port}">
              Remove
            </button>
          </td>
        `;
        tbody.appendChild(tr);
      }
    }

    document.getElementById("status-json").textContent =
      JSON.stringify(status, null, 2);
  } catch (err) {
    console.error("refreshStatus error:", err);
    document.getElementById("status-summary").innerHTML =
      '<span class="tag bad">Error loading status</span>';
  }
}

// Handle add/update form
document.getElementById("add-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const port = parseInt(document.getElementById("port").value, 10);
  const name = document.getElementById("name").value || ("input-" + port);
  const sample_rate = parseInt(document.getElementById("sample_rate").value, 10);

  try {
    await fetchJSON("/add_input", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ port, name, sample_rate }),
    });
    await refreshStatus();
  } catch (err) {
    alert("Error adding input: " + err.message);
  }
});

// Handle mute/remove buttons
document.getElementById("inputs-body").addEventListener("click", async (e) => {
  const btn = e.target.closest("button");
  if (!btn) return;
  const action = btn.dataset.action;
  const port = parseInt(btn.dataset.port, 10);
  if (!port) return;

  try {
    if (action === "toggle-mute") {
      const status = await fetchJSON("/status");
      const s = (status.inputs || []).find(x => x.port === port);
      if (!s) throw new Error("Input not found");
      const newMute = !s.mute;
      await fetchJSON("/mute_input", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ port, mute: newMute }),
      });
    } else if (action === "remove") {
      await fetchJSON("/remove_input", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ port }),
      });
    }
    await refreshStatus();
  } catch (err) {
    alert("Error: " + err.message);
  }
});

// Test tone buttons
async function sendTestTone(enable) {
  const freq = parseFloat(document.getElementById("tone_freq").value) || 1000;
  const level = parseFloat(document.getElementById("tone_level").value) || 0.5;
  try {
    await fetchJSON("/test_tone", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ enable, freq, level }),
    });
    await refreshStatus();
  } catch (err) {
    alert("Error setting test tone: " + err.message);
  }
}
document.getElementById("btn-tone-on").addEventListener("click", () => sendTestTone(true));
document.getElementById("btn-tone-off").addEventListener("click", () => sendTestTone(false));

// Output mode controls
document.getElementById("btn-output-apply").addEventListener("click", async () => {
  const mode = document.getElementById("output_mode").value;
  try {
    await fetchJSON("/output_mode", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ mode }),
    });
    await refreshStatus();
  } catch (err) {
    alert("Error setting output mode: " + err.message);
  }
});

// Browser audio enable
document.getElementById("btn-web-audio").addEventListener("click", () => {
  ensureWebAudio();
});

// Connect WS on load
connectWS();

// Initial load + periodic refresh
refreshStatus();
setInterval(refreshStatus, 2000);
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# HTTP Handlers (JSON + UI + WS)
# ---------------------------------------------------------------------------

async def handle_health(req):
    return web.json_response({"status": "ok"})

async def handle_status(req):
    return web.json_response(req.app["mixer"].status())

async def handle_inputs(req):
    return web.json_response(req.app["mixer"].inputs_status())

async def handle_add(req):
    mixer: Mixer = req.app["mixer"]
    data = await req.json()
    try:
        port = int(data["port"])
        name = data.get("name", f"input-{port}")
        sr = int(data.get("sample_rate", 8000))
    except (KeyError, ValueError, TypeError) as e:
        return web.json_response({"error": f"Invalid payload: {e}"}, status=400)

    await mixer.add_input(port, name, sr)
    return web.json_response({"ok": True})

async def handle_remove(req):
    mixer: Mixer = req.app["mixer"]
    data = await req.json()
    try:
        port = int(data["port"])
    except (KeyError, ValueError, TypeError) as e:
        return web.json_response({"error": f"Invalid payload: {e}"}, status=400)

    await mixer.remove_input(port)
    return web.json_response({"ok": True})

async def handle_mute(req):
    mixer: Mixer = req.app["mixer"]
    data = await req.json()
    try:
        port = int(data["port"])
        mute = bool(data["mute"])
    except (KeyError, ValueError, TypeError) as e:
        return web.json_response({"error": f"Invalid payload: {e}"}, status=400)

    try:
        await mixer.set_mute(port, mute)
    except KeyError as e:
        return web.json_response({"error": str(e)}, status=404)

    return web.json_response({"ok": True, "port": port, "mute": mute})

async def handle_test_tone(req):
    mixer: Mixer = req.app["mixer"]
    data = await req.json()
    try:
        enable = bool(data.get("enable", True))
        freq = data.get("freq", None)
        level = data.get("level", None)
    except Exception as e:
        return web.json_response({"error": f"Invalid payload: {e}"}, status=400)

    await mixer.set_test_tone(enable, freq=freq, level=level)
    return web.json_response({"ok": True})

async def handle_output_mode(req):
    mixer: Mixer = req.app["mixer"]
    data = await req.json()
    mode = str(data.get("mode", "alsa"))
    try:
        await mixer.set_output_mode(mode)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    return web.json_response({"ok": True, "mode": mixer.output_mode})

async def handle_ui(req):
    return web.Response(text=HTML_PAGE, content_type="text/html")

async def handle_ws_audio(request: web.Request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    WS_CLIENTS.add(ws)
    log.info("WS audio client connected (total=%d)", len(WS_CLIENTS))

    try:
        async for msg in ws:
            if msg.type == WSMsgType.ERROR:
                log.warning("WS connection closed with exception %s", ws.exception())
    finally:
        WS_CLIENTS.discard(ws)
        log.info("WS audio client disconnected (total=%d)", len(WS_CLIENTS))

    return ws

# ---------------------------------------------------------------------------
# Startup / Shutdown
# ---------------------------------------------------------------------------

async def on_startup(app):
    mixer: Mixer = app["mixer"]

    # INITIAL_INPUTS: "23456:p25:8000,40000:fm:48000" etc.
    if INITIAL_INPUTS:
        for spec in INITIAL_INPUTS.split(","):
            spec = spec.strip()
            if not spec:
                continue
            parts = spec.split(":")
            if len(parts) not in (2, 3):
                log.warning("Invalid INITIAL_INPUTS spec: %s", spec)
                continue
            try:
                port = int(parts[0])
                name = parts[1] if parts[1] else f"input-{port}"
                sr = int(parts[2]) if len(parts) == 3 else 8000
                await mixer.add_input(port, name, sr)
            except Exception as e:
                log.error("Failed to add INITIAL_INPUTS spec '%s': %s", spec, e)

    await mixer.start()

async def on_cleanup(app):
    await app["mixer"].shutdown()

def create_app():
    mixer = Mixer(OUTPUT_RATE, FRAME_SEC, STALE_SEC, ALSA_DEVICE)
    app = web.Application()
    app["mixer"] = mixer

    # UI
    app.router.add_get("/", handle_ui)

    # JSON API
    app.router.add_get("/healthz", handle_health)
    app.router.add_get("/status", handle_status)
    app.router.add_get("/inputs", handle_inputs)
    app.router.add_post("/add_input", handle_add)
    app.router.add_post("/remove_input", handle_remove)
    app.router.add_post("/mute_input", handle_mute)
    app.router.add_post("/test_tone", handle_test_tone)
    app.router.add_post("/output_mode", handle_output_mode)

    # WebSocket audio
    app.router.add_get("/ws/audio", handle_ws_audio)

    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)

    return app

def main():
    log.info("Starting recon-mixer on port %d", HTTP_PORT)
    web.run_app(create_app(), host="0.0.0.0", port=HTTP_PORT)

if __name__ == "__main__":
    main()
