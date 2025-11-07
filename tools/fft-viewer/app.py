#!/usr/bin/env python3
import os
import json
import time
import socket
import threading
from dataclasses import dataclass, field

import numpy as np
import requests
from flask import Flask, jsonify, request
from flask_sock import Sock

HUB_URL = os.getenv("HUB_URL", "http://hub:8080")
HUB_HOST = os.getenv("HUB_HOST", "hub")
FFT_PORT = int(os.getenv("FFT_PORT", "8090"))

FFT_SIZE = 1024           # internal FFT size, was 2048
DOWNSAMPLED_BINS = 256     # bins sent to browser, was 512
IDLE_TIMEOUT = 30.0        # seconds without viewer activity => disconnect IQ

app = Flask(__name__)
sock = Sock(app)


# --------- Helpers to talk to hub ---------


def get_sdrs():
    r = requests.get(f"{HUB_URL}/api/sdrs", timeout=5)
    r.raise_for_status()
    return r.json()


def find_sdr(serial):
    for s in get_sdrs():
        if s.get("serial") == serial:
            return s
    return None


def send_control(serial, freq=None, samp_rate=None, gain=None):
    print(f"[fft] send_control: serial={serial} freq={freq} sr={samp_rate} gain={gain}", flush=True)
    dev = find_sdr(serial)
    if not dev:
        print(f"[fft] send_control: not_found for serial={serial}", flush=True)
        return {"ok": False, "error": "not_found"}

    ctl_port = dev.get("control_port")
    if not ctl_port:
        print(f"[fft] send_control: missing control_port for serial={serial}", flush=True)
        return {"ok": False, "error": "missing_control_port"}

    cmd = {"cmd": "set_config"}
    if freq is not None:
        cmd["freq"] = int(freq)
    if samp_rate is not None:
        cmd["samp_rate"] = int(samp_rate)
    if gain is not None:
        cmd["gain"] = int(gain)

    print(f"[fft] send_control -> hub:{ctl_port} {cmd}", flush=True)

    try:
        with socket.create_connection((HUB_HOST, ctl_port), timeout=5) as s:
            s.sendall((json.dumps(cmd) + "\n").encode("utf-8"))
            data = s.recv(4096).decode("utf-8").strip()
            print(f"[fft] send_control resp: {data}", flush=True)
            if not data:
                return {"ok": False, "error": "no_response"}
            return json.loads(data)
    except Exception as e:
        print(f"[fft] send_control error: {e}", flush=True)
        return {"ok": False, "error": str(e)}


# --------- Streaming state / background workers ---------


@dataclass
class StreamState:
    serial: str
    latest_bins: list = field(default_factory=list)
    last_error: str | None = None
    last_access: float = field(default_factory=lambda: time.time())
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    thread: threading.Thread | None = None
    running: bool = False


streams: dict[str, StreamState] = {}
streams_lock = threading.Lock()


def ensure_stream(serial: str) -> StreamState:
    """
    Get or create a StreamState and start a background worker thread
    that keeps a persistent IQ connection while the viewer is active.
    """
    with streams_lock:
        state = streams.get(serial)
        if state is None:
            state = StreamState(serial=serial)
            streams[serial] = state

        if not state.running or state.thread is None or not state.thread.is_alive():
            state.running = True
            t = threading.Thread(target=stream_worker, args=(state,), daemon=True)
            state.thread = t
            t.start()

        return state


def stream_worker(state: StreamState):
    """
    Background worker: for a given SDR serial, maintain a connection to
    the rtl_tcp IQ port *only* while the viewer has polled recently.
    Continuously updates state.latest_bins with the most recent FFT.
    """
    sock = None

    while state.running:
        now = time.time()
        idle = now - state.last_access

        # If nobody has hit the viewer for a while, close IQ and idle.
        if idle > IDLE_TIMEOUT:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass
                sock = None
            time.sleep(0.5)
            continue

        # Make sure we know where this SDR's IQ port is.
        dev = None
        try:
            dev = find_sdr(state.serial)
        except Exception as e:
            with state.lock:
                state.last_error = f"sdr_lookup_failed: {e}"
            time.sleep(1.0)
            continue

        if not dev:
            with state.lock:
                state.last_error = "sdr_not_found"
            time.sleep(1.0)
            continue

        iq_port = dev.get("iq_port")
        if not iq_port:
            with state.lock:
                state.last_error = "missing_iq_port"
            time.sleep(1.0)
            continue

        # Connect if needed
        if sock is None:
            try:
                sock = socket.create_connection((HUB_HOST, iq_port), timeout=5)
                sock.settimeout(1.0)
                print(f"[fft] stream_worker: connected to IQ for serial={state.serial} port={iq_port}", flush=True)
                with state.lock:
                    state.last_error = None
            except Exception as e:
                with state.lock:
                    state.last_error = f"iq_connect_failed: {e}"
                sock = None
                time.sleep(1.0)
                continue

        # Read IQ and compute FFT
        try:
            data = sock.recv(4096)
            if not data:
                # server closed; reconnect next loop
                sock.close()
                sock = None
                time.sleep(0.2)
                continue
        except socket.timeout:
            # Just try again
            continue
        except Exception as e:
            with state.lock:
                state.last_error = f"recv_error: {e}"
            try:
                sock.close()
            except Exception:
                pass
            sock = None
            time.sleep(0.5)
            continue

        # rtl_tcp default: unsigned 8-bit interleaved I/Q
        arr = np.frombuffer(data, dtype=np.uint8)
        if arr.size < 2 * FFT_SIZE:
            continue

        arr = arr[-2 * FFT_SIZE :]
        i = arr[0::2].astype(np.float32) - 127.5
        q = arr[1::2].astype(np.float32) - 127.5
        samples = i + 1j * q

        window = np.hanning(FFT_SIZE)
        spectrum = np.fft.fftshift(np.fft.fft(samples * window))
        psd = 20 * np.log10(np.abs(spectrum) + 1e-6)

        if psd.size >= DOWNSAMPLED_BINS:
            psd_resampled = psd.reshape(DOWNSAMPLED_BINS, -1).mean(axis=1)
        else:
            psd_resampled = psd

        with state.lock:
            state.latest_bins = psd_resampled.tolist()
            # last_error stays as-is if None / cleared earlier


# --------- WebSocket FFT endpoint (Flask-Sock) ---------


@sock.route("/ws/fft/<serial>")
def ws_fft(ws, serial):
    """
    WebSocket: push FFT bins for this SDR at a steady rate.
    Uses the same StreamState / stream_worker as the HTTP polling path.
    """
    print(f"[fft] WebSocket client connected for serial={serial}", flush=True)
    state = ensure_stream(serial)

    try:
        while True:
            state.last_access = time.time()

            with state.lock:
                bins = list(state.latest_bins)
                err = state.last_error

            if not bins:
                msg = {"ok": False, "serial": serial, "error": err or "no_data_yet"}
            else:
                msg = {"ok": True, "serial": serial, "bins": bins, "error": err}

            ws.send(json.dumps(msg))
            # ~20 Hz
            # time.sleep(0.05) 
            time.sleep(0.02) #was .05
    except Exception as e:
        # Any send/connection error ends up here – treat it as a normal close.
        print(f"[fft] WebSocket closed for serial={serial}: {e}", flush=True)
    finally:
        print(f"[fft] WebSocket handler exit for serial={serial}", flush=True)



# --------- HTTP API (SDR list, config, optional FFT fallback) ---------


@app.get("/api/sdrs")
def api_sdrs():
    try:
        return jsonify(get_sdrs())
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/api/sdr/<serial>/config")
def api_config_sdr(serial):
    print(f"[fft] api_config_sdr hit for serial={serial}", flush=True)
    data = request.get_json(force=True) or {}
    freq = data.get("freq")
    sr = data.get("samp_rate") or data.get("rate")
    gain = data.get("gain")
    resp = send_control(serial, freq=freq, samp_rate=sr, gain=gain)
    status = 200 if resp.get("ok", True) else 500
    return jsonify(resp), status


@app.get("/api/fft/<serial>")
def api_fft(serial):
    """
    Optional HTTP polling FFT endpoint (not used by the WS UI, but handy for debugging).
    """
    state = ensure_stream(serial)
    state.last_access = time.time()

    with state.lock:
        bins = list(state.latest_bins)
        err = state.last_error

    if not bins:
        return jsonify({"ok": False, "error": err or "no_data_yet"}), 200

    return jsonify(
        {
            "ok": True,
            "serial": serial,
            "bins": bins,
            "error": err,
        }
    )


# --------- Web UI (FFT + Waterfall) ---------


@app.route("/")
def index():
    html = """<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>RECON-KIT FFT Viewer</title>
<style>
body {
  font-family: system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  margin: 0;
  padding: 16px;
  background: #020617;
  color: #e5e7eb;
}
h1 { margin: 0 0 8px 0; }
.topbar {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}
button {
  padding: 4px 10px;
  border-radius: 999px;
  border: none;
  cursor: pointer;
  font-size: 13px;
  background: #1f2937;
  color: #e5e7eb;
}
button:hover { background: #4b5563; }
#panels {
  display: grid;
  grid-template-columns: repeat(auto-fit,minmax(320px,1fr));
  gap: 12px;
}
.panel {
  background: #020617;
  border: 1px solid #111827;
  border-radius: 10px;
  padding: 8px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.panel-header {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: center;
  font-size: 12px;
}
.panel-header select,
.panel-header input {
  background: #020617;
  border: 1px solid #1f2937;
  border-radius: 999px;
  padding: 2px 6px;
  color: #e5e7eb;
  font-size: 12px;
  width: 90px;
}
.panel-header label { font-size: 11px; color: #9ca3af; }
.canvas-wrap {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
canvas {
  width: 100%;
  background: #020617;
  border-radius: 8px;
}
.fft-canvas {
  height: 140px;
}
.waterfall-canvas {
  height: 140px;
}
.badge {
  font-size: 11px;
  padding: 2px 6px;
  border-radius: 999px;
  background: #111827;
  color: #9ca3af;
}
.status-text {
  font-size: 11px;
  color: #9ca3af;
}
</style>
</head>
<body>
  <h1>RECON-KIT FFT Viewer</h1>
  <div class="topbar">
    <span class="badge">Tools / FFT + Waterfall</span>
    <button id="add-panel">Add Panel</button>
  </div>
  <div id="panels"></div>

<script>
let sdrList = [];
let panelIdCounter = 0;
const panels = new Map(); // id -> panel object

async function loadSDRs() {
  try {
    const res = await fetch("/api/sdrs");
    if (!res.ok) return;
    sdrList = await res.json();
    panels.forEach(p => p.refreshSDRSelect());
  } catch(e) {}
}

class FFTPanel {
  constructor(id) {
    this.id = id;
    this.serial = null;
    this.ws = null;

    this.root = document.createElement("div");
    this.root.className = "panel";

    this.header = document.createElement("div");
    this.header.className = "panel-header";

    this.sdrSelect = document.createElement("select");
    this.sdrSelect.innerHTML = "<option value=''>Select SDR</option>";
    this.sdrSelect.onchange = () => this.setSerial(this.sdrSelect.value);

    this.freqInput = document.createElement("input");
    this.freqInput.placeholder = "Freq Hz";

    this.rateInput = document.createElement("input");
    this.rateInput.placeholder = "Rate Hz";

    this.gainInput = document.createElement("input");
    this.gainInput.placeholder = "Gain";

    const applyBtn = document.createElement("button");
    applyBtn.textContent = "Apply";
    applyBtn.onclick = () => this.applyConfig();

    const closeBtn = document.createElement("button");
    closeBtn.textContent = "Close";
    closeBtn.onclick = () => this.destroy();

    const label = (text, el) => {
      const span = document.createElement("span");
      span.textContent = text + " ";
      span.style.whiteSpace = "nowrap";
      span.appendChild(el);
      return span;
    };

    this.header.appendChild(this.sdrSelect);
    this.header.appendChild(label("F:", this.freqInput));
    this.header.appendChild(label("R:", this.rateInput));
    this.header.appendChild(label("G:", this.gainInput));
    this.header.appendChild(applyBtn);
    this.header.appendChild(closeBtn);

    this.status = document.createElement("div");
    this.status.className = "status-text";
    this.status.textContent = "Idle";

    this.canvasWrap = document.createElement("div");
    this.canvasWrap.className = "canvas-wrap";

    // FFT line canvas
    this.fftCanvas = document.createElement("canvas");
    this.fftCanvas.width = 600;
    this.fftCanvas.height = 140;
    this.fftCanvas.className = "fft-canvas";
    this.fftCtx = this.fftCanvas.getContext("2d");

    // Waterfall canvas
    this.waterCanvas = document.createElement("canvas");
    this.waterCanvas.width = 600;
    this.waterCanvas.height = 140;
    this.waterCanvas.className = "waterfall-canvas";
    this.waterCtx = this.waterCanvas.getContext("2d");

    this.canvasWrap.appendChild(this.fftCanvas);
    this.canvasWrap.appendChild(this.waterCanvas);

    this.root.appendChild(this.header);
    this.root.appendChild(this.status);
    this.root.appendChild(this.canvasWrap);
    document.getElementById("panels").appendChild(this.root);

    this.refreshSDRSelect();
    this.drawEmpty();
  }

  refreshSDRSelect() {
    const current = this.serial;
    this.sdrSelect.innerHTML = "<option value=''>Select SDR</option>";
    sdrList.forEach(sdr => {
      const opt = document.createElement("option");
      opt.value = sdr.serial;
      opt.textContent = `${sdr.serial} (IQ:${sdr.iq_port})`;
      if (sdr.serial === current) opt.selected = true;
      this.sdrSelect.appendChild(opt);
    });
  }

  setSerial(serial) {
    if (serial === this.serial) return;
    this.serial = serial || null;
    this.closeWS();
    if (this.serial) {
      this.status.textContent = `Viewing ${this.serial}`;
      this.openWS();
    } else {
      this.status.textContent = "Idle";
      this.drawEmpty();
    }
  }

  openWS() {
    if (!this.serial) return;
    const loc = window.location;
    const proto = loc.protocol === "https:" ? "wss" : "ws";
    const url = `${proto}://${loc.host}/ws/fft/${encodeURIComponent(this.serial)}`;
    const ws = new WebSocket(url);
    this.ws = ws;

    ws.onopen = () => {
      this.status.textContent = `Viewing ${this.serial} (live)`;
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (!data.ok || !data.bins) {
          if (data.error) {
            this.status.textContent = `Error: ${data.error}`;
          }
          return;
        }
        this.status.textContent = `Viewing ${this.serial} (live)`;
        this.drawSpectrum(data.bins);
        this.drawWaterfall(data.bins);
      } catch (e) {
        this.status.textContent = "Error parsing data";
      }
    };

    ws.onerror = () => {
      this.status.textContent = "WebSocket error";
    };

    ws.onclose = () => {
      if (this.serial) {
        this.status.textContent = "WebSocket closed";
      }
    };
  }

  closeWS() {
    if (this.ws) {
      try { this.ws.close(); } catch(e) {}
      this.ws = null;
    }
  }

  async applyConfig() {
    if (!this.serial) return;
    const payload = {};
    const f = parseInt(this.freqInput.value);
    const r = parseInt(this.rateInput.value);
    const g = parseInt(this.gainInput.value);
    if (!isNaN(f)) payload.freq = f;
    if (!isNaN(r)) payload.samp_rate = r;
    if (!isNaN(g)) payload.gain = g;

    await fetch(`/api/sdr/${encodeURIComponent(this.serial)}/config`, {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(payload),
    });
  }

  drawEmpty() {
    const ctx1 = this.fftCtx;
    const ctx2 = this.waterCtx;
    const w1 = this.fftCanvas.width;
    const h1 = this.fftCanvas.height;
    const w2 = this.waterCanvas.width;
    const h2 = this.waterCanvas.height;

    ctx1.fillStyle = "#020617";
    ctx1.fillRect(0,0,w1,h1);
    ctx1.fillStyle = "#4b5563";
    ctx1.font = "12px system-ui";
    ctx1.fillText("No data", 10, 20);

    ctx2.fillStyle = "#020617";
    ctx2.fillRect(0,0,w2,h2);
  }

  drawSpectrum(bins) {
    const ctx = this.fftCtx;
    const w = this.fftCanvas.width;
    const h = this.fftCanvas.height;
    ctx.fillStyle = "#020617";
    ctx.fillRect(0,0,w,h);

    if (!bins || !bins.length) {
      this.drawEmpty();
      return;
    }

    const minVal = Math.min(...bins);
    const maxVal = Math.max(...bins);
    const span = maxVal - minVal || 1;

    ctx.strokeStyle = "#60a5fa";
    ctx.beginPath();
    bins.forEach((v, i) => {
      const x = (i / (bins.length - 1)) * w;
      const norm = (v - minVal) / span;
      const y = h - norm * (h - 10) - 5;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  drawWaterfall(bins) {
    const ctx = this.waterCtx;
    const w = this.waterCanvas.width;
    const h = this.waterCanvas.height;

    if (!bins || !bins.length) return;

    const minVal = Math.min(...bins);
    const maxVal = Math.max(...bins);
    const span = maxVal - minVal || 1;

    // Scroll existing image up by 1 pixel
    ctx.drawImage(this.waterCanvas, 0, -1);

    // Draw new row at bottom
    const img = ctx.getImageData(0, h - 1, w, 1);
    const data = img.data;

    for (let x = 0; x < w; x++) {
      const idx = Math.floor(x / (w - 1) * (bins.length - 1));
      const v = bins[idx];
      const norm = Math.max(0, Math.min(1, (v - minVal) / span));

      const i4 = x * 4;
      const b = Math.floor(255 * norm);
      const g = Math.floor(255 * norm);
      const r = Math.floor(255 * Math.pow(norm, 0.5));

      data[i4 + 0] = r;
      data[i4 + 1] = g;
      data[i4 + 2] = 255 - b;
      data[i4 + 3] = 255;
    }

    ctx.putImageData(img, 0, h - 1);
  }

  destroy() {
    this.closeWS();
    this.root.remove();
    panels.delete(this.id);
  }
}

document.getElementById("add-panel").onclick = () => {
  const id = ++panelIdCounter;
  const panel = new FFTPanel(id);
  panels.set(id, panel);
};

loadSDRs();
setInterval(loadSDRs, 5000);
</script>
</body>
</html>
"""
    return html


if __name__ == "__main__":
    # Flask dev server; works fine for this internal tool and supports Flask-Sock websockets
    app.run(host="0.0.0.0", port=FFT_PORT, threaded=True)
