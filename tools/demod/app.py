#!/usr/bin/env python3
import os
import json
import socket
from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np
import requests
from flask import Flask, jsonify, request
from flask_sock import Sock

HUB_URL = os.getenv("HUB_URL", "http://hub:8080")
HUB_HOST = os.getenv("HUB_HOST", "hub")
DEMOD_PORT = int(os.getenv("DEMOD_PORT", "8100"))

# We used to "aim" for this, now we do a two-stage plan inside ws_audio.
AUDIO_RATE = 51200.0  # Hz (kept for reference / future use)

app = Flask(__name__)
sock = Sock(app)

# ------------- Helpers to talk to hub -------------


def get_sdrs():
    r = requests.get(f"{HUB_URL}/api/sdrs", timeout=5)
    r.raise_for_status()
    return r.json()


def find_sdr(serial: str):
    for s in get_sdrs():
        if s.get("serial") == serial:
            return s
    return None


# ------------- Demod state (in-memory) -------------


@dataclass
class DemodState:
    enabled: bool = False
    mode: str = "off"   # "off", "fm", etc.
    bw_hz: int = 25_000  # default to 25 kHz


demod_states: Dict[str, DemodState] = {}


def ensure_state_for_existing_sdrs():
    """
    Make sure demod_states has an entry for every currently-known SDR serial.
    """
    try:
        sdrs = get_sdrs()
    except Exception as e:
        print(f"[demod] ensure_state_for_existing_sdrs error: {e}", flush=True)
        return

    for s in sdrs:
        serial = s.get("serial")
        if not serial:
            continue
        if serial not in demod_states:
            demod_states[serial] = DemodState()


def get_state_dict():
    """
    Return demod_states as a plain dict {serial: {...}}.
    """
    ensure_state_for_existing_sdrs()
    return {serial: asdict(st) for serial, st in demod_states.items()}


# ------------- API endpoints (config / status) -------------


@app.get("/api/sdrs")
def api_sdrs():
    """
    Proxy SDR list from hub, same as fft-viewer.
    """
    try:
        data = get_sdrs()
        return jsonify(data)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.get("/api/demod")
def api_demod_all():
    """
    Return all demod states keyed by SDR serial.
    """
    return jsonify(get_state_dict())


@app.post("/api/demod/<serial>/config")
def api_demod_config(serial):
    """
    Update demod configuration for a single SDR.
    Body: { enabled: bool, mode: "off"|"fm"|..., bw_hz: int }
    """
    ensure_state_for_existing_sdrs()
    if serial not in demod_states:
        demod_states[serial] = DemodState()

    state = demod_states[serial]

    data = request.get_json(force=True) or {}
    enabled = bool(data.get("enabled", False))
    mode = data.get("mode") or ("fm" if enabled else "off")
    bw = int(data.get("bw_hz", state.bw_hz))

    # Normalize: if not enabled, force "off"
    if not enabled:
        mode = "off"

    state.enabled = enabled
    state.mode = mode
    state.bw_hz = bw

    print(f"[demod] config update serial={serial} enabled={enabled} mode={mode} bw={bw}", flush=True)

    return jsonify({"ok": True, "state": asdict(state)})


# ------------- WebSocket audio: /ws/audio/<serial> -------------


@sock.route("/ws/audio/<serial>")
def ws_audio(ws, serial):
    """
    Per-connection FM demod from SDR IQ to float32 mono audio frames.

    - Connects to hub IQ relay for this SDR.
    - Uses SDR's current sample_rate (from hub) to choose decimation.
    - Two-stage decimation:
        RF -> ~120 kHz baseband
        baseband -> ~24 kHz audio
    - FM discriminator at the baseband rate.
    - Sends ONLY binary frames of float32 samples (no JSON metadata).
    """
    print(f"[demod] WebSocket audio connected for serial={serial}", flush=True)

    dev = find_sdr(serial)
    if not dev:
        print(f"[demod] ws_audio: SDR not found for serial={serial}", flush=True)
        return

    iq_port = dev.get("iq_port")
    if not iq_port:
        print(f"[demod] ws_audio: missing iq_port for serial={serial}", flush=True)
        return

    # RF sample rate from hub
    rf_rate = dev.get("sample_rate") or 2_048_000
    try:
        rf_rate = int(rf_rate)
    except Exception:
        rf_rate = 2_048_000

    # --- Decimation plan ---
    # 1) RF -> baseband ~120 kHz
    target_base = 120_000.0
    decim1 = max(1, int(round(rf_rate / target_base)))
    base_rate = rf_rate / decim1

    # 2) base -> audio ~24 kHz (fine for voice)
    target_audio = 24_000.0
    decim2 = max(1, int(round(base_rate / target_audio)))
    eff_audio_rate = base_rate / decim2

    # Demod BW + simple deviation heuristic
    ensure_state_for_existing_sdrs()
    st = demod_states.get(serial, DemodState())
    bw_hz = max(1, int(st.bw_hz or 25_000))
    max_dev = max(1.0, bw_hz / 2.0)

    # Simple scaling constant; tuned around 25 kHz voice FM
    fm_scale = float(0.3 * (25_000.0 / max_dev))

    print(
        f"[demod] ws_audio RF={rf_rate}Hz decim1={decim1} base≈{base_rate:.1f}Hz "
        f"decim2={decim2} audio≈{eff_audio_rate:.1f}Hz bw={bw_hz} dev≈{max_dev:.1f}",
        flush=True,
    )

    # Connect to IQ stream from hub
    try:
        sock_iq = socket.create_connection((HUB_HOST, iq_port), timeout=5)
        sock_iq.settimeout(2.0)
        print(
            f"[demod] ws_audio: connected to IQ relay for serial={serial} port={iq_port}",
            flush=True,
        )
    except Exception as e:
        print(f"[demod] ws_audio error serial={serial}: iq_connect_failed: {e}", flush=True)
        return

    prev_rf_sample = None     # last complex at RF rate
    prev_base_sample = None   # last complex at base_rate
    lp_state = 0.0            # audio low-pass state

    try:
        while True:
            try:
                data = sock_iq.recv(65536)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[demod] ws_audio recv error serial={serial}: {e}", flush=True)
                break

            if not data:
                print(f"[demod] ws_audio: IQ socket closed serial={serial}", flush=True)
                break

            # rtl_tcp: unsigned 8-bit interleaved I/Q
            arr = np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 127.5
            if arr.size < 4:
                continue

            i = arr[0::2]
            q = arr[1::2]
            x_rf = i + 1j * q

            # continuity at RF
            if prev_rf_sample is not None:
                x_rf = np.concatenate([prev_rf_sample, x_rf])
            prev_rf_sample = x_rf[-1:].copy()

            # --- decimate RF -> base_rate ---
            x_base = x_rf[::decim1]
            if x_base.size < 2:
                continue

            # continuity at base_rate for FM discriminator
            if prev_base_sample is not None:
                x_base = np.concatenate([prev_base_sample, x_base])
            prev_base_sample = x_base[-1:].copy()

            # --- FM discriminator at base_rate ---
            z = x_base[1:] * np.conj(x_base[:-1])
            fm = np.angle(z).astype(np.float32) * fm_scale

            # --- decimate base_rate -> audio ---
            audio = fm[::decim2]
            if audio.size == 0:
                continue

            # --- audio conditioning ---

            # 1) remove DC
            audio -= audio.mean()

            # 2) simple low-pass ~3 kHz at eff_audio_rate
            fc = 3000.0
            alpha = (2.0 * np.pi * fc) / (2.0 * np.pi * fc + eff_audio_rate)

            y = np.empty_like(audio)
            s = lp_state
            for idx, v in enumerate(audio):
                s = s + alpha * (v - s)
                y[idx] = s
            lp_state = s

            # 3) gentle overall gain
            y *= 0.8

            try:
                ws.send(y.astype(np.float32).tobytes())
            except Exception as e:
                print(f"[demod] ws_audio send error serial={serial}: {e}", flush=True)
                break

    finally:
        try:
            sock_iq.close()
        except Exception:
            pass
        print(f"[demod] WebSocket audio closed for serial={serial}", flush=True)


# ------------- Simple Web UI -------------


@app.route("/")
def index():
    # NOTE: plain triple-quoted string, NOT an f-string, to avoid brace issues.
    html = """
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>RECON-KIT Demodulator</title>
<style>
body {
  font-family: system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  margin: 0;
  padding: 16px;
  background: #020617;
  color: #e5e7eb;
}
h1 { margin: 0 0 8px 0; }
.subtitle {
  font-size: 12px;
  color: #9ca3af;
  margin-bottom: 16px;
}
#sdr-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit,minmax(260px,1fr));
  gap: 10px;
}
.demod-card {
  background: #020617;
  border: 1px solid #111827;
  border-radius: 10px;
  padding: 8px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.demod-card h2 {
  font-size: 14px;
  margin: 0;
}
.small-label {
  font-size: 11px;
  color: #9ca3af;
}
button {
  padding: 4px 10px;
  border-radius: 999px;
  border: none;
  cursor: pointer;
  font-size: 12px;
  background: #1f2937;
  color: #e5e7eb;
}
button:hover { background: #4b5563; }
select, input[type=number] {
  background: #020617;
  border: 1px solid #1f2937;
  border-radius: 999px;
  padding: 2px 6px;
  color: #e5e7eb;
  font-size: 12px;
}
</style>
</head>
<body>
  <h1>RECON-KIT Demodulator</h1>
  <div class="subtitle">Configure basic demodulation per SDR (FM only for now) and listen to audio.</div>
  <div id="sdr-cards"></div>

<script>
console.log("demod UI script loaded");

let sdrList = [];
let demodStatus = {}; // serial -> { enabled, mode, bw_hz }

// audio per SDR
const audioState = {}; // serial -> { ws, ctx, playHead }

async function loadSDRs() {
  console.log("loadSDRs() called");
  try {
    const res = await fetch("/api/sdrs");
    if (!res.ok) {
      console.error("loadSDRs non-OK", res.status);
      return;
    }
    sdrList = await res.json();
    console.log("loadSDRs got", sdrList);
    renderSDRs();
  } catch (e) {
    console.error("loadSDRs error", e);
  }
}

async function loadDemodStatus() {
  console.log("loadDemodStatus() called");
  try {
    const res = await fetch("/api/demod");
    if (!res.ok) {
      console.error("loadDemodStatus non-OK", res.status);
      return;
    }
    const data = await res.json();
    console.log("loadDemodStatus got", data);
    demodStatus = data || {};
    renderSDRs();
  } catch (e) {
    console.error("loadDemodStatus error", e);
  }
}

// ===== Audio via WebSocket (binary float32) =====

function stopAudio(serial) {
  const st = audioState[serial];
  if (!st) return;
  console.log("stopAudio", serial);
  try {
    if (st.ws && st.ws.readyState === WebSocket.OPEN) {
      st.ws.close();
    }
  } catch (e) {}
  try {
    if (st.ctx && st.ctx.state !== "closed") {
      st.ctx.close();
    }
  } catch (e) {}
  audioState[serial] = null;
}

function startAudio(serial) {
  const existing = audioState[serial];
  if (existing?.ws && existing.ws.readyState === WebSocket.OPEN) {
    console.log("startAudio: already running for", serial);
    return;
  }

  const proto = (location.protocol === "https:") ? "wss://" : "ws://";
  const url = proto + location.host + "/ws/audio/" + encodeURIComponent(serial);
  console.log("startAudio: connecting", url);

  const ws = new WebSocket(url);
  ws.binaryType = "arraybuffer";

  const st = {
    ws,
    ctx: null,
    playHead: 0,
  };
  audioState[serial] = st;

  ws.onopen = async () => {
    console.log("audio WS open", serial);
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    // Let browser pick its preferred rate; we'll just schedule by time.
    st.ctx = new AudioCtx();
    try {
      await st.ctx.resume();
    } catch (e) {
      console.warn("audio ctx resume failed", e);
    }
    st.playHead = st.ctx.currentTime + 0.3; // a bit more initial buffer
  };

  ws.onmessage = (event) => {
    const data = event.data;

    // We expect ONLY binary frames with float32 audio
    if (!(data instanceof ArrayBuffer)) {
      console.warn("non-binary audio frame for", serial, data);
      return;
    }
    if (!st.ctx) return;

    const samples = new Float32Array(data);
    if (!samples.length) return;

    const sr = st.ctx.sampleRate;
    const audioBuffer = st.ctx.createBuffer(1, samples.length, sr);
    audioBuffer.getChannelData(0).set(samples);
    const src = st.ctx.createBufferSource();
    src.buffer = audioBuffer;
    src.connect(st.ctx.destination);

    const startTime = Math.max(st.ctx.currentTime, st.playHead);
    src.start(startTime);
    st.playHead = startTime + audioBuffer.duration;
  };

  ws.onclose = () => {
    console.log("audio WS close", serial);
    audioState[serial] = null;
  };

  ws.onerror = (e) => {
    console.error("audio WS error", serial, e);
  };
}


function renderSDRs() {
  const container = document.getElementById("sdr-cards");
  if (!container) {
    console.error("#sdr-cards not found in DOM");
    return;
  }

  console.log("renderSDRs: sdrList length", sdrList.length);

  // Remember current selections before we rebuild the DOM
  const existingModes = {};
  const existingBW = {};

  container.querySelectorAll(".demod-card").forEach(card => {
    const serial = card.dataset.serial;
    if (!serial) return;
    const modeSel = card.querySelector("select[data-role='mode']");
    const bwInput = card.querySelector("input[data-role='bw']");
    if (modeSel) existingModes[serial] = modeSel.value;
    if (bwInput) existingBW[serial] = bwInput.value;
  });

  container.innerHTML = "";

  if (!sdrList || sdrList.length === 0) {
    const msg = document.createElement("div");
    msg.textContent = "No SDRs detected.";
    msg.className = "small-label";
    container.appendChild(msg);
    return;
  }

  sdrList.forEach(sdr => {
    const serial = sdr.serial;
    const status = demodStatus[serial] || {};
    const card = document.createElement("div");
    card.className = "demod-card";
    card.dataset.serial = serial;

    const title = document.createElement("h2");
    title.textContent = `SDR ${serial} (IQ:${sdr.iq_port})`;

    const statusText = document.createElement("div");
    statusText.className = "small-label";
    if (status.enabled && status.mode === "fm") {
      statusText.textContent = `FM demod ON, BW ${status.bw_hz || 25000} Hz`;
    } else {
      statusText.textContent = "Demod OFF";
    }

    const controls = document.createElement("div");
    controls.style.display = "flex";
    controls.style.gap = "6px";
    controls.style.alignItems = "center";
    controls.style.flexWrap = "wrap";

    const labelMode = document.createElement("span");
    labelMode.textContent = "Mode:";
    labelMode.className = "small-label";

    const modeSel = document.createElement("select");
    modeSel.dataset.role = "mode";
    const optOff = document.createElement("option");
    optOff.value = "";
    optOff.textContent = "Off";
    const optFM = document.createElement("option");
    optFM.value = "fm";
    optFM.textContent = "FM";
    modeSel.appendChild(optOff);
    modeSel.appendChild(optFM);

    let uiMode = existingModes[serial] ?? "";
    if (status.enabled && status.mode === "fm") {
      uiMode = "fm";
    }
    modeSel.value = uiMode;

    const labelBw = document.createElement("span");
    labelBw.textContent = "BW (Hz):";
    labelBw.className = "small-label";

    const bwInput = document.createElement("input");
    bwInput.type = "number";
    bwInput.min = "5000";
    bwInput.max = "300000";
    bwInput.step = "5000";
    bwInput.style.width = "90px";
    bwInput.dataset.role = "bw";
    const defaultBw = status.bw_hz || 25000;
    bwInput.value = existingBW[serial] ?? defaultBw;

    const applyBtn = document.createElement("button");
    applyBtn.textContent = "Apply";
    applyBtn.onclick = async () => {
      const mode = modeSel.value;
      const enabled = mode === "fm";
      const bw = parseInt(bwInput.value) || 25000;
      console.log("apply demod", serial, {enabled, mode, bw});

      try {
        const res = await fetch(`/api/demod/${encodeURIComponent(serial)}/config`, {
          method: "POST",
          headers: {"Content-Type":"application/json"},
          body: JSON.stringify({enabled, mode, bw_hz: bw}),
        });
        const data = await res.json();
        console.log("apply demod resp", data);
        if (!res.ok || !data.ok) {
          statusText.textContent = "Error applying config";
        } else {
          statusText.textContent = enabled
            ? `FM demod ON, BW ${bw} Hz`
            : "Demod OFF";
          // Refresh state from backend once after apply
          loadDemodStatus();
        }
      } catch (e) {
        console.error("apply demod error", e);
        statusText.textContent = "Error applying config";
      }
    };

    const audioBtn = document.createElement("button");
    const updateAudioButtonLabel = () => {
      const st = audioState[serial];
      const on = !!(st && st.ws && st.ws.readyState === WebSocket.OPEN);
      audioBtn.textContent = on ? "Audio: On" : "Audio: Off";
    };
    audioBtn.onclick = () => {
      const st = audioState[serial];
      const on = !!(st && st.ws && st.ws.readyState === WebSocket.OPEN);
      if (on) {
        stopAudio(serial);
      } else {
        startAudio(serial);
      }
      setTimeout(updateAudioButtonLabel, 200);
    };
    updateAudioButtonLabel();

    controls.appendChild(labelMode);
    controls.appendChild(modeSel);
    controls.appendChild(labelBw);
    controls.appendChild(bwInput);
    controls.appendChild(applyBtn);
    controls.appendChild(audioBtn);

    card.appendChild(title);
    card.appendChild(statusText);
    card.appendChild(controls);
    container.appendChild(card);
  });
}

// Kick off initial loads once DOM is ready
window.addEventListener("load", () => {
  console.log("demod UI window load");
  loadSDRs();
  loadDemodStatus();
  // Only poll SDR list periodically; demod status only on apply / initial
  setInterval(loadSDRs, 5000);
});
</script>
</body>
</html>
"""
    return html


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=DEMOD_PORT, threaded=True)
