#!/usr/bin/env python3
import os
import signal
import subprocess
import threading
import time

from flask import Flask, jsonify, request, render_template_string

app = Flask(__name__)

# Global processes + lock
_rtl_proc = None
_aplay_proc = None
_proc_lock = threading.Lock()
_last_start = None

# Current settings / state
state = {
    "running": False,
    "freq": 100.1e6,         # Hz
    "bandwidth": 50e3,      # Hz (used to pick rtl_fm -s)
    "gain": 0,               # 0 = auto
    "samplerate": 48000,     # audio sample rate for aplay
    "audio_device": "plughw:2,0",  # Pi headphones jack (card 2, device 0)
    "volume": 80,            # percent (amixer)
    "device_index": 0,       # which RTL-SDR: 0, 1, ...
}

HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>RTL-FM Controller</title>
<style>
body { font-family: system-ui, sans-serif; padding: 20px; max-width: 700px; }
label { display:block; margin-top:8px; }
input[type="number"], input[type="text"] { width: 220px; padding:5px; }
button { margin-top:10px; padding:6px 10px; }
pre { background:#111; color:#eee; padding:8px; max-height:200px; overflow:auto; }
.status { margin-top: 12px; border: 1px solid #ccc; border-radius:5px; padding:8px; }
.range-row { display:flex; align-items:center; gap:10px; margin-top:8px; }
</style>
</head>
<body>
<h2>RTL-FM Controller</h2>

<label>Frequency (MHz)
  <input id="freq" type="number" step="0.001" value="100.100">
</label>
<label>Bandwidth (kHz)
  <input id="bandwidth" type="number" step="1" value="200">
</label>
<label>Gain (0=auto)
  <input id="gain" type="number" step="1" value="0">
</label>
<label>SDR Index (-d)
  <input id="device_index" type="number" step="1" value="0">
</label>
<label>Audio Device
  <input id="audio_device" type="text" value="plughw:2,0">
</label>

<div class="range-row">
  <label style="margin-top:0;">Volume (%)</label>
  <input id="volume" type="range" min="0" max="100" value="80">
  <span id="volumeVal">80</span>
</div>

<div>
  <button id="start">Start / Retune</button>
  <button id="stop">Stop</button>
  <button id="status">Refresh</button>
</div>

<div class="status" id="statusBox">Loading...</div>
<button id="logs">Logs</button>
<pre id="logText"></pre>

<script>
function j(url, m='GET', b=null){
  return fetch(url, {
    method: m,
    headers: {'Content-Type':'application/json'},
    body: b ? JSON.stringify(b) : null
  }).then(r => r.json());
}

const freqInput   = document.getElementById('freq');
const bwInput     = document.getElementById('bandwidth');
const gainInput   = document.getElementById('gain');
const devIdxInput = document.getElementById('device_index');
const devInput    = document.getElementById('audio_device');
const volInput    = document.getElementById('volume');
const volLabel    = document.getElementById('volumeVal');
const statusBox   = document.getElementById('statusBox');
const logText     = document.getElementById('logText');

async function refresh(){
  try {
    const s = await j('/api/status');
    statusBox.innerHTML =
      `<b>Running:</b> ${s.running}<br>
       <b>Freq:</b> ${(s.freq/1e6).toFixed(3)} MHz<br>
       <b>Bandwidth:</b> ${(s.bandwidth/1e3)} kHz<br>
       <b>Gain:</b> ${s.gain}<br>
       <b>SDR index (-d):</b> ${s.device_index}<br>
       <b>Audio:</b> ${s.audio_device}<br>
       <b>Volume:</b> ${s.volume}%<br>
       <b>Last start:</b> ${s.last_start ? new Date(s.last_start*1000).toLocaleString() : '-'}`;

    if (document.activeElement !== freqInput)
      freqInput.value = (s.freq/1e6).toFixed(3);
    if (document.activeElement !== bwInput)
      bwInput.value = (s.bandwidth/1e3);
    if (document.activeElement !== gainInput)
      gainInput.value = s.gain;
    if (document.activeElement !== devIdxInput)
      devIdxInput.value = s.device_index;
    if (document.activeElement !== devInput)
      devInput.value = s.audio_device;
    if (document.activeElement !== volInput) {
      volInput.value = s.volume;
      volLabel.textContent = s.volume;
    }
  } catch (e) {
    statusBox.innerHTML = "Error fetching status: " + e;
  }
}

document.getElementById('start').onclick = async () => {
  const payload = {
    freq: parseFloat(freqInput.value) * 1e6,
    bandwidth: parseFloat(bwInput.value) * 1e3,
    gain: parseInt(gainInput.value),
    device_index: parseInt(devIdxInput.value),
    audio_device: devInput.value
  };
  try {
    const r = await j('/api/start', 'POST', payload);
    if (!r.ok) {
      alert('Start failed: ' + JSON.stringify(r));
    }
  } catch (e) {
    alert('Start error: ' + e);
  }
  await refresh();
};

document.getElementById('stop').onclick = async () => {
  try {
    await j('/api/stop', 'POST');
  } catch (e) {
    alert('Stop error: ' + e);
  }
  await refresh();
};

document.getElementById('status').onclick = refresh;

document.getElementById('logs').onclick = async () => {
  try {
    const r = await j('/api/logs');
    logText.textContent = JSON.stringify(r, null, 2);
  } catch (e) {
    logText.textContent = 'Error fetching logs: ' + e;
  }
};

volInput.oninput = async () => {
  volLabel.textContent = volInput.value;
  try {
    await j('/api/volume', 'POST', { volume: parseInt(volInput.value) });
  } catch (e) {
    console.warn('Volume error', e);
  }
};

refresh();
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/status")
def api_status():
    with _proc_lock:
        running = (
            _rtl_proc is not None and _rtl_proc.poll() is None and
            _aplay_proc is not None and _aplay_proc.poll() is None
        )
    s = dict(state)
    s["running"] = running
    s["pid_rtl"] = _rtl_proc.pid if _rtl_proc is not None else None
    s["pid_aplay"] = _aplay_proc.pid if _aplay_proc is not None else None
    s["last_start"] = _last_start
    return jsonify(s)


def set_volume(percent: int):
    """Use amixer to set Pi volume on card 2."""
    vol = max(0, min(100, int(percent)))
    state["volume"] = vol
    controls = ["Headphones", "PCM", "Master"]
    for ctl in controls:
        try:
            r = subprocess.run(
                ["amixer", "-c", "2", "sset", ctl, f"{vol}%"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if r.returncode == 0:
                break
        except Exception:
            continue


def build_rtl_cmd():
    """Build rtl_fm command as a list for subprocess (no shell)."""
    freq_hz = int(state["freq"])
    bw_hz = int(state["bandwidth"]) if state["bandwidth"] else 200000
    gain = int(state["gain"])

    rtl_s = max(24000, min(192000, bw_hz))
    cmd = [
        "rtl_fm",
        "-d", str(int(state["device_index"])),
        "-f", str(freq_hz),
        "-M", "fm",
        "-s", str(rtl_s),
        "-E", "deemp",
    ]
    if gain:
        cmd += ["-g", str(gain)]
    else:
        cmd += ["-g", "0"]
    return cmd


def build_aplay_cmd():
    """Build aplay command as a list for subprocess (no shell)."""
    return [
        "aplay",
        "-f", "S16_LE",
        "-r", str(int(state["samplerate"])),
        "-c", "1",
        "--device", state["audio_device"],
        "-",
    ]


def _stop_current_locked():
    """Stop current rtl_fm + aplay; caller must hold _proc_lock."""
    global _rtl_proc, _aplay_proc
    # Stop rtl_fm gently first
    if _rtl_proc is not None and _rtl_proc.poll() is None:
        try:
            _rtl_proc.send_signal(signal.SIGINT)
        except Exception:
            try:
                _rtl_proc.terminate()
            except Exception:
                pass
        try:
            _rtl_proc.wait(timeout=3)
        except Exception:
            try:
                _rtl_proc.kill()
            except Exception:
                pass
    _rtl_proc = None

    # Then stop aplay
    if _aplay_proc is not None and _aplay_proc.poll() is None:
        try:
            _aplay_proc.terminate()
        except Exception:
            pass
        try:
            _aplay_proc.wait(timeout=3)
        except Exception:
            try:
                _aplay_proc.kill()
            except Exception:
                pass
    _aplay_proc = None

    state["running"] = False


@app.route("/api/start", methods=["POST"])
def api_start():
    global _rtl_proc, _aplay_proc, _last_start

    data = request.get_json() or {}

    # Update state from payload
    for key in ("freq", "bandwidth", "gain", "samplerate", "audio_device", "device_index"):
        if key in data:
            if isinstance(state[key], float):
                state[key] = float(data[key])
            elif isinstance(state[key], int):
                state[key] = int(data[key])
            else:
                state[key] = data[key]

    with _proc_lock:
        # Stop any existing chain (for retune)
        _stop_current_locked()

        # Volume
        set_volume(state.get("volume", 80))

        rtl_cmd = build_rtl_cmd()
        aplay_cmd = build_aplay_cmd()

        # Start rtl_fm first
        try:
            _rtl_proc = subprocess.Popen(
                rtl_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except Exception as e:
            _rtl_proc = None
            return jsonify({"ok": False, "error": f"failed to start rtl_fm: {e}"}), 500

        # Then start aplay, consuming rtl_fm's stdout
        try:
            _aplay_proc = subprocess.Popen(
                aplay_cmd,
                stdin=_rtl_proc.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except Exception as e:
            # If aplay fails, stop rtl_fm too
            try:
                _rtl_proc.send_signal(signal.SIGINT)
            except Exception:
                try:
                    _rtl_proc.terminate()
                except Exception:
                    pass
            _rtl_proc = None
            _aplay_proc = None
            return jsonify({"ok": False, "error": f"failed to start aplay: {e}"}), 500

        _last_start = time.time()
        state["running"] = True

    return jsonify({"ok": True, "rtl_cmd": rtl_cmd, "aplay_cmd": aplay_cmd,
                    "pid_rtl": _rtl_proc.pid, "pid_aplay": _aplay_proc.pid})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    global _rtl_proc, _aplay_proc
    with _proc_lock:
        if (_rtl_proc is None or _rtl_proc.poll() is not None) and \
           (_aplay_proc is None or _aplay_proc.poll() is not None):
            _rtl_proc = None
            _aplay_proc = None
            state["running"] = False
            return jsonify({"ok": True, "stopped": False, "reason": "not running"})
        _stop_current_locked()
    return jsonify({"ok": True, "stopped": True})


@app.route("/api/logs")
def api_logs():
    # Return some stderr from both processes
    with _proc_lock:
        if _rtl_proc is None and _aplay_proc is None:
            return jsonify({"ok": False, "reason": "not running"}), 400
        try:
            rtl_err = b""
            aplay_err = b""
            if _rtl_proc is not None and _rtl_proc.stderr is not None:
                rtl_err = _rtl_proc.stderr.read(4096)
            if _aplay_proc is not None and _aplay_proc.stderr is not None:
                aplay_err = _aplay_proc.stderr.read(4096)
            return jsonify({
                "ok": True,
                "rtl_stderr": rtl_err.decode(errors="ignore"),
                "aplay_stderr": aplay_err.decode(errors="ignore"),
            })
        except Exception as e:
            return jsonify({"ok": False, "reason": str(e)}), 500


@app.route("/api/volume", methods=["POST"])
def api_volume():
    data = request.get_json() or {}
    vol = int(data.get("volume", state.get("volume", 80)))
    vol = max(0, min(100, vol))
    set_volume(vol)
    return jsonify({"ok": True, "volume": vol})


if __name__ == "__main__":
    print("Starting RTL-FM control server on :8080", flush=True)
    set_volume(state["volume"])
    app.run(host="0.0.0.0", port=8080)
