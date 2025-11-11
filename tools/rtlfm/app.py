#!/usr/bin/env python3
import signal
import subprocess
import threading
import time
import math
import struct
import json
import os

from flask import Flask, jsonify, request, render_template_string

app = Flask(__name__)

# Try to import Vosk for STT (optional)
try:
    from vosk import Model, KaldiRecognizer
    _vosk_available = True
except Exception:
    Model = None
    KaldiRecognizer = None
    _vosk_available = False

# Global processes + lock
_rtl_proc = None
_aplay_proc = None
_pipe_thread = None
_proc_lock = threading.Lock()
_last_start = None

# STT state
_stt_model = None
_stt_recognizer = None
_transcripts = []  # list of {time, offset, freq_hz, text, final?}

# Current settings / state
state = {
    "running": False,
    "freq": 453.45e6,        # Hz, default
    "bandwidth": 50e3,       # Hz, good for ~25 kHz FM channels
    "gain": 0,               # 0 = auto
    "samplerate": 48000,     # audio sample rate for aplay
    "audio_device": "plughw:2,0",  # Pi headphones jack (card 2, device 0)
    "volume": 80,            # percent (amixer)
    "device_index": 0,       # which RTL-SDR: 0, 1, ...
    "squelch": 20,           # rtl_fm -l value (0=off)
    # Modulation (-M)
    "modulation": "fm",      # fm, wbfm, raw, am, usb, lsb
    # Beep / detection config
    "open_threshold": 2000,  # audio level needed to consider "open"
    "close_threshold": 250,  # level below which we consider "closed" again
    "beep_freq": 1000,       # Hz, tone for this SDR
    # rtl_fm built-in scan config
    "scan_enabled": False,
    "scan_start": 453.0e6,   # Hz
    "scan_end": 454.0e6,     # Hz
    "scan_step": 25e3,       # Hz
    # STT (Vosk)
    "stt_enabled": False,
    "stt_model_path": "/opt/models/vosk-model-small-en-us-0.15",
    # Debug: last commands
    "last_rtl_cmd": [],
    "last_aplay_cmd": [],
}


def make_beep(sr=48000, freq=1000, duration=0.06, volume=0.4):
    """
    Generate a short S16_LE mono beep.
    """
    n_samples = int(sr * duration)
    frames = []
    for n in range(n_samples):
        sample = int(volume * 32767 * math.sin(2 * math.pi * freq * n / sr))
        frames.append(struct.pack("<h", sample))
    return b"".join(frames)


HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>RTL-FM Controller</title>
<style>
body { font-family: system-ui, sans-serif; padding: 20px; max-width: 1200px; }
label { display:block; margin-top:8px; }
input[type="number"], input[type="text"], select { width: 220px; padding:5px; }
button { margin-top:10px; padding:6px 10px; }
pre { background:#111; color:#eee; padding:8px; max-height:200px; overflow:auto; }
.status { margin-top: 12px; border: 1px solid #ccc; border-radius:5px; padding:8px; }
.range-row { display:flex; align-items:center; gap:10px; margin-top:8px; }
.inline { display:flex; gap:16px; flex-wrap:wrap; }
.inline > div { min-width: 260px; }
small { color:#555; }
h3 { margin-bottom:4px; }
</style>
</head>
<body>
<h2>RTL-FM Controller + Optional STT</h2>

<div class="inline">
  <div>
    <h3>Receiver</h3>
    <label>Frequency (MHz)
      <input id="freq" type="number" step="0.001" value="453.450">
    </label>
    <label>Bandwidth (kHz)
      <input id="bandwidth" type="number" step="1" value="50">
    </label>
    <label>Modulation (-M)
      <select id="modulation">
        <option value="fm">fm (NBFM)</option>
        <option value="wbfm">wbfm (WBFM)</option>
        <option value="raw">raw (IQ / raw)</option>
        <option value="am">am</option>
        <option value="usb">usb</option>
        <option value="lsb">lsb</option>
      </select>
    </label>
    <label>Gain (0=auto)
      <input id="gain" type="number" step="1" value="0">
    </label>
    <label>SDR Index (-d)
      <input id="device_index" type="number" step="1" value="0">
    </label>
    <label>Squelch (-l, 0 = off)
      <input id="squelch" type="number" step="1" value="20">
    </label>
    <label>Audio Device
      <input id="audio_device" type="text" value="plughw:2,0">
    </label>
  </div>

  <div>
    <h3>Audio / Beep</h3>
    <div class="range-row">
      <label style="margin-top:0;">Volume (%)</label>
      <input id="volume" type="range" min="0" max="100" value="80">
      <span id="volumeVal">80</span>
    </div>
    <label>Beep frequency (Hz)
      <input id="beep_freq" type="number" step="50" value="1000">
    </label>
    <small>Use different beep frequencies on each SDR (e.g. 800 Hz vs 1500 Hz)</small>
    <label>Open threshold (audio level)
      <input id="open_threshold" type="number" step="100" value="2000">
    </label>
    <label>Close threshold (audio level)
      <input id="close_threshold" type="number" step="50" value="250">
    </label>
    <small>Set close threshold &lt; open threshold to avoid chattering.</small>
  </div>

  <div>
    <h3>Scan (rtl_fm built-in)</h3>
    <label>Scan start (MHz)
      <input id="scan_start" type="number" step="0.001" value="453.000">
    </label>
    <label>Scan end (MHz)
      <input id="scan_end" type="number" step="0.001" value="454.000">
    </label>
    <label>Scan step (kHz)
      <input id="scan_step" type="number" step="1" value="25">
    </label>
    <label>
      <input id="scan_enabled" type="checkbox">
      Enable scan (rtl_fm -f start:end:step)
    </label>
    <small>Changes take effect when you click Start / Retune.</small>
  </div>

  <div>
    <h3>STT (Vosk)</h3>
    <label>
      <input id="stt_enabled" type="checkbox">
      Enable STT
    </label>
    <label>Model path
      <input id="stt_model_path" type="text" value="/opt/models/vosk-model-small-en-us-0.15">
    </label>
    <small id="sttStatusNote"></small>
    <div style="margin-top:8px;">
      <button id="refresh_transcripts">Refresh transcripts</button>
      <button id="clear_transcripts">Clear transcripts</button>
    </div>
  </div>
</div>

<div>
  <button id="start">Start / Retune</button>
  <button id="stop">Stop</button>
  <button id="status">Refresh</button>
</div>

<div class="status" id="statusBox">Loading...</div>

<h3>Transcripts</h3>
<pre id="transcriptsBox"></pre>

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

const freqInput        = document.getElementById('freq');
const bwInput          = document.getElementById('bandwidth');
const modInput         = document.getElementById('modulation');
const gainInput        = document.getElementById('gain');
const devIdxInput      = document.getElementById('device_index');
const sqlInput         = document.getElementById('squelch');
const devInput         = document.getElementById('audio_device');
const volInput         = document.getElementById('volume');
const volLabel         = document.getElementById('volumeVal');
const statusBox        = document.getElementById('statusBox');
const logText          = document.getElementById('logText');
const beepFreqInput    = document.getElementById('beep_freq');
const openThreshInput  = document.getElementById('open_threshold');
const closeThreshInput = document.getElementById('close_threshold');

const scanStartInput   = document.getElementById('scan_start');
const scanEndInput     = document.getElementById('scan_end');
const scanStepInput    = document.getElementById('scan_step');
const scanEnabledInput = document.getElementById('scan_enabled');

const sttEnabledInput  = document.getElementById('stt_enabled');
const sttModelInput    = document.getElementById('stt_model_path');
const sttStatusNote    = document.getElementById('sttStatusNote');
const transcriptsBox   = document.getElementById('transcriptsBox');

async function refresh(){
  try {
    const s = await j('/api/status');
    statusBox.innerHTML =
      `<b>Running:</b> ${s.running}<br>
       <b>Freq (display):</b> ${(s.freq/1e6).toFixed(3)} MHz<br>
       <b>Bandwidth:</b> ${(s.bandwidth/1e3)} kHz<br>
       <b>Modulation (-M):</b> ${s.modulation}<br>
       <b>Gain:</b> ${s.gain}<br>
       <b>Squelch (-l):</b> ${s.squelch}<br>
       <b>SDR index (-d):</b> ${s.device_index}<br>
       <b>Audio:</b> ${s.audio_device}<br>
       <b>Volume:</b> ${s.volume}%<br>
       <b>Beep freq:</b> ${s.beep_freq} Hz<br>
       <b>Open threshold:</b> ${s.open_threshold}<br>
       <b>Close threshold:</b> ${s.close_threshold}<br>
       <b>Scan enabled:</b> ${s.scan_enabled}<br>
       <b>Scan start:</b> ${(s.scan_start/1e6).toFixed(3)} MHz<br>
       <b>Scan end:</b> ${(s.scan_end/1e6).toFixed(3)} MHz<br>
       <b>Scan step:</b> ${(s.scan_step/1e3).toFixed(1)} kHz<br>
       <b>STT enabled:</b> ${s.stt_enabled}<br>
       <b>Vosk available:</b> ${s.vosk_available}<br>
       <b>Transcripts stored:</b> ${s.transcript_count}<br>
       <b>PID rtl_fm:</b> ${s.pid_rtl || '-'}<br>
       <b>PID aplay:</b> ${s.pid_aplay || '-'}<br>
       <b>Last rtl_fm cmd:</b> ${s.last_rtl_cmd ? s.last_rtl_cmd.join(' ') : '-'}<br>
       <b>Last aplay cmd:</b> ${s.last_aplay_cmd ? s.last_aplay_cmd.join(' ') : '-'}<br>
       <b>Last start:</b> ${s.last_start ? new Date(s.last_start*1000).toLocaleString() : '-'}`;

    if (document.activeElement !== freqInput)
      freqInput.value = (s.freq/1e6).toFixed(3);
    if (document.activeElement !== bwInput)
      bwInput.value = (s.bandwidth/1e3);
    if (document.activeElement !== modInput)
      modInput.value = s.modulation || 'fm';
    if (document.activeElement !== gainInput)
      gainInput.value = s.gain;
    if (document.activeElement !== devIdxInput)
      devIdxInput.value = s.device_index;
    if (document.activeElement !== sqlInput)
      sqlInput.value = s.squelch;
    if (document.activeElement !== devInput)
      devInput.value = s.audio_device;
    if (document.activeElement !== volInput) {
      volInput.value = s.volume;
      volLabel.textContent = s.volume;
    }
    if (document.activeElement !== beepFreqInput)
      beepFreqInput.value = s.beep_freq;
    if (document.activeElement !== openThreshInput)
      openThreshInput.value = s.open_threshold;
    if (document.activeElement !== closeThreshInput)
      closeThreshInput.value = s.close_threshold;

    if (document.activeElement !== scanStartInput)
      scanStartInput.value = (s.scan_start/1e6).toFixed(3);
    if (document.activeElement !== scanEndInput)
      scanEndInput.value = (s.scan_end/1e6).toFixed(3);
    if (document.activeElement !== scanStepInput)
      scanStepInput.value = (s.scan_step/1e3).toFixed(1);
    scanEnabledInput.checked = s.scan_enabled;

    sttEnabledInput.checked = s.stt_enabled;
    if (document.activeElement !== sttModelInput)
      sttModelInput.value = s.stt_model_path || "";
    if (!s.vosk_available) {
      sttStatusNote.textContent = "Vosk not installed in this container.";
    } else {
      sttStatusNote.textContent = s.stt_enabled
        ? "STT enabled (model path must be valid)."
        : "STT available but disabled.";
    }
  } catch (e) {
    statusBox.innerHTML = "Error fetching status: " + e;
  }
}

document.getElementById('start').onclick = async () => {
  const payload = {
    freq:        parseFloat(freqInput.value) * 1e6,
    bandwidth:   parseFloat(bwInput.value) * 1e3,
    modulation:  modInput.value,
    gain:        parseInt(gainInput.value),
    device_index:parseInt(devIdxInput.value),
    squelch:     parseInt(sqlInput.value),
    audio_device:devInput.value,
    beep_freq:   parseInt(beepFreqInput.value),
    open_threshold: parseInt(openThreshInput.value),
    close_threshold:parseInt(closeThreshInput.value),

    scan_enabled: scanEnabledInput.checked,
    scan_start:   parseFloat(scanStartInput.value) * 1e6,
    scan_end:     parseFloat(scanEndInput.value) * 1e6,
    scan_step:    parseFloat(scanStepInput.value) * 1e3,

    stt_enabled:    sttEnabledInput.checked,
    stt_model_path: sttModelInput.value,
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

document.getElementById('refresh_transcripts').onclick = async () => {
  try {
    const r = await j('/api/transcripts');
    if (!r.ok) {
      transcriptsBox.textContent = "Error: " + (r.reason || "unknown");
      return;
    }
    let lines = [];
    for (const t of r.transcripts) {
      const ts = new Date(t.time * 1000).toLocaleTimeString();
      const mhz = (t.freq_hz / 1e6).toFixed(3);
      const off = t.offset.toFixed(1);
      lines.push(`[${ts} +${off}s @ ${mhz} MHz] ${t.text}`);
    }
    transcriptsBox.textContent = lines.join("\\n") || "(no transcripts yet)";
  } catch (e) {
    transcriptsBox.textContent = "Error fetching transcripts: " + e;
  }
};

document.getElementById('clear_transcripts').onclick = async () => {
  try {
    await j('/api/transcripts/clear', 'POST');
    transcriptsBox.textContent = "";
  } catch (e) {
    transcriptsBox.textContent = "Error clearing transcripts: " + e;
  }
};

scanEnabledInput.onchange = () => {
  // purely UI; real enable/disable happens on Start/Retune
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
    global _rtl_proc, _aplay_proc
    with _proc_lock:
        rtl = _rtl_proc
        aplay = _aplay_proc

    # Determine actual running state from processes
    rtl_ok = rtl is not None and rtl.poll() is None
    aplay_ok = aplay is not None and aplay.poll() is None
    running = rtl_ok and aplay_ok

    state["running"] = running

    s = dict(state)
    s["pid_rtl"] = rtl.pid if rtl is not None else None
    s["pid_aplay"] = aplay.pid if _aplay_proc is not None else None
    s["last_start"] = _last_start
    s["vosk_available"] = _vosk_available
    s["transcript_count"] = len(_transcripts)
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
    """
    Build rtl_fm command as a list for subprocess (no shell).

    If scan_enabled:
      use built-in scan syntax: -f start_freq:end_freq:step_size
      (only if start < end and step > 0, else fall back to single freq)
    else:
      use single frequency: -f freq_hz
    """
    freq_hz = int(state["freq"])
    bw_hz = int(state["bandwidth"]) if state["bandwidth"] else 50000
    gain = int(state["gain"])
    squelch = int(state.get("squelch", 0))

    rtl_s = max(24000, min(192000, bw_hz))

    # Sanitise modulation
    mod = (state.get("modulation") or "fm").lower()
    allowed = {"fm", "wbfm", "raw", "am", "usb", "lsb"}
    if mod not in allowed:
        mod = "fm"

    cmd = [
        "rtl_fm",
        "-d", str(int(state["device_index"])),
        "-M", mod,
        "-s", str(rtl_s),
        "-E", "deemp",
    ]

    # Frequency / scan
    if state.get("scan_enabled", False):
        start = int(state.get("scan_start", freq_hz))
        end = int(state.get("scan_end", freq_hz))
        step = int(state.get("scan_step", 25_000))
        if end > start and step > 0:
            freq_arg = f"{start}:{end}:{step}"
        else:
            # invalid scan params; fall back to single freq
            freq_arg = str(freq_hz)
    else:
        freq_arg = str(freq_hz)

    cmd += ["-f", freq_arg]

    # Gain
    if gain:
        cmd += ["-g", str(gain)]
    else:
        cmd += ["-g", "0"]

    # Squelch
    if squelch > 0:
        cmd += ["-l", str(squelch)]

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
    global _rtl_proc, _aplay_proc, _pipe_thread

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

    if _aplay_proc is not None:
        if _aplay_proc.poll() is None:
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

    _pipe_thread = None
    state["running"] = False


def _init_stt_if_needed(samplerate: int) -> bool:
    """
    Lazily load Vosk model / recognizer if STT is enabled and Vosk is available.
    Returns True if STT will be active, False otherwise.
    """
    global _stt_model, _stt_recognizer

    if not state.get("stt_enabled", False):
        return False
    if not _vosk_available:
        return False

    model_path = state.get("stt_model_path") or ""
    if not os.path.isdir(model_path):
        return False

    try:
        if _stt_model is None or not isinstance(_stt_model, Model):
            _stt_model = Model(model_path)
        _stt_recognizer = KaldiRecognizer(_stt_model, samplerate)
        return True
    except Exception:
        _stt_model = None
        _stt_recognizer = None
        return False


def _pipe_audio_loop():
    """
    Forward PCM from rtl_fm -> aplay and inject a single beep
    when squelch "opens" (carrier/audio present).

    Also, if STT is enabled, feed the same PCM chunks into Vosk
    and append completed phrases into _transcripts.
    """
    global _pipe_thread, _transcripts, _stt_recognizer

    # Snapshot config once for this run
    with _proc_lock:
        open_th = int(state.get("open_threshold", 2000))
        close_th = int(state.get("close_threshold", 250))
        beep_freq = int(state.get("beep_freq", 1000))
        beep_freq = max(200, min(beep_freq, 4000))
        samplerate = int(state["samplerate"])
        beep_chunk = make_beep(sr=samplerate, freq=beep_freq)
        freq_hz = float(state.get("freq", 0.0))

    # STT init
    stt_on = _init_stt_if_needed(samplerate)

    # Tunables
    HANG_SEC = 0.7           # keep squelch_open this long after last strong audio
    MIN_BEEP_INTERVAL = 1.2  # minimum time between beeps

    squelch_open = False
    last_strong_time = 0.0   # last time level >= open_th
    last_beep_time = 0.0

    bytes_per_sample = 2
    total_samples = 0  # for STT offset

    while True:
        with _proc_lock:
            rtl = _rtl_proc
            aplay = _aplay_proc
            local_stt_on = stt_on and (_stt_recognizer is not None)

        if rtl is None or aplay is None:
            break

        try:
            chunk = rtl.stdout.read(4096)
        except Exception:
            break

        if not chunk:
            break

        total_samples += len(chunk) // bytes_per_sample

        # Compute a crude max-abs level over this chunk
        level = 0
        try:
            for (s,) in struct.iter_unpack("<h", chunk):
                if s < 0:
                    s = -s
                if s > level:
                    level = s
        except Exception:
            level = 0

        now = time.time()

        # --- squelch / beep logic ---

        if level >= open_th:
            # strong audio present
            if not squelch_open:
                # we're crossing from closed -> open
                if now - last_beep_time >= MIN_BEEP_INTERVAL:
                    try:
                        aplay.stdin.write(beep_chunk)
                        aplay.stdin.flush()
                    except Exception:
                        pass
                    last_beep_time = now
            squelch_open = True
            last_strong_time = now

        elif level <= close_th:
            # weak/quiet: only close squelch if we've been quiet long enough
            if squelch_open and (now - last_strong_time) >= HANG_SEC:
                squelch_open = False

        # --- forward the actual audio to aplay ---
        try:
            aplay.stdin.write(chunk)
            aplay.stdin.flush()
        except Exception:
            break

        # --- optional STT ---
        if local_stt_on:
            try:
                if _stt_recognizer.AcceptWaveform(chunk):
                    result = _stt_recognizer.Result()
                    data = json.loads(result) if result else {}
                    text = (data.get("text") or "").strip()
                    if text:
                        offset_sec = total_samples / float(samplerate)
                        entry = {
                            "time": time.time(),
                            "offset": offset_sec,
                            "freq_hz": freq_hz,
                            "text": text,
                        }
                        _transcripts.append(entry)
                        if len(_transcripts) > 200:
                            _transcripts = _transcripts[-200:]
            except Exception:
                # disable STT if it explodes, but keep audio running
                with _proc_lock:
                    state["stt_enabled"] = False
                _stt_recognizer = None
                stt_on = False

    # At end, flush final STT result
    if stt_on and _stt_recognizer is not None:
        try:
            final = _stt_recognizer.FinalResult()
            data = json.loads(final) if final else {}
            text = (data.get("text") or "").strip()
            if text:
                offset_sec = total_samples / float(samplerate)
                entry = {
                    "time": time.time(),
                    "offset": offset_sec,
                    "freq_hz": freq_hz,
                    "text": text,
                    "final": True,
                }
                _transcripts.append(entry)
                if len(_transcripts) > 200:
                    _transcripts = _transcripts[-200:]
        except Exception:
            pass

    with _proc_lock:
        _pipe_thread = None


@app.route("/api/start", methods=["POST"])
def api_start():
    global _rtl_proc, _aplay_proc, _pipe_thread, _last_start

    data = request.get_json() or {}

    # Update state from payload
    for key in (
        "freq", "bandwidth", "modulation", "gain", "samplerate",
        "audio_device", "device_index", "squelch",
        "open_threshold", "close_threshold", "beep_freq",
        "scan_start", "scan_end", "scan_step",
        "stt_model_path",
    ):
        if key in data:
            if key == "modulation":
                state["modulation"] = str(data[key]).lower()
                continue
            if isinstance(state[key], float):
                state[key] = float(data[key])
            elif isinstance(state[key], int):
                state[key] = int(data[key])
            else:
                state[key] = data[key]

    if "scan_enabled" in data:
        state["scan_enabled"] = bool(data["scan_enabled"])
    if "stt_enabled" in data:
        state["stt_enabled"] = bool(data["stt_enabled"])

    with _proc_lock:
        # Stop any existing chain (for retune)
        _stop_current_locked()

        # Volume
        set_volume(state.get("volume", 80))

        rtl_cmd = build_rtl_cmd()
        aplay_cmd = build_aplay_cmd()

        # Save last commands for debug / status
        state["last_rtl_cmd"] = rtl_cmd
        state["last_aplay_cmd"] = aplay_cmd

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

        # Then start aplay, consuming PCM from us (thread)
        try:
            _aplay_proc = subprocess.Popen(
                aplay_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except Exception as e:
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

        # Start audio pipe thread
        _pipe_thread = threading.Thread(target=_pipe_audio_loop, daemon=True)
        _pipe_thread.start()

        _last_start = time.time()
        state["running"] = True

    return jsonify({
        "ok": True,
        "rtl_cmd": rtl_cmd,
        "aplay_cmd": aplay_cmd,
        "pid_rtl": _rtl_proc.pid,
        "pid_aplay": _aplay_proc.pid,
    })


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


@app.route("/api/transcripts")
def api_transcripts():
    return jsonify({
        "ok": True,
        "transcripts": _transcripts,
        "stt_enabled": state.get("stt_enabled", False),
        "vosk_available": _vosk_available,
    })


@app.route("/api/transcripts/clear", methods=["POST"])
def api_transcripts_clear():
    global _transcripts
    _transcripts = []
    return jsonify({"ok": True})


if __name__ == "__main__":
    print("Starting RTL-FM control server on :8080", flush=True)
    set_volume(state["volume"])
    app.run(host="0.0.0.0", port=8080)
