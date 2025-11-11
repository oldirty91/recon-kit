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
except Exception as e:
    print("STT: Vosk import failed:", e, flush=True)
    Model = None
    KaldiRecognizer = None
    _vosk_available = False

# Global processes + lock
_rtl_proc = None
_aplay_proc = None
_pipe_thread = None
_proc_lock = threading.Lock()
_last_start = None

# STT model + transcripts
_stt_model = None            # cached Vosk model
_transcripts = []            # list of {time, offset, freq_hz, text, final?}

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

    # Beep / detection config (offsets above noise floor)
    # These are offsets added to a learned "quiet" level.
    "open_threshold": 2000,  # offset: noise_floor + this = open level
    "close_threshold": 250,  # offset: noise_floor + this = close level

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
    <label>Open threshold offset
      <input id="open_threshold" type="number" step="100" value="2000">
    </label>
    <label>Close threshold offset
      <input id="close_threshold" type="number" step="50" value="250">
    </label>
    <small>Dynamic thresholds = noise floor + these offsets (only when squelch > 0).</small>
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
      Enable STT (only used when squelch > 0)
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
       <b>Open threshold offset:</b> ${s.open_threshold}<br>
       <b>Close threshold offset:</b> ${s.close_threshold}<br>
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
      sttStatusNote.textContent = "Vosk not installed or failed to load.";
    } else {
      sttStatusNote.textContent = s.stt_enabled
        ? "STT enabled (squelch-gated)."
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
      const off = (t.offset || 0).toFixed(1);
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

    rtl_ok = rtl is not None and rtl.poll() is None
    aplay_ok = aplay is not None and aplay.poll() is None
    running = rtl_ok and aplay_ok

    state["running"] = running

    s = dict(state)
    s["pid_rtl"] = rtl.pid if rtl is not None else None
    s["pid_aplay"] = aplay.pid if aplay is not None else None
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
    """
    freq_hz = int(state["freq"])
    bw_hz = int(state["bandwidth"]) if state["bandwidth"] else 50000
    gain = int(state["gain"])
    squelch = int(state.get("squelch", 0))

    rtl_s = max(24000, min(192000, bw_hz))

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

    if state.get("scan_enabled", False):
        start = int(state.get("scan_start", freq_hz))
        end = int(state.get("scan_end", freq_hz))
        step = int(state.get("scan_step", 25_000))
        if end > start and step > 0:
            freq_arg = f"{start}:{end}:{step}"
        else:
            freq_arg = str(freq_hz)
    else:
        freq_arg = str(freq_hz)

    cmd += ["-f", freq_arg]

    if gain:
        cmd += ["-g", str(gain)]
    else:
        cmd += ["-g", "0"]

    if squelch > 0:
        cmd += ["-l", str(squelch)]

    return cmd


def build_aplay_cmd():
    return [
        "aplay",
        "-f", "S16_LE",
        "-r", str(int(state["samplerate"])),
        "-c", "1",
        "--device", state["audio_device"],
        "-",
    ]


def _stop_current_locked():
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


def _ensure_stt_model():
    """
    Lazily load / cache Vosk model if STT is enabled and squelch > 0.
    Returns the Model or None and prints debug.
    """
    global _stt_model

    if not _vosk_available:
        print("STT: Vosk not available in this container.", flush=True)
        return None
    if not state.get("stt_enabled", False):
        print("STT: disabled in state.", flush=True)
        return None
    if int(state.get("squelch", 0)) <= 0:
        print("STT: squelch <= 0, STT will not run.", flush=True)
        return None

    model_path = state.get("stt_model_path") or ""
    if not os.path.isdir(model_path):
        print(f"STT: model path not a directory: {model_path}", flush=True)
        return None

    try:
        if _stt_model is None or getattr(_stt_model, "_path", None) != model_path:
            print(f"STT: loading model from {model_path}", flush=True)
            _stt_model = Model(model_path)
            _stt_model._path = model_path
            print("STT: model loaded OK.", flush=True)
        else:
            print(f"STT: reusing cached model {model_path}", flush=True)
        return _stt_model
    except Exception as e:
        print("STT: failed to load model:", e, flush=True)
        _stt_model = None
        return None


def _pipe_audio_loop():
    """
    Core loop:
      - read PCM from rtl_fm
      - forward original PCM to aplay (for listening)
      - detect "transmissions" (TX) with a simple hang-based state machine
        for beeps only
      - run STT as a continuous recognizer independent of TX
    """
    global _pipe_thread, _transcripts

    with _proc_lock:
        open_off    = int(state.get("open_threshold", 400))   # margin above noise
        close_off   = int(state.get("close_threshold", 300))
        beep_freq   = int(state.get("beep_freq", 1000))
        beep_freq   = max(200, min(beep_freq, 4000))
        samplerate  = int(state["samplerate"])                # rtl_fm/aplay sample rate
        squelch_val = int(state.get("squelch", 0))
        beep_chunk  = make_beep(sr=samplerate, freq=beep_freq)
        freq_hz     = float(state.get("freq", 0.0))
        stt_enabled = bool(state.get("stt_enabled", False))
        stt_model_path = state.get("stt_model_path", "")

    # ---- STT model & recognizer (continuous) ----
    stt_model = _ensure_stt_model()
    stt_allowed = stt_enabled and (stt_model is not None)

    STT_RATE  = 16000       # Vosk model rate
    STT_DECIM = 3           # ~48k -> ~16k
    STT_GAIN  = 6           # extra gain just for Vosk

    recognizer = None
    if stt_allowed:
        try:
            recognizer = KaldiRecognizer(stt_model, STT_RATE)
            print(
                f"STT: continuous recognizer started @ {freq_hz/1e6:.3f} MHz "
                f"model={stt_model_path}, STT_RATE={STT_RATE}",
                flush=True,
            )
        except Exception as e:
            print("STT: failed to create continuous recognizer:", e, flush=True)
            recognizer = None
            with _proc_lock:
                state["stt_enabled"] = False
            stt_allowed = False

    # STT debug accumulators
    stt_samples = 0
    stt_sum_sq  = 0.0

    def _make_stt_chunk(raw_chunk: bytes) -> bytes:
        """
        Downsample raw S16_LE chunk by factor STT_DECIM and apply STT_GAIN.
        Returns new S16_LE bytes at ~16 kHz.
        """
        out = bytearray()
        try:
            for i, (s,) in enumerate(struct.iter_unpack("<h", raw_chunk)):
                if i % STT_DECIM != 0:
                    continue
                v = s * STT_GAIN
                if v > 32767:
                    v = 32767
                elif v < -32768:
                    v = -32768
                out += struct.pack("<h", int(v))
        except Exception as e:
            print("STT: error in _make_stt_chunk:", e, flush=True)
            return b""
        return bytes(out)

    print(
        f"PIPE: start; squelch={squelch_val}, STT allowed={stt_allowed}, "
        f"open_off={open_off}, close_off={close_off}",
        flush=True,
    )

    # ---- TX detection for beeps (simple hang logic) ----
    MIN_RMS_NOISE = 20.0       # below this is basically silence

    HANG_SEC        = 0.7      # keep TX open this long after last "loud"
    MIN_TX_DURATION = 0.7      # minimum duration to consider it a TX
    MIN_TX_GAP      = 0.7      # gap between TXs
    MIN_BEEP_INT    = 0.4      # minimum gap between beeps

    tx_active        = False
    tx_start_time    = 0.0
    last_tx_end_time = 0.0
    hang_until       = 0.0
    last_beep_time   = 0.0

    # adaptive noise floor (only learned when not in TX)
    noise_floor = 0.0
    alpha       = 0.98  # smoothing

    while True:
        with _proc_lock:
            rtl = _rtl_proc
            aplay = _aplay_proc

        if rtl is None or aplay is None:
            break

        try:
            chunk = rtl.stdout.read(4096)
        except Exception as e:
            print("PIPE: read error:", e, flush=True)
            break

        if not chunk:
            break

        # ---- RMS for detection / noise floor ----
        n_samples = len(chunk) // 2
        if n_samples <= 0:
            continue

        sum_sq = 0
        try:
            for (s,) in struct.iter_unpack("<h", chunk):
                sum_sq += s * s
        except Exception:
            sum_sq = 0

        rms = math.sqrt(sum_sq / n_samples) if n_samples > 0 else 0.0
        now = time.time()

        # ---- Update noise floor (only when not TX and squelch>0) ----
        if squelch_val > 0 and not tx_active:
            if noise_floor == 0.0:
                noise_floor = rms
            else:
                noise_floor = alpha * noise_floor + (1.0 - alpha) * rms

        if squelch_val > 0:
            base = max(noise_floor, MIN_RMS_NOISE)
            open_th  = base + float(open_off)
            close_th = base + float(close_off)
        else:
            # squelch disabled → no TX detection / beeps
            open_th = close_th = float("inf")

        # ---- TX state machine (for beeps only) ----
        if squelch_val > 0:
            if not tx_active:
                # consider starting TX if we cross open_th and had enough gap
                if rms > open_th and (now - last_tx_end_time) >= MIN_TX_GAP:
                    tx_active     = True
                    tx_start_time = now
                    hang_until    = now + HANG_SEC
                    print(
                        f"TX START: rms={rms:.1f}, noise_floor={noise_floor:.1f}, "
                        f"open_th={open_th:.1f}, close_th={close_th:.1f}",
                        flush=True,
                    )
            else:
                # already in TX
                if rms > close_th:
                    # refresh hang timer while loud enough
                    hang_until = now + HANG_SEC

                tx_duration = now - tx_start_time

                # end TX only after hang time + minimum duration
                if now > hang_until and tx_duration >= MIN_TX_DURATION:
                    tx_active        = False
                    last_tx_end_time = now
                    print(
                        f"TX END: dur={tx_duration:.2f}s, rms={rms:.1f}, "
                        f"noise_floor={noise_floor:.1f}, open_th={open_th:.1f}, "
                        f"close_th={close_th:.1f}",
                        flush=True,
                    )

                    # single beep at end of TX
                    if now - last_beep_time >= MIN_BEEP_INT:
                        try:
                            aplay.stdin.write(beep_chunk)
                            aplay.stdin.flush()
                            print("BEEP: end-of-transmission", flush=True)
                        except Exception as e:
                            print("BEEP: write error:", e, flush=True)
                        last_beep_time = now

        # ---- Always forward original audio to aplay ----
        try:
            aplay.stdin.write(chunk)
            aplay.stdin.flush()
        except Exception as e:
            print("PIPE: aplay write error:", e, flush=True)
            break

        # ---- Continuous STT feed (independent of TX) ----
        if stt_allowed and recognizer is not None:
            stt_chunk = _make_stt_chunk(chunk)
            if stt_chunk:
                ns = len(stt_chunk) // 2
                stt_samples += ns
                # debug RMS for STT stream
                try:
                    sum_sq_stt = 0
                    for (s,) in struct.iter_unpack("<h", stt_chunk):
                        sum_sq_stt += s * s
                    stt_sum_sq += sum_sq_stt
                except Exception:
                    pass

                try:
                    if recognizer.AcceptWaveform(stt_chunk):
                        res = recognizer.Result()
                        try:
                            data = json.loads(res) if res else {}
                        except Exception:
                            data = {}
                        txt = (data.get("text") or "").strip()
                        stt_rms = (
                            math.sqrt(stt_sum_sq / stt_samples)
                            if stt_samples > 0 else 0.0
                        )
                        print(
                            f"STT: Result raw={data}, stt_samples={stt_samples}, "
                            f"stt_rms={stt_rms:.1f}",
                            flush=True,
                        )
                        if txt:
                            entry = {
                                "time": time.time(),
                                "offset": stt_samples / float(STT_RATE)
                                if STT_RATE > 0 else 0.0,
                                "freq_hz": freq_hz,
                                "text": txt,
                                "final": True,
                            }
                            _transcripts.append(entry)
                            if len(_transcripts) > 200:
                                _transcripts = _transcripts[-200:]
                            print("STT: stored transcript:", entry, flush=True)
                        # reset STT energy stats each utterance
                        stt_samples = 0
                        stt_sum_sq  = 0.0
                    else:
                        # Optional: see partials for debugging
                        # p = recognizer.PartialResult()
                        # print("STT: partial:", p, flush=True)
                        pass
                except Exception as e:
                    print("STT: AcceptWaveform error:", e, flush=True)
                    with _proc_lock:
                        state["stt_enabled"] = False
                    recognizer = None
                    stt_allowed = False

    # ---- Flush STT on exit ----
    if stt_allowed and recognizer is not None:
        try:
            final = recognizer.FinalResult()
            try:
                data = json.loads(final) if final else {}
            except Exception:
                data = {}
            txt = (data.get("text") or "").strip()
            stt_rms = (
                math.sqrt(stt_sum_sq / stt_samples)
                if stt_samples > 0 else 0.0
            )
            print(
                f"STT: FinalResult on loop exit: raw={data}, "
                f"stt_samples={stt_samples}, stt_rms={stt_rms:.1f}",
                flush=True,
            )
            if txt:
                entry = {
                    "time": time.time(),
                    "offset": stt_samples / float(STT_RATE)
                    if STT_RATE > 0 else 0.0,
                    "freq_hz": freq_hz,
                    "text": txt,
                    "final": True,
                }
                _transcripts.append(entry)
                if len(_transcripts) > 200:
                    _transcripts = _transcripts[-200:]
                print("STT: stored transcript on exit:", entry, flush=True)
        except Exception as e:
            print("STT: FinalResult error on exit:", e, flush=True)

    with _proc_lock:
        _pipe_thread = None




@app.route("/api/start", methods=["POST"])
def api_start():
    global _rtl_proc, _aplay_proc, _pipe_thread, _last_start

    data = request.get_json() or {}

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

    print("API /start: state =", state, flush=True)

    with _proc_lock:
        _stop_current_locked()
        set_volume(state.get("volume", 80))

        rtl_cmd = build_rtl_cmd()
        aplay_cmd = build_aplay_cmd()
        state["last_rtl_cmd"] = rtl_cmd
        state["last_aplay_cmd"] = aplay_cmd

        print("Starting rtl_fm:", rtl_cmd, flush=True)
        try:
            _rtl_proc = subprocess.Popen(
                rtl_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except Exception as e:
            _rtl_proc = None
            print("Failed to start rtl_fm:", e, flush=True)
            return jsonify({"ok": False, "error": f"failed to start rtl_fm: {e}"}), 500

        print("Starting aplay:", aplay_cmd, flush=True)
        try:
            _aplay_proc = subprocess.Popen(
                aplay_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except Exception as e:
            print("Failed to start aplay:", e, flush=True)
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
