#!/usr/bin/env python3
import os
import signal
import subprocess
import threading
import time
import math
import struct
import json

from flask import Flask, jsonify, request, render_template_string

# Try Vosk STT
try:
    import vosk  # type: ignore
    _VOSK_OK = True
except Exception:
    vosk = None
    _VOSK_OK = False

app = Flask(__name__)

# Global processes + lock
_rtl_proc = None
_aplay_proc = None
_pipe_thread = None
_proc_lock = threading.Lock()
_last_start = None

# STT globals
_stt_model = None
_stt_rec = None
_stt_model_path_loaded = None
_stt_lock = threading.Lock()
_transcripts = []  # list of dicts: {time, offset, freq_hz, text, final}
_STT_MAX_ITEMS = 200

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
    "open_threshold": 400,   # default offsets, not absolute
    "close_threshold": 300,
    "beep_freq": 1000,       # Hz, tone for this SDR
    # rtl_fm built-in scan config (range)
    "scan_enabled": False,
    "scan_start": 453.0e6,   # Hz
    "scan_end": 454.0e6,     # Hz
    "scan_step": 25e3,       # Hz
    # NEW: explicit comma-separated scan frequency list (MHz)
    # If set, we pass multiple -f entries and ignore range scanning.
    "scan_freqlist": "",
    # STT config
    "stt_enabled": True,
    # Default to SMALL model (your request)
    "stt_model_path": "/opt/models/vosk-model-small-en-us-0.15",
    "stt_gain": 6.0,         # linear gain applied to STT audio path
    # Debug: last commands
    "last_rtl_cmd": [],
    "last_aplay_cmd": [],
}


def make_beep(sr=48000, freq=1000, duration=0.06, volume=0.4):
    """Generate a short S16_LE mono beep."""
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
table { border-collapse: collapse; width: 100%; font-size: 0.9em; }
th, td { border: 1px solid #ccc; padding: 4px 6px; text-align: left; }
th { background: #eee; }
</style>
</head>
<body>
<h2>RTL-FM Controller</h2>

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
      <input id="open_threshold" type="number" step="50" value="400">
    </label>
    <label>Close threshold offset
      <input id="close_threshold" type="number" step="50" value="300">
    </label>
    <small>Thresholds are offsets above the tracked noise floor.</small>
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

    <label style="margin-top:12px;">Frequency list (MHz, comma-separated)
      <input id="scan_freqlist" type="text" placeholder="e.g. 462.5625,462.5875,462.6125">
    </label>
    <small>If set, uses multiple <code>-f</code> entries and overrides range/center.</small>
  </div>

  <div>
    <h3>Speech-to-Text (Vosk)</h3>
    <label>
      <input id="stt_enabled" type="checkbox" checked>
      Enable STT
    </label>
    <small id="stt_avail_note"></small>
    <label>Model path in container
      <input id="stt_model_path" type="text" value="/opt/models/vosk-model-small-en-us-0.15">
    </label>
    <label>STT gain (x)
      <input id="stt_gain" type="number" step="0.5" value="6.0">
    </label>
    <small>Gain applied only on STT path (not the audio you hear).</small>
  </div>
</div>

<div>
  <button id="start">Start / Retune</button>
  <button id="stop">Stop</button>
  <button id="status">Refresh</button>
</div>

<div class="status" id="statusBox">Loading...</div>
<button id="logs">Logs</button>
<pre id="logText"></pre>

<h3>Transcripts (last 50, newest first)</h3>
<table>
  <thead>
    <tr>
      <th>Time</th>
      <th>Freq (MHz)</th>
      <th>Text</th>
    </tr>
  </thead>
  <tbody id="transBody">
  </tbody>
</table>

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
const scanListInput    = document.getElementById('scan_freqlist');

const sttEnabledInput  = document.getElementById('stt_enabled');
const sttModelInput    = document.getElementById('stt_model_path');
const sttGainInput     = document.getElementById('stt_gain');
const sttAvailNote     = document.getElementById('stt_avail_note');
const transBody        = document.getElementById('transBody');

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
       <b>Open offset:</b> ${s.open_threshold}<br>
       <b>Close offset:</b> ${s.close_threshold}<br>
       <b>Scan enabled:</b> ${s.scan_enabled}<br>
       <b>Scan start:</b> ${(s.scan_start/1e6).toFixed(3)} MHz<br>
       <b>Scan end:</b> ${(s.scan_end/1e6).toFixed(3)} MHz<br>
       <b>Scan step:</b> ${(s.scan_step/1e3).toFixed(1)} kHz<br>
       <b>Freq list:</b> ${s.scan_freqlist || '(none)'}<br>
       <b>STT enabled:</b> ${s.stt_enabled}<br>
       <b>STT available:</b> ${s.stt_available}<br>
       <b>STT model path:</b> ${s.stt_model_path}<br>
       <b>STT gain:</b> ${s.stt_gain}<br>
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
    if (document.activeElement !== scanListInput)
      scanListInput.value = s.scan_freqlist || "";
    scanEnabledInput.checked = s.scan_enabled;

    sttEnabledInput.checked = s.stt_enabled;
    if (document.activeElement !== sttModelInput)
      sttModelInput.value = s.stt_model_path || "";
    if (document.activeElement !== sttGainInput)
      sttGainInput.value = s.stt_gain;

    sttAvailNote.textContent = s.stt_available
      ? "Vosk available."
      : "Vosk not available in this container.";
  } catch (e) {
    statusBox.innerHTML = "Error fetching status: " + e;
  }
}

document.getElementById('start').onclick = async () => {
  const payload = {
    freq: parseFloat(freqInput.value) * 1e6,
    bandwidth: parseFloat(bwInput.value) * 1e3,
    modulation: modInput.value,
    gain: parseInt(gainInput.value),
    device_index: parseInt(devIdxInput.value),
    squelch: parseInt(sqlInput.value),
    audio_device: devInput.value,
    beep_freq: parseInt(beepFreqInput.value),
    open_threshold: parseInt(openThreshInput.value),
    close_threshold: parseInt(closeThreshInput.value),

    scan_enabled: scanEnabledInput.checked,
    scan_start: parseFloat(scanStartInput.value) * 1e6,
    scan_end: parseFloat(scanEndInput.value) * 1e6,
    scan_step: parseFloat(scanStepInput.value) * 1e3,
    scan_freqlist: scanListInput.value,   // NEW

    stt_enabled: sttEnabledInput.checked,
    stt_model_path: sttModelInput.value,
    stt_gain: parseFloat(sttGainInput.value),
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

async function refreshTranscripts() {
  try {
    const r = await j('/api/transcripts');
    transBody.innerHTML = "";
    if (!r.ok || !r.items) return;
    // NEWEST FIRST: take last 50, then reverse for newest-first display
    const items = r.items.slice(-50).reverse();
    for (const t of items) {
      const tr = document.createElement('tr');
      const tdTime = document.createElement('td');
      const tdFreq = document.createElement('td');
      const tdText = document.createElement('td');

      const d = new Date(t.time * 1000);
      tdTime.textContent = d.toLocaleTimeString();
      tdFreq.textContent = (t.freq_hz/1e6).toFixed(3);
      tdText.textContent = t.text;

      tr.appendChild(tdTime);
      tr.appendChild(tdFreq);
      tr.appendChild(tdText);
      transBody.appendChild(tr);
    }
  } catch (e) {
    // ignore
  }
}

// Poll status + transcripts
refresh();
setInterval(refreshTranscripts, 2000);
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
    s["stt_available"] = bool(_VOSK_OK)
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

    Priority:
      1) If scan_freqlist is non-empty: add multiple -f <Hz> entries (overrides everything).
      2) Else if scan_enabled: -f start:end:step
      3) Else: -f freq_hz (single)
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

    # NEW: explicit list of frequencies (MHz, comma-separated)
    freqlist = (state.get("scan_freqlist") or "").strip()
    if freqlist:
        for token in freqlist.split(","):
            t = token.strip()
            if not t:
                continue
            try:
                mhz = float(t)
                cmd += ["-f", str(int(round(mhz * 1e6)))]
            except Exception:
                continue
    else:
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


def _ensure_stt_model():
    """Load Vosk model if needed."""
    global _stt_model, _stt_rec, _stt_model_path_loaded
    if not _VOSK_OK:
        return False

    model_path = state.get("stt_model_path") or ""
    if not model_path:
        return False

    if _stt_model is not None and _stt_model_path_loaded == model_path:
        return True

    if not os.path.isdir(model_path):
        print(f"STT: model path does not exist: {model_path}", flush=True)
        return False

    print(f"STT: loading model from {model_path}", flush=True)
    _stt_model = vosk.Model(model_path)
    _stt_model_path_loaded = model_path
    print("STT: model loaded OK.", flush=True)

    # recognizer created later in _make_stt_recognizer
    _stt_rec = None
    return True


def _make_stt_recognizer():
    global _stt_model, _stt_rec
    if not _ensure_stt_model():
        return None
    if _stt_rec is None:
        stt_rate = 16000
        _stt_rec = vosk.KaldiRecognizer(_stt_model, stt_rate)
        print(
            f"STT: continuous recognizer started @ {state['freq']/1e6:.3f} MHz "
            f"model={state['stt_model_path']}, STT_RATE={stt_rate}",
            flush=True
        )
    return _stt_rec


def _stt_feed(chunk_48k: bytes, stt_gain: float):
    """Downsample 48k -> 16k and feed into recognizer."""
    if not state.get("stt_enabled", False):
        return
    if not _VOSK_OK:
        return

    with _stt_lock:
        rec = _make_stt_recognizer()
        if rec is None:
            return

        # Interpret as 16-bit little-endian mono @ 48kHz
        try:
            samples = list(struct.iter_unpack("<h", chunk_48k))
        except Exception:
            return

        if not samples:
            return

        # Downsample by 3 (48k -> 16k) and apply gain
        out = bytearray()
        for i in range(0, len(samples), 3):
            s = samples[i][0]
            s = int(max(-32767, min(32767, s * stt_gain)))
            out.extend(struct.pack("<h", s))

        if not out:
            return

        # Feed into vosk
        if rec.AcceptWaveform(bytes(out)):
            res = rec.Result()
            try:
                obj = json.loads(res)
            except Exception:
                obj = {}
            text = (obj.get("text") or "").strip()
            if text:
                # simple RMS-ish for debug
                vals = [struct.unpack("<h", out[i:i+2])[0] for i in range(0, len(out), 2)]
                stt_rms = (sum(v*v for v in vals)/len(vals))**0.5 if vals else 0.0
                print(
                    f"STT: Result raw={obj}, stt_samples={len(vals)}, stt_rms={stt_rms:.1f}",
                    flush=True
                )
                _store_transcript(text)
        else:
            # optional: could inspect PartialResult; ignoring for now
            pass


def _store_transcript(text: str):
    """Append one transcript entry, keep bounded length."""
    global _transcripts
    item = {
        "time": time.time(),
        "offset": 0.0,  # placeholder; could track stream offset later
        "freq_hz": state.get("freq", 0.0),
        "text": text,
        "final": True,
    }
    _transcripts.append(item)
    if len(_transcripts) > _STT_MAX_ITEMS:
        _transcripts = _transcripts[-_STT_MAX_ITEMS:]
    print(f"STT: stored transcript: {item}", flush=True)


def _pipe_audio_loop():
    """
    Forward PCM from rtl_fm -> aplay and:
      - detect TX start/end based on dynamic thresholds
      - beep once at end of TX
      - optionally feed audio to Vosk STT (separate gain path)
    """
    global _pipe_thread

    with _proc_lock:
        squelch = int(state.get("squelch", 0))
        open_off = int(state.get("open_threshold", 400))
        close_off = int(state.get("close_threshold", 300))
        beep_freq = int(state.get("beep_freq", 1000))
        beep_freq = max(200, min(beep_freq, 4000))
        beep_chunk = make_beep(sr=int(state["samplerate"]), freq=beep_freq)
        stt_enabled = bool(state.get("stt_enabled", False))
        stt_gain = float(state.get("stt_gain", 6.0))

    HANG_SEC = 0.6
    noise_floor = 0.0
    noise_alpha = 0.01

    in_tx = False
    last_above = 0.0
    tx_start_time = None

    print(
        f"PIPE: start; squelch={squelch}, STT allowed={stt_enabled}, "
        f"open_off={open_off}, close_off={close_off}",
        flush=True
    )

    while True:
        with _proc_lock:
            rtl = _rtl_proc
            aplay = _aplay_proc

        if rtl is None or aplay is None:
            break

        try:
            chunk = rtl.stdout.read(4096)
        except Exception:
            break

        if not chunk:
            break

        # RMS on 16-bit mono audio
        try:
            vals = [s[0] for s in struct.iter_unpack("<h", chunk)]
        except Exception:
            vals = []

        if vals:
            sq = sum(v*v for v in vals) / len(vals)
            rms = math.sqrt(sq)
        else:
            rms = 0.0

        now = time.time()

        # Initialize / update noise floor when below threshold
        if noise_floor == 0.0:
            noise_floor = rms
        # only update noise floor from quiet-ish chunks
        if rms < noise_floor * 1.5:
            noise_floor = (1.0 - noise_alpha) * noise_floor + noise_alpha * rms

        # Dynamic thresholds (offsets above noise floor)
        open_th = noise_floor + (open_off if open_off > 0 else 0)
        close_th = noise_floor + (close_off if close_off > 0 else open_off)

        # --- TX state machine ---
        if not in_tx and rms >= open_th:
            in_tx = True
            tx_start_time = now
            last_above = now
            print(
                f"TX START: rms={rms:.1f}, noise_floor={noise_floor:.1f}, "
                f"open_th={open_th:.1f}, close_th={close_th:.1f}",
                flush=True
            )

        elif in_tx:
            if rms >= close_th:
                last_above = now
            elif now - last_above >= HANG_SEC:
                dur = now - (tx_start_time or now)
                print(
                    f"TX END: dur={dur:.2f}s, rms={rms:.1f}, "
                    f"noise_floor={noise_floor:.1f}, open_th={open_th:.1f}, "
                    f"close_th={close_th:.1f}",
                    flush=True
                )
                # one beep at end-of-TX, only if squelch is active
                if squelch > 0:
                    try:
                        aplay.stdin.write(beep_chunk)
                        aplay.stdin.flush()
                        print("BEEP: end-of-transmission", flush=True)
                    except Exception:
                        pass
                in_tx = False
                tx_start_time = None
                last_above = 0.0

        # --- STT path (continuous) ---
        if stt_enabled and squelch > 0:
            _stt_feed(chunk, stt_gain)

        # --- forward audio to speakers ---
        try:
            aplay.stdin.write(chunk)
            aplay.stdin.flush()
        except Exception:
            break

    with _proc_lock:
        _pipe_thread = None


@app.route("/api/start", methods=["POST"])
def api_start():
    global _rtl_proc, _aplay_proc, _pipe_thread, _last_start

    data = request.get_json() or {}
    print("API /start: state =", state, flush=True)

    # Update state from payload
    for key in (
        "freq", "bandwidth", "modulation", "gain", "samplerate",
        "audio_device", "device_index", "squelch",
        "open_threshold", "close_threshold", "beep_freq",
        "scan_start", "scan_end", "scan_step", "scan_freqlist",
        "stt_enabled", "stt_model_path", "stt_gain",
    ):
        if key in data:
            if key == "modulation":
                state["modulation"] = str(data[key]).lower()
                continue
            if key in ("stt_model_path", "audio_device", "scan_freqlist"):
                state[key] = str(data[key])
            elif isinstance(state.get(key), float):
                state[key] = float(data[key])
            elif isinstance(state.get(key), int):
                state[key] = int(data[key])
            else:
                state[key] = data[key]

    if "scan_enabled" in data:
        state["scan_enabled"] = bool(data["scan_enabled"])

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

    # STT model load happens lazily in _pipe_audio_loop / _make_stt_recognizer
    if state.get("stt_enabled") and _VOSK_OK:
        threading.Thread(target=_ensure_stt_model, daemon=True).start()

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
    return jsonify({"ok": True, "items": _transcripts})


if __name__ == "__main__":
    print("Starting RTL-FM control server on :8080", flush=True)
    if _VOSK_OK:
        print("Vosk Python module available.", flush=True)
    else:
        print("Vosk not available in this container.", flush=True)
    set_volume(state["volume"])
    app.run(host="0.0.0.0", port=8080)
