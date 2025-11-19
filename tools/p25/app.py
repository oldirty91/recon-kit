#!/usr/bin/env python3
import os
import json
import subprocess
import threading
from pathlib import Path
from typing import Optional, Dict, Any
import re
import csv

from flask import Flask, jsonify, request, Response

app = Flask(__name__)

# ---------- Paths / Config ----------

OP25_RX = "/opt/op25/op25/gr-op25_repeater/apps/rx.py"
OP25_APPS_DIR = "/opt/op25/op25/gr-op25_repeater/apps"
OP25_TDMA_DIR = "/opt/op25/op25/gr-op25_repeater/apps/tdma"
OP25_PY_DIR = "/opt/op25/op25/gr-op25_repeater/python"
OP25_TX_DIR = "/opt/op25/op25/gr-op25_repeater/apps/tx"  # for op25_c4fm_mod

TRUNK_TSV = "/config/trunk.tsv"
TALKGROUPS_TSV = "/config/riscon.tsv"   # talkgroup list

LOG_DIR = Path("/var/log/op25")
LOG_DIR.mkdir(parents=True, exist_ok=True)
STDERR_LOG = LOG_DIR / "op25_stderr.log"

DEFAULT_SAMPLE_RATE = 1000000  # 1 Msps
DEFAULT_GAIN = 40
DEFAULT_DEVICE_INDEX = 0

# ---------- State ----------

_proc_lock = threading.Lock()
_proc: Optional[subprocess.Popen] = None
_last_cmd: Optional[Dict[str, Any]] = None
_last_exit_code: Optional[int] = None

# ---------- Talkgroup map ----------

TGID_MAP: dict[int, str] = {}

# In-memory ring buffer of recent TG events (persists while container is running)
RECENT_TG_EVENTS: list = []
RECENT_TG_LOCK = threading.Lock()


def _load_talkgroup_map():
    """
    Load talkgroup names from /config/riscon.tsv into TGID_MAP.

    Supports your simple 2-column TSV:
        DEC <tab> AlphaTag

    If you later switch to a Radioreference-style headered TSV,
    this will still work as long as the first column is the DEC
    and the second is some human-friendly name.
    """
    global TGID_MAP
    tg_file = Path(TALKGROUPS_TSV)

    if not tg_file.exists():
        print("[p25] talkgroup file /config/riscon.tsv not found, TG names disabled", flush=True)
        return

    TGID_MAP.clear()
    count = 0

    try:
        with tg_file.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t")

            for row in reader:
                if not row:
                    continue

                # First column should be DEC (numeric talkgroup ID)
                dec_str = (row[0] or "").strip()
                if not dec_str or not re.match(r"^\d+$", dec_str):
                    # skip non-numeric / header-ish rows if they ever appear
                    continue

                try:
                    tgid = int(dec_str)
                except ValueError:
                    continue

                # Second column is your label (Alpha Tag)
                name = ""
                if len(row) > 1:
                    name = (row[1] or "").strip()

                if not name:
                    name = f"TG {tgid}"

                TGID_MAP[tgid] = name
                count += 1

        print(f"[p25] loaded {count} talkgroup entries from {tg_file}", flush=True)

    except Exception as e:
        print(f"[p25] failed to load talkgroup map: {e}", flush=True)


# ---------- Subprocess monitor ----------

def _monitor_proc(p: subprocess.Popen):
    """Background monitor: wait for process to exit and record exit code."""
    global _proc, _last_exit_code
    exit_code = p.wait()
    with _proc_lock:
        _last_exit_code = exit_code
        _proc = None


def _build_cmd(device_index: int, sample_rate: int, lna_gain: int) -> Dict[str, Any]:
    """
    Build the rx.py command line.
    Use osmosdr-style args: rtl=<index>, no separate -d.
    """
    args_str = f"rtl={device_index}"

    cmd = [
        "python3",
        OP25_RX,
        "--args", args_str,
        "-S", str(sample_rate),
        "-q", "0",
        "--gains", f"lna:{lna_gain}",
        "-T", TRUNK_TSV,
        "-2",            # phase 2
        "--nocrypt",     # skip encrypted
        "-l", "8765",    # OP25 terminal / HTTP port (no http:// prefix)
        "-U",            # enable UDP audio
        "-O", "plughw:2,0",  # ALSA output device on the Pi
        "-v", "10",      # verbose so we see TG + radio IDs in the log
    ]

    return {
        "cmd": cmd,
        "overrides": {
            "device_index": device_index,
            "sample_rate": sample_rate,
            "lna_gain": lna_gain,
        },
        "stderr_path": str(STDERR_LOG),
    }


# ---------- API endpoints ----------

@app.get("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.get("/status")
def status():
    with _proc_lock:
        running = _proc is not None and _proc.poll() is None
        pid = _proc.pid if running and _proc is not None else None
        return jsonify({
            "running": running,
            "pid": pid,
            "last_exit_code": _last_exit_code,
            "last_config": _last_cmd,
        })


@app.post("/start")
def start():
    global _proc, _last_cmd, _last_exit_code

    data = request.get_json(silent=True) or {}
    device_index = int(data.get("device_index", DEFAULT_DEVICE_INDEX))
    sample_rate = int(data.get("sample_rate", DEFAULT_SAMPLE_RATE))
    lna_gain = int(data.get("lna_gain", DEFAULT_GAIN))

    # Sanity checks
    if not os.path.exists(OP25_RX):
        return jsonify({"error": f"rx.py not found at {OP25_RX}"}), 500

    if not os.path.exists(TRUNK_TSV):
        return jsonify({"error": f"trunk.tsv not found at {TRUNK_TSV}"}), 500

    with _proc_lock:
        if _proc is not None and _proc.poll() is None:
            return jsonify({"error": "op25 already running"}), 409

        cfg = _build_cmd(device_index, sample_rate, lna_gain)
        cmd = cfg["cmd"]
        _last_cmd = cfg
        _last_exit_code = None

        STDERR_LOG.parent.mkdir(parents=True, exist_ok=True)
        stderr_f = open(STDERR_LOG, "ab", buffering=0)

        # Ensure all OP25 dirs are on PYTHONPATH
        env = os.environ.copy()
        extra_paths = [
            OP25_APPS_DIR,
            OP25_TDMA_DIR,
            OP25_PY_DIR,
            OP25_TX_DIR,
        ]
        existing_py = env.get("PYTHONPATH", "")
        extra = ":".join(extra_paths)
        env["PYTHONPATH"] = extra + (":" + existing_py if existing_py else "")

        # run from /config so relative tsv paths work
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=stderr_f,
            env=env,
            cwd="/config",
        )
        _proc = p

        t = threading.Thread(target=_monitor_proc, args=(p,), daemon=True)
        t.start()

        return jsonify({
            "status": "started",
            "pid": p.pid,
            "cmd": cmd,
            "stderr_path": str(STDERR_LOG),
            "pythonpath": env["PYTHONPATH"],
            "cwd": "/config",
        })


@app.post("/stop")
def stop():
    global _proc
    with _proc_lock:
        if _proc is None or _proc.poll() is not None:
            return jsonify({"status": "not_running"})
        _proc.terminate()
        return jsonify({"status": "stopping", "pid": _proc.pid})


@app.get("/logs")
def logs():
    """Return tail of stderr log."""
    if not STDERR_LOG.exists():
        return jsonify({"lines": [], "error": "no log file"}), 200

    try:
        with STDERR_LOG.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            start = max(0, size - 4096)
            f.seek(start, os.SEEK_SET)
            chunk = f.read().decode(errors="replace")
        lines = chunk.splitlines()[-200:]
        return jsonify({"lines": lines})
    except Exception as e:
        return jsonify({"lines": [], "error": str(e)}), 500


# ---------- Talkgroup history / now-playing ----------

_TG_PATTERNS = [
    re.compile(r"\btgid[ =:]+(\d+)", re.I),
    re.compile(r"\btg[ =:]+(\d+)", re.I),
]
_SRC_PATTERNS = [
    re.compile(r"\bsrcaddr[ =:]+(\d+)", re.I),
    re.compile(r"\bsrc[ =:]+(\d+)", re.I),
]


def _parse_tg_history_from_log(max_events: int = 200):
    """
    Update and return a persistent in-memory history of talkgroup events.

    - Parses only the tail of the stderr log (last ~64kB).
    - Extracts new TG/src events and appends them to RECENT_TG_EVENTS.
    - Keeps only the last max_events items.
    - If no new TG lines are found, we still return the existing history.
    """
    global RECENT_TG_EVENTS

    # If there is no log file, just return whatever we have in memory
    if not STDERR_LOG.exists():
        with RECENT_TG_LOCK:
            return list(RECENT_TG_EVENTS)

    try:
        with STDERR_LOG.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            start = max(0, size - 65536)  # last 64kB
            f.seek(start, os.SEEK_SET)
            chunk = f.read().decode(errors="replace")
    except Exception:
        # On read error, fall back to whatever we already have
        with RECENT_TG_LOCK:
            return list(RECENT_TG_EVENTS)

    lines = chunk.splitlines()
    new_events = []

    for line in lines:
        tgid = None
        src = None

        # Find talkgroup
        for pat in _TG_PATTERNS:
            m = pat.search(line)
            if m:
                try:
                    tgid = int(m.group(1))
                except ValueError:
                    tgid = None
                break

        if tgid is None:
            continue

        # Find source (radio ID)
        for pat in _SRC_PATTERNS:
            m = pat.search(line)
            if m:
                try:
                    src = int(m.group(1))
                except ValueError:
                    src = None
                break

        # crude timestamp: first two tokens if they look like MM/DD/YY HH:MM:SS...
        timestamp = None
        parts = line.split()
        if len(parts) >= 2 and re.match(r"\d{2}/\d{2}/\d{2}", parts[0]):
            timestamp = parts[0] + " " + parts[1]

        label = TGID_MAP.get(tgid)

        ev = {
            "timestamp": timestamp,
            "talkgroup": tgid,
            "label": label,
            "srcaddr": src,
            "raw": line.strip(),
        }
        new_events.append(ev)

    with RECENT_TG_LOCK:
        # Build a set of existing keys so we don't spam duplicates every poll
        existing_keys = {
            (e.get("timestamp"), e.get("talkgroup"), e.get("srcaddr"))
            for e in RECENT_TG_EVENTS
        }

        for ev in new_events:
            key = (ev.get("timestamp"), ev.get("talkgroup"), ev.get("srcaddr"))
            if key in existing_keys:
                continue
            RECENT_TG_EVENTS.append(ev)
            existing_keys.add(key)

        # Trim to last max_events
        if len(RECENT_TG_EVENTS) > max_events:
            RECENT_TG_EVENTS = RECENT_TG_EVENTS[-max_events:]

        # Return a copy so callers can't mutate our buffer
        return list(RECENT_TG_EVENTS)


def _parse_nowplaying_from_log():
    """
    Return the most recent talkgroup + src from the log.
    (Still uses direct log tail; history uses the ring buffer.)
    """
    if not STDERR_LOG.exists():
        return None, None, None

    try:
        with STDERR_LOG.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            start = max(0, size - 8192)
            f.seek(start, os.SEEK_SET)
            chunk = f.read().decode(errors="replace")
    except Exception:
        return None, None, None

    lines = chunk.splitlines()
    tgid = None
    src = None
    raw = None

    for line in reversed(lines):
        if tgid is None:
            for pat in _TG_PATTERNS:
                m = pat.search(line)
                if m:
                    try:
                        tgid = int(m.group(1))
                    except ValueError:
                        tgid = None
                    raw = line
                    break

        if src is None:
            for pat in _SRC_PATTERNS:
                m = pat.search(line)
                if m:
                    try:
                        src = int(m.group(1))
                    except ValueError:
                        src = None
                    break

        if tgid is not None:
            break

    return tgid, src, raw


@app.get("/history")
def history():
    events = _parse_tg_history_from_log(max_events=200)
    return jsonify({"events": events})


@app.get("/nowplaying")
def nowplaying():
    tgid, srcaddr, raw = _parse_nowplaying_from_log()
    if tgid is None:
        return jsonify({
            "talkgroup": None,
            "srcaddr": None,
            "raw": None,
            "age_seconds": None,
        })
    return jsonify({
        "talkgroup": tgid,
        "srcaddr": srcaddr,
        "raw": raw,
        "age_seconds": 0.0,
    })


# ---------- Simple Web GUI ----------

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Recon P25 Control</title>
  <style>
    body { font-family: sans-serif; padding: 1rem; background: #111; color: #eee; }
    h1 { margin-top: 0; }
    .controls, .status, .logs, .help { margin-bottom: 1rem; padding: 1rem; background: #222; border-radius: 8px; }
    label { display: inline-block; width: 120px; }
    input { background: #111; color: #eee; border: 1px solid #444; border-radius: 4px; padding: 4px 6px; }
    button { margin-right: 0.5rem; padding: 0.4rem 0.8rem; border-radius: 4px; background: #2b6; color: #fff; border: none; cursor: pointer; }
    button.stop { background: #b33; }
    pre { max-height: 400px; overflow-y: auto; background: #000; padding: 0.5rem; border-radius: 4px; }
    a { color: #6cf; }
    .badge { display: inline-block; padding: 0.1rem 0.5rem; border-radius: 999px; font-size: 0.8rem; }
    .badge.ok { background: #2b6; }
    .badge.err { background: #b33; }
    .nowplaying {
      margin-bottom: 1rem;
      padding: 1rem;
      background: #222;
      border-radius: 8px;
    }
    .nowplaying pre {
      max-height: 240px;
      overflow-y: auto;
      background: #111;
      padding: 0.5rem;
      border-radius: 4px;
      font-size: 0.8rem;
      line-height: 1.3;
    }
  </style>
</head>
<body>
  <h1>Recon P25 Decoder</h1>

  <div class="controls">
    <h2>Controls</h2>
    <div>
      <label>RTL Device Index</label>
      <input id="device_index" type="number" value="0" min="0" />
    </div>
    <div>
      <label>Sample Rate (Hz)</label>
      <input id="sample_rate" type="number" value="1000000" />
    </div>
    <div>
      <label>LNA Gain (dB)</label>
      <input id="lna_gain" type="number" value="40" />
    </div>
    <div style="margin-top:0.5rem;">
      <button onclick="start()">Start</button>
      <button class="stop" onclick="stop()">Stop</button>
      <span id="health" class="badge">health: ?</span>
    </div>
  </div>

  <div class="nowplaying">
    <h2>Now Playing (History)</h2>
    <pre id="nowplaying">Idle / no recent voice traffic</pre>
  </div>

  <div class="status">
    <h2>Status</h2>
    <pre id="status">loading...</pre>
  </div>

  <div class="logs">
    <h2>stderr log (rx.py)</h2>
    <pre id="logs">loading...</pre>
  </div>

  <div class="help">
    <h2>OP25 Web UI</h2>
    <p>
      Once running, OP25 exposes its own web console at
      <code>http://&lt;this_container_host&gt;:8765</code>.
    </p>
  </div>

<script>
async function refreshStatus() {
  try {
    const r = await fetch('/status');
    const j = await r.json();
    document.getElementById('status').textContent = JSON.stringify(j, null, 2);
  } catch (e) {
    document.getElementById('status').textContent = 'error: ' + e;
  }
}

async function refreshHealth() {
  try {
    const r = await fetch('/healthz');
    const j = await r.json();
    const el = document.getElementById('health');
    if (j.status === 'ok') {
      el.textContent = 'health: ok';
      el.className = 'badge ok';
    } else {
      el.textContent = 'health: ' + (j.status || 'unknown');
      el.className = 'badge err';
    }
  } catch (e) {
    const el = document.getElementById('health');
    el.textContent = 'health: error';
    el.className = 'badge err';
  }
}

async function refreshLogs() {
  try {
    const r = await fetch('/logs');
    const j = await r.json();
    if (j.lines) {
      document.getElementById('logs').textContent = j.lines.join('\\n');
    } else {
      document.getElementById('logs').textContent = 'no logs';
    }
  } catch (e) {
    document.getElementById('logs').textContent = 'error: ' + e;
  }
}

async function start() {
  const device_index = parseInt(document.getElementById('device_index').value || '0', 10);
  const sample_rate = parseInt(document.getElementById('sample_rate').value || '1000000', 10);
  const lna_gain = parseInt(document.getElementById('lna_gain').value || '40', 10);

  try {
    const r = await fetch('/start', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ device_index, sample_rate, lna_gain })
    });
    const j = await r.json();
    alert('start: ' + JSON.stringify(j));
  } catch (e) {
    alert('start error: ' + e);
  }
  refreshStatus();
}

async function stop() {
  try {
    const r = await fetch('/stop', { method: 'POST' });
    const j = await r.json();
    alert('stop: ' + JSON.stringify(j));
  } catch (e) {
    alert('stop error: ' + e);
  }
  refreshStatus();
}

async function refreshNowPlaying() {
  try {
    const r = await fetch('/history');
    const j = await r.json();
    const el = document.getElementById('nowplaying');

    if (!j.events || j.events.length === 0) {
      el.textContent = 'No recent voice activity';
      return;
    }

    const events = j.events || [];

    // newest first
    const lines = events.slice().reverse().map(ev => {
      const parts = [];

      if (ev.timestamp) {
        parts.push(ev.timestamp);
      }

      if (ev.talkgroup) {
        let tgText = `TG ${ev.talkgroup}`;
        if (ev.label) {
          tgText += ` (${ev.label})`;
        }
        parts.push(tgText);
      }

      if (ev.srcaddr) {
        parts.push(`Radio ${ev.srcaddr}`);
      }

      return parts.join('  •  ');
    });

    el.textContent = lines.join('\\n');
  } catch (e) {
    const el = document.getElementById('nowplaying');
    el.textContent = 'error: ' + e;
  }
}

setInterval(refreshStatus, 2000);
setInterval(refreshLogs, 2000);
setInterval(refreshHealth, 5000);
setInterval(refreshNowPlaying, 1000);
refreshStatus();
refreshLogs();
refreshHealth();
refreshNowPlaying();
</script>
</body>
</html>
"""

@app.get("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


if __name__ == "__main__":
    _load_talkgroup_map()
    app.run(host="0.0.0.0", port=9005)
