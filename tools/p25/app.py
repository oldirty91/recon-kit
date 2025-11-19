#!/usr/bin/env python3
import os
import json
import subprocess
import threading
from pathlib import Path
from typing import Optional, Dict, Any
import re
import time


from flask import Flask, jsonify, request, Response

app = Flask(__name__)

# ---------- Paths / Config ----------
OP25_RX = "/opt/op25/op25/gr-op25_repeater/apps/rx.py"
OP25_APPS_DIR = "/opt/op25/op25/gr-op25_repeater/apps"
OP25_TDMA_DIR = "/opt/op25/op25/gr-op25_repeater/apps/tdma"
OP25_PY_DIR = "/opt/op25/op25/gr-op25_repeater/python"
OP25_TX_DIR = "/opt/op25/op25/gr-op25_repeater/apps/tx"  # for op25_c4fm_mod

TRUNK_TSV = "/config/trunk.tsv"
TALKGROUPS_TSV = "/config/riscon.tsv"   # talkgroups (used by trunking logic, not directly on cli)

LOG_DIR = Path("/var/log/op25")
STDERR_LOG = LOG_DIR / "op25_stderr.log"

DEFAULT_SAMPLE_RATE = 1000000  # 1 Msps
DEFAULT_GAIN = 40
DEFAULT_DEVICE_INDEX = 0

LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------- State ----------
_proc_lock = threading.Lock()
_proc: Optional[subprocess.Popen] = None
_last_cmd: Optional[Dict[str, Any]] = None
_last_exit_code: Optional[int] = None


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
        # "-l", "http:0.0.0.0:8765",  # OP25 web UI
        "-l", "8765",  # OP25 web UI
        "-U",
        "-O", "plughw:2,0",
        "-v", "10",
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
        return jsonify({
            "running": running,
            "pid": _proc.pid if running and _proc else None,
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

        # ---- Make sure all OP25 dirs are on PYTHONPATH ----
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

        # *** IMPORTANT: run from /config so 'riscon.tsv' is found ***
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=stderr_f,
            env=env,
            cwd="/config",          # <--- this is the key line
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

# ---------- "Now Playing" (Talkgroup + Radio ID) ----------

# Very loose regex patterns to catch common OP25 log formats.
# We can tighten these once we see your actual log lines.
_TG_PATTERNS = [
    re.compile(r'\btgid[ =:]+(\d+)', re.I),
    re.compile(r'\btg[ =:]+(\d+)', re.I),
]
_SRC_PATTERNS = [
    re.compile(r'\bsrcaddr[ =:]+(\d+)', re.I),
    re.compile(r'\bsrc[ =:]+(\d+)', re.I),
]

def _parse_nowplaying_from_log():
    """
    Look at the end of the stderr log and try to pull out the latest
    talkgroup + radio ID line.
    Returns (tgid: int|None, srcaddr: int|None, raw_line: str|None).
    """
    if not STDERR_LOG.exists():
        return None, None, None

    try:
        with STDERR_LOG.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            start = max(0, size - 8192)  # last ~8kB
            f.seek(start, os.SEEK_SET)
            chunk = f.read().decode(errors="replace")
    except Exception:
        return None, None, None

    lines = chunk.splitlines()
    tgid = None
    srcaddr = None
    raw_line = None

    # Walk backwards looking for the most recent line with a TG
    for line in reversed(lines):
        if tgid is None:
            for pat in _TG_PATTERNS:
                m = pat.search(line)
                if m:
                    tgid = int(m.group(1))
                    raw_line = line
                    break

        if srcaddr is None:
            for pat in _SRC_PATTERNS:
                m = pat.search(line)
                if m:
                    srcaddr = int(m.group(1))
                    # don't break; we still want tgid if not found yet
                    break

        if tgid is not None:
            # Good enough; we found a TG line
            break

    return tgid, srcaddr, raw_line


@app.get("/nowplaying")
def nowplaying():
    """
    Return the most recent talkgroup + radio ID seen in the log.
    """
    tgid, srcaddr, raw = _parse_nowplaying_from_log()
    if tgid is None:
        return jsonify({
            "talkgroup": None,
            "srcaddr": None,
            "raw": None,
            "age_seconds": None,
        })

    # We don’t have an exact timestamp from the log without parsing it,
    # so just return "0" (just now) for now; we can refine later if needed.
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
    .nowplaying { margin-bottom: 1rem; padding: 1rem; background: #222; border-radius: 8px; }
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
    <h2>Now Playing</h2>
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
    const r = await fetch('/nowplaying');
    const j = await r.json();
    const el = document.getElementById('nowplaying');

    if (!j.talkgroup) {
      el.textContent = 'Idle / no recent voice traffic';
      return;
    }

    let line = `TG ${j.talkgroup}`;
    if (j.srcaddr) {
      line += `    Radio ${j.srcaddr}`;
    }
    if (j.age_seconds != null) {
      line += `    • ${j.age_seconds.toFixed(1)}s ago`;
    }

    el.textContent = line + (j.raw ? `\n${j.raw}` : '');
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
    app.run(host="0.0.0.0", port=9005)
