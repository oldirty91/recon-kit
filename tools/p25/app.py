#!/usr/bin/env python3
import os
import json
import socket
import threading
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List
import re
import csv

from flask import Flask, jsonify, request, Response

app = Flask(__name__)

# ---------- Paths / Config ----------

CONFIG_PATH = os.getenv("P25_PROFILE_CONFIG", "/config/p25_profiles.json")

OP25_APPS_DIR = "/opt/op25/op25/gr-op25_repeater/apps"
OP25_TDMA_DIR = "/opt/op25/op25/gr-op25_repeater/apps/tdma"
OP25_PY_DIR   = "/opt/op25/op25/gr-op25_repeater/python"
OP25_TX_DIR   = "/opt/op25/op25/gr-op25_repeater/apps/tx"

TRUNK_TSV_DEFAULT = "/config/trunk.tsv"  # still used in your raw_cmd
LOG_DIR = Path("/var/log/op25")
LOG_DIR.mkdir(parents=True, exist_ok=True)
STDERR_LOG = LOG_DIR / "op25_stderr.log"

DEFAULT_SAMPLE_RATE = 1000000  # informational only now; real value comes from raw_cmd
DEFAULT_GAIN = 40
DEFAULT_DEVICE_INDEX = 0

# Base port to start allocating UDP listen ports for fanout
UDP_BASE_PORT = int(os.getenv("P25_UDP_BASE_PORT", "9100"))

# ---------- State ----------

# profiles loaded from JSON
PROFILES: Dict[str, Dict[str, Any]] = {}

# running instances:
#   key: instance_id (e.g. "riscon-0-1")
#   value: dict with proc, profile_id, rtl_index, udp_port, forwarder_thread, forwarder_stop_event
INSTANCES: Dict[str, Dict[str, Any]] = {}
INSTANCES_LOCK = threading.Lock()
INSTANCE_COUNTER = 0  # just to make unique ids

_last_exit_code: Optional[int] = None
_last_cmd: Optional[List[str]] = None

# ---------- Talkgroup map ----------

TGID_MAP: Dict[int, str] = {}

# ---------- Talkgroup history parsing ----------

_TG_PATTERNS = [
    re.compile(r"\btgid[ =:]+(\d+)", re.I),
    re.compile(r"\btg[ =:]+(\d+)", re.I),
]
_SRC_PATTERNS = [
    re.compile(r"\bsrcaddr[ =:]+(\d+)", re.I),
    re.compile(r"\bsrc[ =:]+(\d+)", re.I),
]

TG_HISTORY: List[Dict[str, Any]] = []
TG_HISTORY_MAX = 200
_LOG_READ_POS: int = 0
_LOG_INODE: Optional[int] = None


# ========== Helpers ==========

def _load_profiles():
    global PROFILES
    cfg_file = Path(CONFIG_PATH)
    if not cfg_file.exists():
        print(f"[p25] profile config {CONFIG_PATH} not found", flush=True)
        PROFILES = {}
        return

    try:
        data = json.loads(cfg_file.read_text(encoding="utf-8"))
        profiles = data.get("profiles", [])
        PROFILES = {}
        for p in profiles:
            pid = p.get("id")
            if not pid:
                continue
            PROFILES[pid] = p
        print(f"[p25] loaded {len(PROFILES)} profiles from {CONFIG_PATH}", flush=True)
    except Exception as e:
        print(f"[p25] failed to load profiles: {e}", flush=True)
        PROFILES = {}


def _load_talkgroup_map_from_profile(profile: Dict[str, Any]):
    """
    Load TGID_MAP from profile['talkgroups_tsv'] if present,
    otherwise fall back to /config/riscon.tsv like before.
    """
    global TGID_MAP
    tg_file = Path(profile.get("talkgroups_tsv", "/config/riscon.tsv"))

    if not tg_file.exists():
        print(f"[p25] talkgroup file {tg_file} not found, TG names disabled", flush=True)
        TGID_MAP.clear()
        return

    TGID_MAP.clear()
    count = 0

    try:
        with tg_file.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t")

            for row in reader:
                if not row:
                    continue

                dec_str = (row[0] or "").strip()
                if not dec_str or not re.match(r"^\d+$", dec_str):
                    continue

                try:
                    tgid = int(dec_str)
                except ValueError:
                    continue

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
        TGID_MAP.clear()


def _next_udp_port() -> int:
    """Assign a unique UDP listen port for a new instance."""
    with INSTANCES_LOCK:
        in_use = {inst["udp_port"] for inst in INSTANCES.values()}
        port = UDP_BASE_PORT
        while port in in_use:
            port += 1
        return port


def _make_instance_id(profile_id: str, rtl_index: int) -> str:
    global INSTANCE_COUNTER
    with INSTANCES_LOCK:
        INSTANCE_COUNTER += 1
        return f"{profile_id}-{rtl_index}-{INSTANCE_COUNTER}"


def _build_cmd_from_profile(profile: Dict[str, Any], rtl_index: int, udp_port: int) -> List[str]:
    """
    Take profile['raw_cmd'] and substitute {rtl_index} and {udp_port}.
    """
    raw_cmd = profile.get("raw_cmd")
    if not raw_cmd or not isinstance(raw_cmd, list):
        raise RuntimeError(f"profile {profile.get('id')} missing raw_cmd list")

    new_cmd = []
    for tok in raw_cmd:
        if isinstance(tok, str):
            tok = tok.replace("{rtl_index}", str(rtl_index)).replace("{udp_port}", str(udp_port))
        new_cmd.append(tok)
    return new_cmd


def _monitor_proc(instance_id: str, proc: subprocess.Popen):
    """Background monitor: record exit and clean up instance when process exits."""
    global _last_exit_code, _last_cmd
    exit_code = proc.wait()
    with INSTANCES_LOCK:
        inst = INSTANCES.get(instance_id)
        if inst is not None:
            inst["running"] = False
        _last_exit_code = exit_code
        # keep _last_cmd as last launched command (for /status)


def _udp_fanout_thread(udp_port: int, targets: List[Dict[str, Any]], stop_event: threading.Event):
    """
    Listen on udp_port and fan the raw audio to all targets.
    """
    print(f"[p25] UDP fanout listening on 0.0.0.0:{udp_port} -> {targets}", flush=True)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", udp_port))
    sock.settimeout(1.0)

    try:
        while not stop_event.is_set():
            try:
                data, addr = sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError:
                break

            for tgt in targets:
                host = tgt.get("host")
                port = int(tgt.get("port", udp_port))
                if not host:
                    continue
                try:
                    sock.sendto(data, (host, port))
                except OSError:
                    # don't kill the thread for a transient send error
                    continue
    finally:
        sock.close()
        print(f"[p25] UDP fanout on port {udp_port} stopped", flush=True)


def _parse_line_to_event(line: str) -> Optional[Dict[str, Any]]:
    """Parse a single stderr line into a talkgroup event dict, or None."""
    tgid = None
    src = None

    for pat in _TG_PATTERNS:
        m = pat.search(line)
        if m:
            try:
                tgid = int(m.group(1))
            except ValueError:
                tgid = None
            break

    if tgid is None:
        return None

    for pat in _SRC_PATTERNS:
        m = pat.search(line)
        if m:
            try:
                src = int(m.group(1))
            except ValueError:
                src = None
            break

    timestamp = None
    parts = line.split()
    if len(parts) >= 2 and re.match(r"\d{2}/\d{2}/\d{2}", parts[0]):
        timestamp = parts[0] + " " + parts[1]

    label = TGID_MAP.get(tgid)

    return {
        "timestamp": timestamp,
        "talkgroup": tgid,
        "label": label,
        "srcaddr": src,
        "raw": line.strip(),
    }


def _update_history_from_log():
    """
    Incrementally read new lines from STDERR_LOG and append talkgroup events
    into TG_HISTORY, trimming to TG_HISTORY_MAX.
    """
    global _LOG_READ_POS, _LOG_INODE, TG_HISTORY

    if not STDERR_LOG.exists():
        return

    try:
        with STDERR_LOG.open("rb") as f:
            st = os.fstat(f.fileno())
            inode = st.st_ino
            size = st.st_size

            if _LOG_INODE is None or inode != _LOG_INODE or _LOG_READ_POS > size:
                _LOG_INODE = inode
                _LOG_READ_POS = 0

            f.seek(_LOG_READ_POS)
            chunk = f.read().decode(errors="replace")
            _LOG_READ_POS = f.tell()
    except Exception:
        return

    if not chunk:
        return

    for line in chunk.splitlines():
        ev = _parse_line_to_event(line)
        if ev is not None:
            TG_HISTORY.append(ev)
            if len(TG_HISTORY) > TG_HISTORY_MAX:
                TG_HISTORY = TG_HISTORY[-TG_HISTORY_MAX:]


# ========== API: basic health / logs / status ==========

@app.get("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.get("/profiles")
def list_profiles():
    out = []
    for pid, p in PROFILES.items():
        out.append({
            "id": pid,
            "label": p.get("label", pid),
            "talkgroups_tsv": p.get("talkgroups_tsv"),
            "udp_targets": p.get("udp_targets", []),
        })
    return jsonify({"profiles": out})


@app.get("/instances")
def list_instances():
    with INSTANCES_LOCK:
        inst_list = []
        for iid, inst in INSTANCES.items():
            inst_list.append({
                "instance_id": iid,
                "profile_id": inst["profile_id"],
                "rtl_index": inst["rtl_index"],
                "udp_port": inst["udp_port"],
                "running": inst.get("running", False),
                "pid": inst["proc"].pid if inst.get("proc") and inst["proc"].poll() is None else None,
            })
    return jsonify({"instances": inst_list})


@app.get("/status")
def status():
    with INSTANCES_LOCK:
        running_any = any(inst.get("running", False) for inst in INSTANCES.values())
        pid_list = [
            inst["proc"].pid for inst in INSTANCES.values()
            if inst.get("proc") and inst["proc"].poll() is None
        ]
        return jsonify({
            "running": running_any,
            "pids": pid_list,
            "last_exit_code": _last_exit_code,
            "last_cmd": _last_cmd,
        })


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


# ========== API: history / nowplaying ==========

@app.get("/history")
def history():
    _update_history_from_log()
    return jsonify({"events": TG_HISTORY})


@app.get("/nowplaying")
def nowplaying():
    _update_history_from_log()
    if not TG_HISTORY:
        return jsonify({
            "talkgroup": None,
            "srcaddr": None,
            "label": None,
            "raw": None,
            "age_seconds": None,
        })
    ev = TG_HISTORY[-1]
    return jsonify({
        "talkgroup": ev.get("talkgroup"),
        "srcaddr": ev.get("srcaddr"),
        "label": ev.get("label"),
        "raw": ev.get("raw"),
        "age_seconds": 0.0,
    })


# ========== API: start/stop instances ==========

@app.post("/start_instance")
def api_start_instance():
    """
    Body:
      {
        "profile_id": "riscon",
        "rtl_index": 0
      }
    """
    global _last_cmd, _last_exit_code

    data = request.get_json(silent=True) or {}
    profile_id = data.get("profile_id")
    rtl_index = int(data.get("rtl_index", DEFAULT_DEVICE_INDEX))

    if not profile_id or profile_id not in PROFILES:
        return jsonify({"error": f"unknown profile_id {profile_id}"}), 400

    profile = PROFILES[profile_id]

    # Optional: sanity check trunk.tsv exists if you rely on it in raw_cmd
    if not os.path.exists(TRUNK_TSV_DEFAULT):
        print(f"[p25] WARNING: {TRUNK_TSV_DEFAULT} not found", flush=True)

    udp_port = _next_udp_port()
    cmd = _build_cmd_from_profile(profile, rtl_index, udp_port)

    # update TG name map for this profile
    _load_talkgroup_map_from_profile(profile)

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

    STDERR_LOG.parent.mkdir(parents=True, exist_ok=True)
    stderr_f = open(STDERR_LOG, "ab", buffering=0)

    # Start UDP fanout thread if we have targets
    udp_targets = profile.get("udp_targets", [])
    fanout_thread = None
    fanout_stop = threading.Event()
    if udp_targets:
        fanout_thread = threading.Thread(
            target=_udp_fanout_thread,
            args=(udp_port, udp_targets, fanout_stop),
            daemon=True,
        )
        fanout_thread.start()

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=stderr_f,
            env=env,
            cwd="/config",
        )
    except Exception as e:
        # stop fanout if process launch failed
        if fanout_thread is not None:
            fanout_stop.set()
        return jsonify({"error": f"failed to start rx.py: {e}"}), 500

    instance_id = _make_instance_id(profile_id, rtl_index)
    inst_rec = {
        "instance_id": instance_id,
        "profile_id": profile_id,
        "rtl_index": rtl_index,
        "udp_port": udp_port,
        "proc": proc,
        "running": True,
        "fanout_thread": fanout_thread,
        "fanout_stop": fanout_stop,
    }

    with INSTANCES_LOCK:
        INSTANCES[instance_id] = inst_rec
        _last_cmd = cmd
        _last_exit_code = None

    t = threading.Thread(target=_monitor_proc, args=(instance_id, proc), daemon=True)
    t.start()

    return jsonify({
        "status": "started",
        "instance_id": instance_id,
        "profile_id": profile_id,
        "rtl_index": rtl_index,
        "udp_port": udp_port,
        "cmd": cmd,
        "udp_targets": udp_targets,
    })


@app.post("/stop_instance")
def api_stop_instance():
    """
    Body can be either:
      { "instance_id": "riscon-0-1" }
    or
      { "rtl_index": 0 }
    If rtl_index is used and multiple instances share it, all are stopped.
    """
    data = request.get_json(silent=True) or {}
    instance_id = data.get("instance_id")
    rtl_index = data.get("rtl_index")

    to_stop: List[str] = []

    with INSTANCES_LOCK:
        if instance_id:
            if instance_id in INSTANCES:
                to_stop.append(instance_id)
            else:
                return jsonify({"error": f"instance_id {instance_id} not found"}), 404
        elif rtl_index is not None:
            rtl_index = int(rtl_index)
            for iid, inst in list(INSTANCES.items()):
                if inst["rtl_index"] == rtl_index:
                    to_stop.append(iid)
            if not to_stop:
                return jsonify({"status": "no_instances_for_rtl", "rtl_index": rtl_index})
        else:
            return jsonify({"error": "must provide instance_id or rtl_index"}), 400

    results = []
    for iid in to_stop:
        with INSTANCES_LOCK:
            inst = INSTANCES.get(iid)
        if not inst:
            continue

        proc: subprocess.Popen = inst["proc"]
        fanout_stop: threading.Event = inst["fanout_stop"]
        fanout_thread: Optional[threading.Thread] = inst["fanout_thread"]

        if proc and proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception as e:
                results.append({"instance_id": iid, "error": str(e)})
                continue

        # stop fanout
        if fanout_stop:
            fanout_stop.set()

        # we don't join thread here; it's daemon, will exit once stop_event set

        with INSTANCES_LOCK:
            INSTANCES.pop(iid, None)

        results.append({"instance_id": iid, "stopped": True})

    return jsonify({"results": results})


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
    .controls, .status, .logs, .help, .instances, .nowplaying {
      margin-bottom: 1rem;
      padding: 1rem;
      background: #222;
      border-radius: 8px;
    }
    label { display: inline-block; width: 140px; }
    input, select {
      background: #111;
      color: #eee;
      border: 1px solid #444;
      border-radius: 4px;
      padding: 4px 6px;
    }
    button {
      margin-right: 0.5rem;
      padding: 0.4rem 0.8rem;
      border-radius: 4px;
      background: #2b6;
      color: #fff;
      border: none;
      cursor: pointer;
    }
    button.stop { background: #b33; }
    pre {
      max-height: 400px;
      overflow-y: auto;
      background: #000;
      padding: 0.5rem;
      border-radius: 4px;
    }
    a { color: #6cf; }
    .badge {
      display: inline-block;
      padding: 0.1rem 0.5rem;
      border-radius: 999px;
      font-size: 0.8rem;
    }
    .badge.ok { background: #2b6; }
    .badge.err { background: #b33; }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
    }
    th, td {
      padding: 4px 6px;
      border-bottom: 1px solid #333;
      text-align: left;
    }
  </style>
</head>
<body>
  <h1>Recon P25 Decoder (Multi-Instance)</h1>

  <div class="controls">
    <h2>Start Instance</h2>
    <div>
      <label>Profile</label>
      <select id="profile_select"></select>
    </div>
    <div>
      <label>RTL Device Index</label>
      <input id="rtl_index" type="number" value="0" min="0" />
    </div>
    <div style="margin-top:0.5rem;">
      <button onclick="startInstance()">Start Instance</button>
      <span id="health" class="badge">health: ?</span>
    </div>
  </div>

  <div class="instances">
    <h2>Running Instances</h2>
    <table>
      <thead>
        <tr>
          <th>Instance ID</th>
          <th>Profile</th>
          <th>RTL</th>
          <th>UDP Port</th>
          <th>PID</th>
          <th>Action</th>
        </tr>
      </thead>
      <tbody id="instances_tbody">
      </tbody>
    </table>
  </div>

  <div class="nowplaying">
    <h2>Now Playing (History)</h2>
    <pre id="nowplaying">Idle / no recent voice activity</pre>
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
async function loadProfiles() {
  try {
    const r = await fetch('/profiles');
    const j = await r.json();
    const sel = document.getElementById('profile_select');
    sel.innerHTML = '';
    (j.profiles || []).forEach(p => {
      const opt = document.createElement('option');
      opt.value = p.id;
      opt.textContent = p.label || p.id;
      sel.appendChild(opt);
    });
  } catch (e) {
    console.error('loadProfiles error', e);
  }
}

async function refreshInstances() {
  try {
    const r = await fetch('/instances');
    const j = await r.json();
    const tbody = document.getElementById('instances_tbody');
    tbody.innerHTML = '';

    (j.instances || []).forEach(inst => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${inst.instance_id}</td>
        <td>${inst.profile_id}</td>
        <td>${inst.rtl_index}</td>
        <td>${inst.udp_port}</td>
        <td>${inst.pid || ''}</td>
        <td><button class="stop" onclick="stopInstance('${inst.instance_id}')">Stop</button></td>
      `;
      tbody.appendChild(tr);
    });
  } catch (e) {
    console.error('refreshInstances error', e);
  }
}

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

async function startInstance() {
  const sel = document.getElementById('profile_select');
  const profile_id = sel.value;
  const rtl_index = parseInt(document.getElementById('rtl_index').value || '0', 10);

  try {
    const r = await fetch('/start_instance', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ profile_id, rtl_index })
    });
    const j = await r.json();
    alert('start_instance: ' + JSON.stringify(j));
  } catch (e) {
    alert('start_instance error: ' + e);
  }
  refreshStatus();
  refreshInstances();
}

async function stopInstance(instance_id) {
  try {
    const r = await fetch('/stop_instance', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ instance_id })
    });
    const j = await r.json();
    alert('stop_instance: ' + JSON.stringify(j));
  } catch (e) {
    alert('stop_instance error: ' + e);
  }
  refreshStatus();
  refreshInstances();
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

    const events = j.events.slice().reverse();

    const lines = events.map(ev => {
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
setInterval(refreshInstances, 3000);

loadProfiles();
refreshInstances();
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
    _load_profiles()
    # default: load TG names from first profile if any
    if PROFILES:
        first = next(iter(PROFILES.values()))
        _load_talkgroup_map_from_profile(first)
    app.run(host="0.0.0.0", port=9005)
