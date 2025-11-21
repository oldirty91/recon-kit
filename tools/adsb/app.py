#!/usr/bin/env python3
import os
import time
import json
import math
import threading
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from flask import Flask, jsonify, request, Response

app = Flask(__name__)

# ----------------------- ENV / CONFIG -----------------------

DUMP1090_BIN = os.getenv("DUMP1090_BIN", "dump1090")

# Default SDR index (can be overridden per API call)
DEFAULT_RTL_INDEX = int(os.getenv("RTL_DEVICE_INDEX", "0"))

# Where dump1090 writes JSON aircraft state
JSON_DIR = Path(os.getenv("JSON_DIR", "/run/dump1090"))
JSON_DIR.mkdir(parents=True, exist_ok=True)

# Optional extra args to pass to dump1090 (space separated)
DUMP1090_EXTRA = os.getenv("DUMP1090_EXTRA", "").strip()

# Reference position (default / fallback if GPS is not available)
HOME_LAT = float(os.getenv("HOME_LAT", "0.0"))
HOME_LON = float(os.getenv("HOME_LON", "0.0"))

# GPS service (recon-kit gps container)
GPS_URL = os.getenv("GPS_URL", "http://gps:8085/position")
GPS_TIMEOUT_SEC = float(os.getenv("GPS_TIMEOUT_SEC", "1.5"))

# How long we consider an aircraft "fresh" (seconds since last seen)
STALE_SEC = float(os.getenv("STALE_SEC", "60"))

# Watchdog check period
WATCHDOG_PERIOD = 1.0

# ----------------------- GLOBAL STATE -----------------------

_dump_proc: Optional[subprocess.Popen] = None
_dump_lock = threading.Lock()
_stop_deadline: Optional[float] = None  # epoch timestamp when we should auto-stop
_current_rtl_index: Optional[int] = None

_last_gps: Dict[str, Any] = {
    "lat": HOME_LAT,
    "lon": HOME_LON,
    "ts": None,
    "ok": False,
    "error": None,
}


# ----------------------- GEO HELPERS ------------------------


def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Great-circle distance in nautical miles.
    """
    R_m = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d_m = R_m * c
    return d_m / 1852.0  # meters -> nautical miles


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Bearing from (lat1, lon1) to (lat2, lon2) in degrees (0 = north, clockwise).
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)

    x = math.sin(dlambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    brng = math.degrees(math.atan2(x, y))
    return (brng + 360.0) % 360.0


def project_xy_m(lat_ref: float, lon_ref: float, lat: float, lon: float) -> Tuple[float, float]:
    """
    Simple equirectangular projection relative to (lat_ref, lon_ref).
    Returns (x, y) in meters (x east, y north).
    """
    lat_rad = math.radians(lat_ref)
    dx = math.radians(lon - lon_ref) * math.cos(lat_rad) * 6371000.0
    dy = math.radians(lat - lat_ref) * 6371000.0
    return dx, dy


# ----------------------- GPS INTEGRATION --------------------


def fetch_gps_position() -> Optional[Tuple[float, float]]:
    """
    Query the GPS microservice at GPS_URL expected to return JSON like:
      {"lat": 41.5, "lon": -70.5, ...}
    Returns (lat, lon) or None if unavailable.
    """
    global _last_gps
    try:
        r = requests.get(GPS_URL, timeout=GPS_TIMEOUT_SEC)
        r.raise_for_status()
        data = r.json()

        # Try common key names
        lat = data.get("lat") or data.get("latitude")
        lon = data.get("lon") or data.get("lng") or data.get("longitude")

        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            raise ValueError(f"Invalid lat/lon in GPS payload: {data}")

        _last_gps = {
            "lat": float(lat),
            "lon": float(lon),
            "ts": time.time(),
            "ok": True,
            "error": None,
        }
        return _last_gps["lat"], _last_gps["lon"]

    except Exception as e:
        _last_gps = {
            "lat": _last_gps.get("lat", HOME_LAT),
            "lon": _last_gps.get("lon", HOME_LON),
            "ts": time.time(),
            "ok": False,
            "error": str(e),
        }
        return None


def get_reference_point() -> Tuple[float, float, Dict[str, Any]]:
    """
    Determine reference point:

    1. If ?lat & ?lon query params are given, use those.
    2. Else try GPS_URL.
    3. Else fall back to HOME_LAT/HOME_LON.

    Returns (lat, lon, gps_meta)
    """
    # 1) explicit query override
    qlat = request.args.get("lat")
    qlon = request.args.get("lon")
    if qlat is not None and qlon is not None:
        try:
            lat = float(qlat)
            lon = float(qlon)
            gps_meta = {
                "source": "query",
                "lat": lat,
                "lon": lon,
                "ok": True,
                "error": None,
            }
            return lat, lon, gps_meta
        except ValueError:
            # fall through to GPS/HOME
            pass

    # 2) GPS container
    gps_pos = fetch_gps_position()
    if gps_pos is not None:
        lat, lon = gps_pos
        gps_meta = {
            "source": "gps",
            "lat": lat,
            "lon": lon,
            "ok": True,
            "error": None,
            "last_gps": _last_gps,
        }
        return lat, lon, gps_meta

    # 3) fallback
    gps_meta = {
        "source": "fallback_home",
        "lat": HOME_LAT,
        "lon": HOME_LON,
        "ok": False,
        "error": _last_gps.get("error"),
        "last_gps": _last_gps,
    }
    return HOME_LAT, HOME_LON, gps_meta


# ----------------------- DUMP1090 CONTROL -------------------


def _build_dump_cmd(rtl_index: int) -> List[str]:
    cmd = [
        DUMP1090_BIN,
        "--device-index",
        str(rtl_index),
        "--net",
        "--write-json",
        str(JSON_DIR),
        "--quiet",
    ]

    if DUMP1090_EXTRA:
        cmd.extend(DUMP1090_EXTRA.split())

    return cmd


def start_dump1090(duration_sec: Optional[float] = None, rtl_index: Optional[int] = None) -> Dict[str, Any]:
    """
    Start dump1090 on the given rtl_index (or DEFAULT_RTL_INDEX if None).
    If already running, returns status without restarting.
    """
    global _dump_proc, _stop_deadline, _current_rtl_index

    with _dump_lock:
        now = time.time()
        if _dump_proc is not None and _dump_proc.poll() is None:
            # Already running; we do NOT switch index mid-flight.
            if duration_sec is not None:
                _stop_deadline = now + duration_sec
            return {
                "status": "already_running",
                "pid": _dump_proc.pid,
                "stop_deadline": _stop_deadline,
                "rtl_index": _current_rtl_index,
            }

        idx = DEFAULT_RTL_INDEX if rtl_index is None else int(rtl_index)
        cmd = _build_dump_cmd(idx)
        try:
            _dump_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
            _current_rtl_index = idx
        except Exception as e:
            _dump_proc = None
            _stop_deadline = None
            _current_rtl_index = None
            return {"status": "error", "error": str(e), "rtl_index": idx}

        if duration_sec is not None:
            _stop_deadline = now + duration_sec
        else:
            _stop_deadline = None

        return {
            "status": "started",
            "pid": _dump_proc.pid,
            "stop_deadline": _stop_deadline,
            "rtl_index": _current_rtl_index,
        }


def stop_dump1090(reason: str = "manual") -> Dict[str, Any]:
    global _dump_proc, _stop_deadline, _current_rtl_index

    with _dump_lock:
        if _dump_proc is None:
            _stop_deadline = None
            _current_rtl_index = None
            return {"status": "not_running"}

        proc = _dump_proc
        _dump_proc = None
        _stop_deadline = None
        _current_rtl_index = None

    # Outside lock: actually terminate
    try:
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
    except Exception as e:
        return {"status": "error", "error": str(e), "reason": reason}

    return {"status": "stopped", "reason": reason}


def dump_status() -> Dict[str, Any]:
    with _dump_lock:
        if _dump_proc is None:
            running = False
            pid = None
        else:
            running = _dump_proc.poll() is None
            pid = _dump_proc.pid

        return {
            "running": running,
            "pid": pid,
            "stop_deadline": _stop_deadline,
            "rtl_index": _current_rtl_index,
            "default_rtl_index": DEFAULT_RTL_INDEX,
        }


def _watchdog_thread():
    global _dump_proc, _stop_deadline, _current_rtl_index

    while True:
        time.sleep(WATCHDOG_PERIOD)
        with _dump_lock:
            proc = _dump_proc
            deadline = _stop_deadline

        if proc is None:
            continue

        # If process died unexpectedly, clear state
        if proc.poll() is not None:
            with _dump_lock:
                _dump_proc = None
                _stop_deadline = None
                _current_rtl_index = None
            continue

        # Auto-stop when deadline passes
        if deadline is not None and time.time() > deadline:
            stop_dump1090(reason="auto_stop_deadline")


# Start watchdog
threading.Thread(target=_watchdog_thread, daemon=True).start()


# ----------------------- AIRCRAFT STATE ---------------------


def read_aircraft_raw() -> Dict[str, Any]:
    """
    Read the latest aircraft.json written by dump1090.
    """
    path = JSON_DIR / "aircraft.json"
    try:
        with path.open("r") as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        return {"now": time.time(), "messages": 0, "aircraft": []}
    except Exception as e:
        print(f"Error reading {path}: {e}", flush=True)
        return {"now": time.time(), "messages": 0, "aircraft": []}


def normalize_aircraft(ref_lat: float, ref_lon: float) -> Dict[str, Any]:
    """
    Read and normalize aircraft, attach range/bearing relative to ref_lat/lon,
    and discard stale tracks.
    """
    data = read_aircraft_raw()
    now = data.get("now", time.time())
    out_ac: List[Dict[str, Any]] = []

    for ac in data.get("aircraft", []):
        # Filter stale
        seen = ac.get("seen", 0)
        if isinstance(seen, (int, float)) and seen > STALE_SEC:
            continue

        # Only compute geo if we have a position
        lat = ac.get("lat")
        lon = ac.get("lon")
        if isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
            try:
                dist_nm = haversine_nm(ref_lat, ref_lon, lat, lon)
                brg = bearing_deg(ref_lat, ref_lon, lat, lon)
            except Exception:
                dist_nm = None
                brg = None
        else:
            dist_nm = None
            brg = None

        ac2 = dict(ac)
        ac2["dist_nm"] = dist_nm
        ac2["bearing_deg"] = brg
        out_ac.append(ac2)

    return {
        "now": now,
        "aircraft": out_ac,
    }


# ----------------------- FILTER ENDPOINTS -------------------


@app.route("/api/aircraft")
def api_aircraft() -> Response:
    lat_ref, lon_ref, gps_meta = get_reference_point()
    data = normalize_aircraft(lat_ref, lon_ref)
    status = dump_status()
    return jsonify(
        {
            "status": status,
            "ref": {"lat": lat_ref, "lon": lon_ref},
            "gps": gps_meta,
            "now": data["now"],
            "aircraft": data["aircraft"],
        }
    )


@app.route("/api/nearby")
def api_nearby() -> Response:
    """
    GET /api/nearby?radius_nm=30&max_alt_ft=10000&lat=..&lon=..
    """
    lat_ref, lon_ref, gps_meta = get_reference_point()
    radius_nm = float(request.args.get("radius_nm", "30"))
    max_alt_ft_str = request.args.get("max_alt_ft")
    max_alt_ft: Optional[float] = float(max_alt_ft_str) if max_alt_ft_str is not None else None

    data = normalize_aircraft(lat_ref, lon_ref)
    matches: List[Dict[str, Any]] = []

    for ac in data["aircraft"]:
        dist = ac.get("dist_nm")
        if dist is None:
            continue
        if dist > radius_nm:
            continue

        if max_alt_ft is not None:
            alt = ac.get("alt_baro") or ac.get("alt_geom") or ac.get("altitude")
            if not isinstance(alt, (int, float)) or alt > max_alt_ft:
                continue

        matches.append(ac)

    return jsonify(
        {
            "ref": {"lat": lat_ref, "lon": lon_ref},
            "gps": gps_meta,
            "radius_nm": radius_nm,
            "max_alt_ft": max_alt_ft,
            "count": len(matches),
            "aircraft": matches,
            "status": dump_status(),
        }
    )


@app.route("/api/inbound")
def api_inbound() -> Response:
    """
    Predict aircraft that will enter a buffer around reference point within time horizon.

    GET /api/inbound?time_min=10&buffer_nm=5&lat=..&lon=..

    - time_min: horizon in minutes (default 10)
    - buffer_nm: radius buffer (default 5 nm)
    """
    lat_ref, lon_ref, gps_meta = get_reference_point()
    time_min = float(request.args.get("time_min", "10"))
    time_sec = time_min * 60.0
    buffer_nm = float(request.args.get("buffer_nm", "5"))
    buffer_m = buffer_nm * 1852.0

    data = normalize_aircraft(lat_ref, lon_ref)
    inbound: List[Dict[str, Any]] = []

    for ac in data["aircraft"]:
        lat = ac.get("lat")
        lon = ac.get("lon")
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            continue

        # Ground speed in knots (dump1090: "gs" or "speed")
        speed_kt = ac.get("gs")
        if not isinstance(speed_kt, (int, float)):
            speed_kt = ac.get("speed")

        if not isinstance(speed_kt, (int, float)) or speed_kt <= 0:
            continue

        # Track (degrees, bearing)
        track = ac.get("track")
        if not isinstance(track, (int, float)):
            continue

        # Project current position into X/Y
        x, y = project_xy_m(lat_ref, lon_ref, lat, lon)
        # Velocity vector
        speed_m_s = speed_kt * 0.514444
        theta = math.radians(track)
        vx = speed_m_s * math.sin(theta)  # x = east
        vy = speed_m_s * math.cos(theta)  # y = north

        # Relative position r0 = (x, y), velocity v = (vx, vy)
        r0x, r0y = x, y
        v2 = vx * vx + vy * vy
        if v2 <= 0:
            continue

        # Time of closest approach
        t_ca = - (r0x * vx + r0y * vy) / v2

        # Clamp to [0, time_sec]
        t_clamped = max(0.0, min(time_sec, t_ca))

        # Position at closest approach
        xc = r0x + vx * t_clamped
        yc = r0y + vy * t_clamped
        d_min = math.hypot(xc, yc)

        if d_min <= buffer_m and 0.0 <= t_clamped <= time_sec:
            ac2 = dict(ac)
            ac2["time_to_closest_sec"] = t_clamped
            ac2["time_to_closest_min"] = t_clamped / 60.0
            ac2["closest_dist_nm"] = d_min / 1852.0
            ac2["dist_nm"] = ac.get("dist_nm")
            ac2["bearing_deg"] = ac.get("bearing_deg")
            inbound.append(ac2)

    inbound.sort(key=lambda a: a.get("time_to_closest_sec", 999999))

    return jsonify(
        {
            "ref": {"lat": lat_ref, "lon": lon_ref},
            "gps": gps_meta,
            "time_horizon_min": time_min,
            "buffer_nm": buffer_nm,
            "count": len(inbound),
            "aircraft": inbound,
            "status": dump_status(),
        }
    )


# ----------------------- CONTROL ENDPOINTS ------------------


@app.route("/api/status")
def api_status() -> Response:
    st = dump_status()
    return jsonify(
        {
            "status": st,
            "config": {
                "DEFAULT_RTL_INDEX": DEFAULT_RTL_INDEX,
                "JSON_DIR": str(JSON_DIR),
                "HOME_LAT": HOME_LAT,
                "HOME_LON": HOME_LON,
                "STALE_SEC": STALE_SEC,
                "GPS_URL": GPS_URL,
            },
            "last_gps": _last_gps,
        }
    )


@app.route("/api/start", methods=["POST"])
def api_start() -> Response:
    """
    POST /api/start
    Optional JSON body:
      {
        "duration_sec": 5,
        "rtl_index": 1
      }
    """
    payload: Dict[str, Any] = {}
    try:
        if request.data:
            payload = request.get_json(force=True, silent=True) or {}
    except Exception:
        payload = {}

    duration_sec = payload.get("duration_sec")
    rtl_index = payload.get("rtl_index")

    if duration_sec is not None:
        try:
            duration_sec = float(duration_sec)
        except ValueError:
            return jsonify({"status": "error", "error": "duration_sec must be numeric"}), 400

    if rtl_index is not None:
        try:
            rtl_index = int(rtl_index)
        except ValueError:
            return jsonify({"status": "error", "error": "rtl_index must be integer"}), 400

    result = start_dump1090(duration_sec=duration_sec, rtl_index=rtl_index)
    return jsonify(result)


@app.route("/api/stop", methods=["POST"])
def api_stop() -> Response:
    result = stop_dump1090(reason="manual_stop")
    return jsonify(result)


@app.route("/api/run_window", methods=["POST"])
def api_run_window() -> Response:
    """
    Convenience endpoint: run for X seconds, then stop automatically.

    POST /api/run_window
    JSON:
      {
        "duration_sec": 5,
        "rtl_index": 1
      }

    Returns immediately with status; you can separately poll /api/aircraft.
    """
    payload = request.get_json(force=True, silent=True) or {}
    duration_sec = payload.get("duration_sec", 5)
    rtl_index = payload.get("rtl_index")

    try:
        duration_sec = float(duration_sec)
    except ValueError:
        return jsonify({"status": "error", "error": "duration_sec must be numeric"}), 400

    if rtl_index is not None:
        try:
            rtl_index = int(rtl_index)
        except ValueError:
            return jsonify({"status": "error", "error": "rtl_index must be integer"}), 400

    result = start_dump1090(duration_sec=duration_sec, rtl_index=rtl_index)
    result["requested_duration_sec"] = duration_sec
    result["requested_rtl_index"] = rtl_index if rtl_index is not None else DEFAULT_RTL_INDEX
    return jsonify(result)


# ----------------------- SIMPLE WEB GUI ---------------------


INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ADS-B Radar</title>
<style>
  body {
    background-color: #111;
    color: #eee;
    font-family: sans-serif;
    margin: 0;
    padding: 0;
  }
  #topbar {
    padding: 10px;
    background: #222;
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
  }
  button {
    padding: 6px 10px;
    border-radius: 4px;
    border: none;
    cursor: pointer;
  }
  button.start {
    background: #2e7d32;
    color: white;
  }
  button.stop {
    background: #c62828;
    color: white;
  }
  #status {
    font-size: 12px;
    color: #ccc;
    flex: 1 1 auto;
  }
  #gpsinfo {
    font-size: 11px;
    color: #8bc34a;
    width: 100%;
  }
  #main {
    display: flex;
    flex-direction: row;
    padding: 10px;
    gap: 10px;
  }
  #radar-container {
    flex: 0 0 500px;
  }
  #radar {
    background: #000;
    border: 1px solid #555;
  }
  #list {
    flex: 1;
    font-size: 12px;
    max-height: 520px;
    overflow-y: auto;
    background: #181818;
    border: 1px solid #333;
    padding: 5px;
  }
  .ac-row {
    border-bottom: 1px solid #333;
    padding: 3px 0;
  }
  .ac-row span {
    display: inline-block;
    min-width: 60px;
  }
  #rtlbox {
    font-size: 12px;
    color: #eee;
  }
  #rtl_index_input {
    width: 40px;
  }
</style>
</head>
<body>
<div id="topbar">
  <button class="start" onclick="startShort()">Run 5s</button>
  <button class="start" onclick="startContinuous()">Start</button>
  <button class="stop" onclick="stopAdsb()">Stop</button>
  <span>Radius (nm):</span>
  <input id="radius" type="number" value="60" min="5" max="200" step="5" style="width:70px;">
  <span>Max Alt (ft):</span>
  <input id="maxAlt" type="number" value="40000" min="1000" max="60000" step="1000" style="width:80px;">
  <span id="rtlbox">RTL idx:
    <input
      id="rtl_index_input"
      type="number"
      value="0"
      min="0"
      step="1"
      oninput="rtlInputDirty = true;"
    >
  </span>
  <div id="status"></div>
  <div id="gpsinfo"></div>
</div>

<div id="main">
  <div id="radar-container">
    <canvas id="radar" width="500" height="500"></canvas>
  </div>
  <div id="list"></div>
</div>

<script>
let pollTimer = null;
let rtlInputDirty = false;

async function api(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

function getRtlIndexFromUI() {
  const el = document.getElementById('rtl_index_input');
  if (!el) return null;
  const v = el.value;
  if (v === '' || v === null || v === undefined) return null;
  const n = Number(v);
  if (Number.isNaN(n)) return null;
  return n;
}

async function startShort() {
  try {
    const rtl_index = getRtlIndexFromUI();
    const body = {duration_sec: 5};
    if (rtl_index !== null) body.rtl_index = rtl_index;

    const resp = await api('/api/run_window', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)
    });
    console.log('run_window', resp);
  } catch (e) {
    console.error(e);
  }
}

async function startContinuous() {
  try {
    const rtl_index = getRtlIndexFromUI();
    const body = {};
    if (rtl_index !== null) body.rtl_index = rtl_index;

    const resp = await api('/api/start', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(body)
    });
    console.log('start', resp);
  } catch (e) {
    console.error(e);
  }
}

async function stopAdsb() {
  try {
    const resp = await api('/api/stop', { method: 'POST' });
    console.log('stop', resp);
  } catch (e) {
    console.error(e);
  }
}

async function poll() {
  try {
    const radius = parseFloat(document.getElementById('radius').value || '60');
    const maxAlt = document.getElementById('maxAlt').value;
    const params = new URLSearchParams({radius_nm: radius.toString()});

    // We let server use GPS reference by default (no lat/lon params here).
    if (maxAlt) params.append('max_alt_ft', maxAlt.toString());

    const resp = await api('/api/nearby?' + params.toString());
    updateStatus(resp);
    updateGps(resp);
    drawRadar(resp);
    updateList(resp);
  } catch (e) {
    console.error(e);
    document.getElementById('status').textContent = 'Error: ' + e.message;
  }
}

function updateStatus(resp) {
  const s = resp.status || {};
  const running = s.running ? 'RUNNING' : 'STOPPED';
  const pid = s.pid || '-';
  const now = new Date().toLocaleTimeString();
  const count = resp.count ?? (resp.aircraft ? resp.aircraft.length : 0);

  const idx =
    (s.rtl_index !== null && s.rtl_index !== undefined)
      ? s.rtl_index
      : s.default_rtl_index;

  // Update status text
  document.getElementById('status').textContent =
    `[${now}] dump1090: ${running} (pid=${pid}, rtl_index=${idx}), aircraft=${count}`;

  // Only sync RTL index input while the user hasn't manually changed it
  const idxInput = document.getElementById('rtl_index_input');
  if (idxInput && !rtlInputDirty) {
    if (idx !== null && idx !== undefined) {
      idxInput.value = idx;
    }
  }
}

function updateGps(resp) {
  const gps = resp.gps || {};
  let txt = 'Ref: ';
  if (gps.source === 'gps') {
    txt += `GPS lat=${gps.lat?.toFixed ? gps.lat.toFixed(5) : gps.lat}, lon=${gps.lon?.toFixed ? gps.lon.toFixed(5) : gps.lon}`;
  } else if (gps.source === 'query') {
    txt += `Query lat=${gps.lat}, lon=${gps.lon}`;
  } else {
    txt += `Home lat=${gps.lat}, lon=${gps.lon}`;
  }
  if (gps.last_gps && gps.last_gps.error && gps.source !== 'gps') {
    txt += ` (GPS error: ${gps.last_gps.error})`;
  }
  document.getElementById('gpsinfo').textContent = txt;
}

function drawRadar(resp) {
  const canvas = document.getElementById('radar');
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;
  const cx = w / 2;
  const cy = h / 2;
  const radius_nm = parseFloat(document.getElementById('radius').value || '60');

  ctx.clearRect(0, 0, w, h);

  // Background
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, w, h);

  // Grid circles
  ctx.strokeStyle = '#0a3';
  ctx.lineWidth = 1;

  const maxR = Math.min(cx, cy) - 10;
  const rings = 4;
  for (let i = 1; i <= rings; i++) {
    const rr = maxR * (i / rings);
    ctx.beginPath();
    ctx.arc(cx, cy, rr, 0, 2 * Math.PI);
    ctx.stroke();

    ctx.fillStyle = '#0a3';
    ctx.font = '10px sans-serif';
    const labelNm = (radius_nm * i / rings).toFixed(0) + ' nm';
    ctx.fillText(labelNm, cx + rr + 4, cy);
  }

  // Bearing lines every 45 deg
  ctx.strokeStyle = '#063';
  for (let b = 0; b < 360; b += 45) {
    const rad = b * Math.PI / 180;
    const x = cx + maxR * Math.sin(rad);
    const y = cy - maxR * Math.cos(rad);
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(x, y);
    ctx.stroke();
  }

  // Ownship center
  ctx.fillStyle = '#0f0';
  ctx.beginPath();
  ctx.arc(cx, cy, 4, 0, 2 * Math.PI);
  ctx.fill();

  // Aircraft
  const ac = resp.aircraft || [];
  for (const a of ac) {
    const dist = a.dist_nm;
    const brg = a.bearing_deg;
    if (dist == null || brg == null) continue;

    const frac = Math.min(dist / radius_nm, 1.0);
    const r = maxR * frac;
    const rad = brg * Math.PI / 180;
    const x = cx + r * Math.sin(rad);
    const y = cy - r * Math.cos(rad);

    ctx.fillStyle = '#ff0';
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, 2 * Math.PI);
    ctx.fill();

    const callsign = a.callsign || a.flight || a.hex || '';
    const alt = a.alt_baro || a.alt_geom || a.altitude || '';
    ctx.fillStyle = '#fff';
    ctx.font = '10px sans-serif';
    ctx.fillText((callsign || '').trim(), x + 5, y - 2);
    if (alt) {
      ctx.fillText(alt + ' ft', x + 5, y + 9);
    }
  }
}

function updateList(resp) {
  const list = document.getElementById('list');
  const ac = resp.aircraft || [];
  ac.sort((a, b) => {
    const da = (a.dist_nm || 9999);
    const db = (b.dist_nm || 9999);
    return da - db;
  });

  let html = '';
  for (const a of ac) {
    const callsign = (a.callsign || a.flight || a.hex || '').trim();
    const alt = a.alt_baro || a.alt_geom || a.altitude || '';
    const dist = a.dist_nm != null ? a.dist_nm.toFixed(1) : '-';
    const brg = a.bearing_deg != null ? a.bearing_deg.toFixed(0) : '-';
    const gs = a.gs || a.speed || '';

    html += '<div class="ac-row">';
    html += '<span><b>' + callsign + '</b></span>';
    html += '<span>alt: ' + alt + '</span>';
    html += '<span>rng: ' + dist + ' nm</span>';
    html += '<span>brg: ' + brg + '°</span>';
    if (gs) html += '<span>gs: ' + gs + ' kt</span>';
    html += '</div>';
  }
  list.innerHTML = html;
}

// start polling
poll();
pollTimer = setInterval(poll, 2000);
</script>
</body>
</html>
"""


@app.route("/")
def index() -> Response:
    return Response(INDEX_HTML, mimetype="text/html")


# ----------------------- MAIN -------------------------------

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8086"))
    app.run(host="0.0.0.0", port=port, threaded=True)
