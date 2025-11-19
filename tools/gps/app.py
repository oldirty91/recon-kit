#!/usr/bin/env python3
import os
import time
import json
import threading
import socketserver
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

import serial           # pyserial
import pynmea2          # NMEA parser

# ---------------------------------------------------------------------
# Env / config
# ---------------------------------------------------------------------
UART_DEV   = os.getenv("UART_DEV", "/dev/serial0")
UART_BAUD  = int(os.getenv("UART_BAUD", "9600"))

HTTP_PORT  = int(os.getenv("HTTP_PORT", "8085"))       # JSON/SSE/UI
TCP_PORT   = int(os.getenv("NMEA_TCP_PORT", "10110"))  # Raw NMEA TCP stream

STALE_SEC  = int(os.getenv("STALE_SEC", "5"))
WRITE_JSON = os.getenv("WRITE_JSON", "/run/gps/position.json")

DEFAULT_LAT = float(os.getenv("DEFAULT_LAT", "41.39"))
DEFAULT_LON = float(os.getenv("DEFAULT_LON", "-71.58"))
DEFAULT_ALT = float(os.getenv("DEFAULT_ALT", "0.0"))

# ---------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------
state = {
    "lat": DEFAULT_LAT,
    "lon": DEFAULT_LON,
    "alt": DEFAULT_ALT,
    "fix": False,
    "sats": 0,
    "age_sec": None,
    "source": "default",
    "time_utc": None,
    "last_update": None,
}
lock = threading.Lock()

# Raw NMEA TCP clients
clients = set()
clients_lock = threading.Lock()


def _now():
    return datetime.now(timezone.utc)


def _write_snapshot():
    """Write current state to JSON file (if configured)."""
    if not WRITE_JSON:
        return
    try:
        os.makedirs(os.path.dirname(WRITE_JSON), exist_ok=True)
        with lock:
            snap = state.copy()
            if isinstance(snap.get("last_update"), datetime):
                snap["last_update"] = snap["last_update"].isoformat()
        with open(WRITE_JSON, "w") as f:
            json.dump(snap, f)
    except Exception:
        # best-effort only
        pass


def _mark_stale_if_needed():
    """Update age and mark fix stale if last position is too old."""
    with lock:
        lu = state.get("last_update")
        if not isinstance(lu, datetime):
            return
        age = (_now() - lu).total_seconds()
        state["age_sec"] = age
        if age > STALE_SEC:
            state["fix"] = False
            state["source"] = "stale"
    _write_snapshot()


# ---------------------------------------------------------------------
# Raw NMEA broadcast over TCP
# ---------------------------------------------------------------------
def _broadcast(line_bytes: bytes):
    """Broadcast raw NMEA line to all connected TCP clients."""
    to_drop = []
    with clients_lock:
        for c in clients:
            try:
                c.sendall(line_bytes)
            except Exception:
                to_drop.append(c)
        for d in to_drop:
            try:
                d.close()
            except Exception:
                pass
            clients.discard(d)


class NMEAHandler(socketserver.BaseRequestHandler):
    """Simple push-only TCP stream. We ignore client input entirely."""

    def setup(self):
        with clients_lock:
            clients.add(self.request)

    def handle(self):
        try:
            # Just block reading to detect disconnect; ignore all incoming data
            while self.request.recv(1):
                pass
        except Exception:
            pass

    def finish(self):
        with clients_lock:
            try:
                clients.remove(self.request)
            except KeyError:
                pass
        try:
            self.request.close()
        except Exception:
            pass


def start_tcp():
    if TCP_PORT <= 0:
        return
    srv = socketserver.ThreadingTCPServer(("0.0.0.0", TCP_PORT), NMEAHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    print(f"[gps] NMEA TCP server listening on :{TCP_PORT}", flush=True)


# ---------------------------------------------------------------------
# Serial reader: UART → state + raw broadcast
# ---------------------------------------------------------------------
def serial_reader():
    """Read UART; parse important sentences; update state; broadcast raw NMEA."""
    print(f"[gps] Opening UART {UART_DEV}@{UART_BAUD}", flush=True)
    while True:
        try:
            with serial.Serial(UART_DEV, UART_BAUD, timeout=1) as ser:
                print("[gps] UART opened", flush=True)
                while True:
                    raw = ser.readline()
                    if not raw:
                        _mark_stale_if_needed()
                        continue

                    # Broadcast raw NMEA to TCP clients
                    _broadcast(raw)

                    line = raw.decode("ascii", errors="ignore").strip()
                    if not line.startswith("$"):
                        continue

                    try:
                        msg = pynmea2.parse(line, check=True)
                        now = _now()
                        updated = False

                        with lock:
                            if isinstance(msg, pynmea2.types.talker.GGA):
                                # Position + quality
                                if msg.lat and msg.lon:
                                    state["lat"] = msg.latitude
                                    state["lon"] = msg.longitude
                                    updated = True
                                try:
                                    state["sats"] = int(msg.num_sats or 0)
                                except Exception:
                                    state["sats"] = 0
                                try:
                                    if msg.altitude:
                                        state["alt"] = float(msg.altitude)
                                except Exception:
                                    pass
                                try:
                                    fixq = int(msg.gps_qual or 0)
                                except Exception:
                                    fixq = 0
                                state["fix"] = fixq >= 1
                                state["source"] = "nmea-gga" if state["fix"] else "nmea-no-fix"
                                state["time_utc"] = now.isoformat()
                                state["last_update"] = now
                                state["age_sec"] = 0.0

                            elif isinstance(msg, pynmea2.types.talker.RMC):
                                # Position + validity
                                if msg.status == "A" and msg.lat and msg.lon:
                                    state["lat"] = msg.latitude
                                    state["lon"] = msg.longitude
                                    state["fix"] = True
                                    state["source"] = "nmea-rmc"
                                else:
                                    state["fix"] = False
                                    state["source"] = "nmea-rmc-v"
                                state["time_utc"] = now.isoformat()
                                state["last_update"] = now
                                state["age_sec"] = 0.0
                                updated = True

                        if updated:
                            _write_snapshot()

                    except Exception:
                        # tolerate parse errors / odd messages
                        pass

        except Exception as e:
            print(f"[gps] UART error: {e}; retrying in 1s", flush=True)
            time.sleep(1)


# ---------------------------------------------------------------------
# HTTP API + simple web UI
# ---------------------------------------------------------------------
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Recon-Kit GPS</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
    body {
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: #0b1120;
        color: #e5e7eb;
        margin: 0;
        padding: 0;
    }
    header {
        padding: 12px 16px;
        background: #020617;
        border-bottom: 1px solid #1f2937;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    h1 {
        font-size: 18px;
        margin: 0;
    }
    .badge {
        font-size: 12px;
        padding: 4px 8px;
        border-radius: 999px;
        border: 1px solid #22c55e33;
        color: #22c55e;
    }
    main {
        padding: 16px;
        max-width: 640px;
        margin: 0 auto;
    }
    .card {
        background: #020617;
        border-radius: 12px;
        padding: 16px;
        border: 1px solid #1f2937;
        box-shadow: 0 10px 30px rgba(0,0,0,0.4);
    }
    .row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 8px;
        font-size: 14px;
    }
    .label {
        color: #9ca3af;
    }
    .value {
        font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
    }
    .status {
        margin-top: 12px;
        font-size: 12px;
        color: #9ca3af;
    }
    .fix-true {
        color: #22c55e;
    }
    .fix-false {
        color: #f97316;
    }
</style>
</head>
<body>
<header>
    <h1>Recon-Kit GPS</h1>
    <div class="badge" id="badge">connecting…</div>
</header>
<main>
    <div class="card">
        <div class="row">
            <div class="label">Latitude</div>
            <div class="value" id="lat">–</div>
        </div>
        <div class="row">
            <div class="label">Longitude</div>
            <div class="value" id="lon">–</div>
        </div>
        <div class="row">
            <div class="label">Altitude (m)</div>
            <div class="value" id="alt">–</div>
        </div>
        <div class="row">
            <div class="label">Fix</div>
            <div class="value" id="fix">–</div>
        </div>
        <div class="row">
            <div class="label">Satellites</div>
            <div class="value" id="sats">–</div>
        </div>
        <div class="row">
            <div class="label">Source</div>
            <div class="value" id="source">–</div>
        </div>
        <div class="row">
            <div class="label">Age (s)</div>
            <div class="value" id="age">–</div>
        </div>
        <div class="row">
            <div class="label">Last update</div>
            <div class="value" id="last_update">–</div>
        </div>
        <div class="status" id="status">
            Waiting for data…
        </div>
    </div>
</main>

<script>
(function() {
    const elLat = document.getElementById("lat");
    const elLon = document.getElementById("lon");
    const elAlt = document.getElementById("alt");
    const elFix = document.getElementById("fix");
    const elSats = document.getElementById("sats");
    const elSource = document.getElementById("source");
    const elAge = document.getElementById("age");
    const elLast = document.getElementById("last_update");
    const elStatus = document.getElementById("status");
    const elBadge = document.getElementById("badge");

    function setBadge(text, ok) {
        elBadge.textContent = text;
        elBadge.style.borderColor = ok ? "#22c55e55" : "#f9731655";
        elBadge.style.color = ok ? "#22c55e" : "#f97316";
    }

    function updateView(snap) {
        elLat.textContent = snap.lat != null ? snap.lat.toFixed(6) : "–";
        elLon.textContent = snap.lon != null ? snap.lon.toFixed(6) : "–";
        elAlt.textContent = snap.alt != null ? snap.alt.toFixed(1) : "–";
        elSats.textContent = snap.sats != null ? String(snap.sats) : "–";
        elSource.textContent = snap.source || "–";

        const hasFix = !!snap.fix;
        elFix.textContent = hasFix ? "LOCKED" : "NO FIX";
        elFix.className = "value " + (hasFix ? "fix-true" : "fix-false");

        if (typeof snap.age_sec === "number") {
            elAge.textContent = snap.age_sec.toFixed(1);
        } else {
            elAge.textContent = "–";
        }

        elLast.textContent = snap.last_update || "–";

        let msg = hasFix ? "GPS fix acquired." : "No valid fix yet.";
        if (typeof snap.age_sec === "number" && snap.age_sec > 5) {
            msg += " (stale)";
        }
        elStatus.textContent = msg;
    }

    function startStream() {
        setBadge("connecting…", false);
        const ev = new EventSource("/stream");
        ev.onopen = function() {
            setBadge("live", true);
        };
        ev.onerror = function() {
            setBadge("disconnected", false);
        };
        ev.onmessage = function(evt) {
            try {
                const snap = JSON.parse(evt.data);
                updateView(snap);
            } catch (e) {
                console.error("Bad GPS payload", e);
            }
        };
    }

    startStream();
})();
</script>
</body>
</html>
"""


class HTTP(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # ensure SSE friendliness

    def _send_bytes(self, code, payload: bytes, content_type: str):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, code, payload):
        b = json.dumps(payload).encode("utf-8")
        self._send_bytes(code, b, "application/json")

    def log_message(self, fmt, *args):
        # Keep logs quiet; comment this out if you want HTTP logs.
        return

    def do_GET(self):
        # Simple UI
        if self.path == "/" or self.path.startswith("/index"):
            self._send_bytes(200, HTML_PAGE.encode("utf-8"), "text/html; charset=utf-8")
            return

        # Health check
        if self.path.startswith("/health"):
            with lock:
                seen = isinstance(state.get("last_update"), datetime)
            self._json(200, {"ok": True, "has_seen_data": seen})
            return

        # JSON position snapshot
        if self.path.startswith("/position"):
            _mark_stale_if_needed()
            with lock:
                snap = state.copy()
                if isinstance(snap.get("last_update"), datetime):
                    snap["last_update"] = snap["last_update"].isoformat()
            self._json(200, snap)
            return

        # Server-Sent Events stream for live updates
        if self.path.startswith("/stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            try:
                while True:
                    _mark_stale_if_needed()
                    with lock:
                        snap = state.copy()
                        if isinstance(snap.get("last_update"), datetime):
                            snap["last_update"] = snap["last_update"].isoformat()
                    data = json.dumps(snap).encode("utf-8")
                    self.wfile.write(b"data: " + data + b"\n\n")
                    self.wfile.flush()
                    time.sleep(1)
            except Exception:
                # client disconnected / error
                return

        # Fallback 404
        self._json(404, {"error": "not found"})


def start_http():
    httpd = HTTPServer(("0.0.0.0", HTTP_PORT), HTTP)
    print(f"[gps] HTTP server listening on :{HTTP_PORT}", flush=True)
    httpd.serve_forever()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    _write_snapshot()

    # Serial reader thread
    t_serial = threading.Thread(target=serial_reader, daemon=True)
    t_serial.start()

    # NMEA TCP stream
    start_tcp()

    # HTTP server (blocking)
    start_http()


if __name__ == "__main__":
    main()
