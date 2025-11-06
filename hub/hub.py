#!/usr/bin/env python3
import os
import time
import json
import socket
import threading
import socketserver
import subprocess
from dataclasses import dataclass
from typing import Dict, Optional

from flask import Flask, jsonify

# ----------------- Config / env -----------------

IQ_BASE_PORT = int(os.getenv("IQ_BASE_PORT", "5550"))     # external relay ports
RTL_BASE_PORT = int(os.getenv("RTL_BASE_PORT", "4550"))   # internal rtl_tcp ports
CTL_BASE_PORT = int(os.getenv("CTL_BASE_PORT", "6000"))
MAX_SDRS = int(os.getenv("MAX_SDRS", "8"))
THROUGHPUT_INTERVAL_SEC = 2.0  # rolling window for throughput

app = Flask(__name__)
sdrs: Dict[str, "SDRDevice"] = {}


@dataclass
class SDRDevice:
    index: int
    serial: str
    iq_port: int          # external relay port (what tools connect to)
    control_port: int
    rtl_port: int         # internal rtl_tcp port (hub only)
    center_freq: Optional[int] = None
    sample_rate: Optional[int] = None
    gain: Optional[int] = None
    status: str = "init"
    last_data_time: float = 0.0
    throughput_kbps: float = 0.0
    rtl_tcp_proc: Optional[subprocess.Popen] = None

    def __post_init__(self):
        # General lock + throughput window
        self.lock = threading.Lock()
        self._bytes_window = 0
        self._last_throughput_ts = time.time()

        # IQ relay state (fan-out to multiple tool clients)
        self.relay_clients = set()
        self.relay_lock = threading.Lock()


# ----------------- RTL-SDR discovery -----------------


def rtl_device_exists(idx: int) -> bool:
    try:
        out = subprocess.check_output(
            ["rtl_test", "-d", str(idx), "-t"],
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
    except Exception:
        return False
    return "No supported devices found" not in out


def discover_sdrs():
    print("[hub] Discovering RTL-SDR devices with rtl_test...")
    for idx in range(MAX_SDRS):
        if not rtl_device_exists(idx):
            continue
        serial = f"SDR{idx:03d}"
        iq_port = IQ_BASE_PORT + idx       # exposed relay port
        rtl_port = RTL_BASE_PORT + idx     # internal rtl_tcp port
        control_port = CTL_BASE_PORT + idx
        dev = SDRDevice(
            index=idx,
            serial=serial,
            iq_port=iq_port,
            control_port=control_port,
            rtl_port=rtl_port,
        )
        dev.status = "discovered"
        dev.last_data_time = time.time()
        sdrs[serial] = dev
        print(
            f"[hub] Found RTL-SDR index={idx} as {serial}, "
            f"rtl_tcp={rtl_port}, IQ={iq_port}, CTL={control_port}"
        )
    if not sdrs:
        print("[hub] WARNING: No RTL-SDR devices detected.")


# ----------------- rtl_tcp process control -----------------


def build_rtl_tcp_cmd(dev: SDRDevice):
    # rtl_tcp binds ONLY on localhost, on internal rtl_port
    cmd = ["rtl_tcp", "-d", str(dev.index), "-a", "127.0.0.1", "-p", str(dev.rtl_port)]
    if dev.center_freq:
        cmd += ["-f", str(dev.center_freq)]
    if dev.sample_rate:
        cmd += ["-s", str(dev.sample_rate)]
    if dev.gain is not None:
        cmd += ["-g", str(dev.gain)]
    return cmd


def start_rtl_tcp(dev: SDRDevice):
    with dev.lock:
        if dev.rtl_tcp_proc and dev.rtl_tcp_proc.poll() is None:
            return
        cmd = build_rtl_tcp_cmd(dev)
        print(f"[hub] Starting rtl_tcp for {dev.serial}: {' '.join(cmd)}")
        dev.status = "starting"
        dev.last_data_time = time.time()
        dev._bytes_window = 0
        dev._last_throughput_ts = time.time()
        dev.rtl_tcp_proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        dev.status = "running"


def stop_rtl_tcp(dev: SDRDevice):
    with dev.lock:
        if dev.rtl_tcp_proc and dev.rtl_tcp_proc.poll() is None:
            print(f"[hub] Stopping rtl_tcp for {dev.serial}")
            dev.rtl_tcp_proc.terminate()
            try:
                dev.rtl_tcp_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                dev.rtl_tcp_proc.kill()
        dev.rtl_tcp_proc = None
        dev.status = "stopped"
        dev.throughput_kbps = 0.0


def restart_rtl_tcp(dev: SDRDevice):
    print(f"[hub] Restarting rtl_tcp for {dev.serial}")
    stop_rtl_tcp(dev)
    time.sleep(0.5)
    start_rtl_tcp(dev)


# ----------------- IQ relay (fan-out) -----------------


def iq_relay_accept_loop(dev: SDRDevice):
    """
    Accept multiple tool clients on dev.iq_port and register them as consumers
    of the IQ stream. The hub will read from rtl_tcp once and broadcast to all.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", dev.iq_port))
    srv.listen(5)
    srv.settimeout(1.0)
    print(f"[hub] IQ relay listening for {dev.serial} on {dev.iq_port}")

    try:
        while True:
            try:
                client, addr = srv.accept()
                client.setblocking(True)  # keep it simple; fan-out will handle slow clients
                print(f"[hub] {dev.serial} relay: client connected from {addr}")
                with dev.relay_lock:
                    dev.relay_clients.add(client)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"[hub] {dev.serial} relay accept error: {e}")
    finally:
        print(f"[hub] IQ relay for {dev.serial} shutting down")
        srv.close()
        with dev.relay_lock:
            for c in list(dev.relay_clients):
                try:
                    c.close()
                except Exception:
                    pass
            dev.relay_clients.clear()


def iq_relay_reader_loop(dev: SDRDevice):
    """
    Maintain a single connection to rtl_tcp on dev.rtl_port and broadcast IQ
    data to all connected clients on dev.iq_port.
    """
    rtl_sock = None

    while True:
        # Ensure rtl_tcp process is running
        proc = dev.rtl_tcp_proc
        if proc is not None and proc.poll() is not None:
            print(
                f"[hub] rtl_tcp for {dev.serial} exited with {proc.returncode}, restarting..."
            )
            start_rtl_tcp(dev)
            time.sleep(1.0)
            continue
        elif proc is None:
            start_rtl_tcp(dev)
            time.sleep(1.0)
            continue

        # Connect to rtl_tcp if needed
        if rtl_sock is None:
            try:
                print(
                    f"[hub] {dev.serial} relay: connecting to rtl_tcp on 127.0.0.1:{dev.rtl_port}"
                )
                rtl_sock = socket.create_connection(("127.0.0.1", dev.rtl_port), timeout=5)
                rtl_sock.settimeout(1.0)
                with dev.lock:
                    # Once we have data, we'll mark as "streaming"
                    if dev.status == "running":
                        dev.status = "streaming"
            except Exception as e:
                print(f"[hub] {dev.serial} relay: failed to connect to rtl_tcp: {e}")
                rtl_sock = None
                with dev.lock:
                    dev.status = "error"
                time.sleep(1.0)
                continue

        # Read IQ and broadcast
        try:
            data = rtl_sock.recv(4096)
            if not data:
                print(f"[hub] {dev.serial} relay: rtl_tcp closed connection, reconnecting")
                rtl_sock.close()
                rtl_sock = None
                with dev.lock:
                    dev.status = "running"  # we'll try to re-establish
                time.sleep(0.5)
                continue
        except socket.timeout:
            # No data in this window; just loop again
            continue
        except Exception as e:
            print(f"[hub] {dev.serial} relay: recv error: {e}")
            try:
                rtl_sock.close()
            except Exception:
                pass
            rtl_sock = None
            with dev.lock:
                dev.status = "error"
            time.sleep(0.5)
            continue

        now = time.time()
        # Update stats from the rtl_tcp stream
        with dev.lock:
            dev.last_data_time = now
            if dev.status in ("running", "error", "dormant"):
                dev.status = "streaming"

            dev._bytes_window += len(data)
            if now - dev._last_throughput_ts >= THROUGHPUT_INTERVAL_SEC:
                elapsed = now - dev._last_throughput_ts
                kbps = (dev._bytes_window / 1024.0) / elapsed
                dev.throughput_kbps = kbps
                dev._bytes_window = 0
                dev._last_throughput_ts = now

        # Fan-out to connected clients
        to_drop = []
        with dev.relay_lock:
            for c in list(dev.relay_clients):
                try:
                    c.sendall(data)
                except Exception:
                    to_drop.append(c)
            for c in to_drop:
                print(f"[hub] {dev.serial} relay: dropping client")
                dev.relay_clients.discard(c)
                try:
                    c.close()
                except Exception:
                    pass


def launch_iq_relays():
    for dev in sdrs.values():
        # Start rtl_tcp once; reader loop will keep it alive
        start_rtl_tcp(dev)
        # Start acceptor and reader threads
        threading.Thread(
            target=iq_relay_accept_loop, args=(dev,), daemon=True
        ).start()
        threading.Thread(
            target=iq_relay_reader_loop, args=(dev,), daemon=True
        ).start()


# ----------------- Control server -----------------


def make_control_handler(dev: SDRDevice):
    class ControlHandler(socketserver.StreamRequestHandler):
        def handle(self_inner):
            try:
                line = self_inner.rfile.readline().decode().strip()
                if not line:
                    return
                req = json.loads(line)
                cmd = req.get("cmd")

                if cmd == "get_status":
                    resp = sdr_to_dict(dev)

                elif cmd == "set_config":
                    freq = req.get("freq")
                    sr = req.get("samp_rate")
                    gain = req.get("gain")
                    with dev.lock:
                        if freq:
                            dev.center_freq = int(freq)
                        if sr:
                            dev.sample_rate = int(sr)
                        if gain is not None:
                            dev.gain = int(gain)
                    # Restart rtl_tcp so new params take effect
                    restart_rtl_tcp(dev)
                    resp = {"ok": True, "status": sdr_to_dict(dev)}

                elif cmd == "restart":
                    restart_rtl_tcp(dev)
                    resp = {"ok": True, "status": sdr_to_dict(dev)}

                else:
                    resp = {"ok": False, "error": "unknown_cmd"}
            except Exception as e:
                resp = {"ok": False, "error": str(e)}
            self_inner.wfile.write((json.dumps(resp) + "\n").encode())

    return ControlHandler


def launch_control_servers():
    for dev in sdrs.values():
        handler_cls = make_control_handler(dev)
        server = socketserver.ThreadingTCPServer(("0.0.0.0", dev.control_port), handler_cls)
        server.daemon_threads = True
        print(f"[hub] Starting control server for {dev.serial} on {dev.control_port}")
        threading.Thread(target=server.serve_forever, daemon=True).start()


# ----------------- Web GUI / API -----------------


def sdr_to_dict(dev: SDRDevice):
    with dev.lock:
        return {
            "index": dev.index,
            "serial": dev.serial,
            "iq_port": dev.iq_port,
            "control_port": dev.control_port,
            "center_freq": dev.center_freq,
            "sample_rate": dev.sample_rate,
            "gain": dev.gain,
            "status": dev.status,
            "last_data_age_sec": time.time() - dev.last_data_time if dev.last_data_time else None,
            "throughput_kbps": round(dev.throughput_kbps, 1),
        }


@app.route("/api/sdrs")
def api_sdrs():
    return jsonify([sdr_to_dict(dev) for dev in sdrs.values()])


@app.route("/api/restart/<serial>", methods=["POST"])
def api_restart(serial):
    dev = sdrs.get(serial)
    if not dev:
        return jsonify({"ok": False, "error": "not_found"}), 404
    restart_rtl_tcp(dev)
    return jsonify({"ok": True, "status": sdr_to_dict(dev)})


@app.route("/")
def index():
    html = """<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>RECON-KIT - SDR Status</title>
<style>
body{font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0;padding:20px;background:#111827;color:#e5e7eb;}
table{border-collapse:collapse;width:100%;background:#020617;border-radius:8px;overflow:hidden;}
th,td{padding:8px 10px;border-bottom:1px solid #1f2937;font-size:14px;text-align:left;}
th{background:#0b1120;font-weight:600;}
.status-pill{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;}
.status-streaming{background:#16a34a33;color:#bbf7d0;}
.status-running{background:#2563eb33;color:#bfdbfe;}
.status-dormant{background:#f9731633;color:#fed7aa;}
.status-stopped,.status-error{background:#b91c1c33;color:#fecaca;}
button{padding:4px 10px;border-radius:999px;border:none;cursor:pointer;font-size:12px;background:#1f2937;color:#e5e7eb;}
button:hover{background:#4b5563;}
.mono{font-family:ui-monospace,Menlo,Consolas,monospace;}
</style>
</head>
<body>
<h1>RECON-KIT</h1>
<table id="sdr-table">
<thead>
<tr>
<th>Index</th><th>Serial</th><th>IQ Port</th><th>Control Port</th>
<th>Freq (Hz)</th><th>Rate (Hz)</th><th>Gain</th>
<th>Status</th><th>Last Data (s ago)</th><th>Throughput (kB/s)</th><th>Actions</th>
</tr>
</thead><tbody></tbody></table>
<script>
function statusClass(s){s=(s||"").toLowerCase();if(s==="streaming")return"status-streaming";
if(s==="running")return"status-running";if(s==="dormant")return"status-dormant";
if(s==="stopped")return"status-stopped";return"status-error";}
async function fetchSDRs(){
 const res=await fetch("/api/sdrs");if(!res.ok)return;
 const data=await res.json();const tb=document.querySelector("#sdr-table tbody");tb.innerHTML="";
 data.forEach(s=>{
  const tr=document.createElement("tr");
  [s.index,s.serial,s.iq_port,s.control_port,s.center_freq||"-",s.sample_rate||"-",s.gain??"-"]
   .forEach(v=>{const td=document.createElement("td");td.textContent=v;if(typeof v==="number"&&v>1e6)td.classList.add("mono");tr.appendChild(td);});
  const st=document.createElement("td");const pill=document.createElement("span");
  pill.textContent=s.status||"unknown";pill.className="status-pill "+statusClass(s.status);st.appendChild(pill);tr.appendChild(st);
  const age=document.createElement("td");age.textContent=s.last_data_age_sec!=null? s.last_data_age_sec.toFixed(1):"-";tr.appendChild(age);
  const thr=document.createElement("td");thr.textContent=s.throughput_kbps!=null? s.throughput_kbps.toFixed(1):"-";tr.appendChild(thr);
  const act=document.createElement("td");const btn=document.createElement("button");
  btn.textContent="Restart";btn.onclick=async()=>{await fetch(`/api/restart/${encodeURIComponent(s.serial)}`,{method:"POST"});fetchSDRs();};
  act.appendChild(btn);tr.appendChild(act);tb.appendChild(tr);
 });}
fetchSDRs();setInterval(fetchSDRs,2000);
</script>
</body></html>"""
    return html


# ----------------- Main -----------------

def main():
    discover_sdrs()
    if sdrs:
        launch_control_servers()
        launch_iq_relays()
    else:
        print("[hub] No SDRs found. Web GUI will still run.")
    app.run(host="0.0.0.0", port=8080, threaded=True)


if __name__ == "__main__":
    main()
