#!/usr/bin/env python3
import os
import socket
import json
import time

HUB_HOST = os.getenv("HUB_HOST", "hub")
IQ_PORT = int(os.getenv("SDR_IQ_PORT", "5550"))
CTL_PORT = int(os.getenv("SDR_CTL_PORT", "6000"))

def send_control(cmd: dict):
    with socket.create_connection((HUB_HOST, CTL_PORT), timeout=5) as s:
        s.sendall((json.dumps(cmd) + "\n").encode("utf-8"))
        resp = s.recv(4096)
    print("[example-tool] control response:", resp.decode().strip())


def main():
    print("[example-tool] Setting SDR config via control port...")
    send_control(
        {
            "cmd": "set_config",
            "freq": 1090000000,   # 1.09 GHz, example
            "samp_rate": 2400000, # 2.4 Msps
            "gain": 40,
        }
    )

    print(f"[example-tool] Connecting to IQ port {IQ_PORT}...")
    with socket.create_connection((HUB_HOST, IQ_PORT), timeout=5) as s:
        # Read some IQ samples
        data = s.recv(16384)
        print(f"[example-tool] Got {len(data)} bytes of IQ data")
        time.sleep(1)

    print("[example-tool] Done.")

if __name__ == "__main__":
    main()
