FROM debian:bookworm-slim

# Install rtl_fm (rtl-sdr), aplay (alsa-utils), Python, Flask, pip, and libatomic
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    rtl-sdr \
    alsa-utils \
    python3 \
    python3-flask \
    python3-pip \
    ca-certificates \
    libatomic1 \
    && rm -rf /var/lib/apt/lists/*

# Install Vosk for STT
RUN pip3 install --no-cache-dir --break-system-packages vosk

WORKDIR /opt/app
COPY app.py /opt/app/app.py

EXPOSE 8080
CMD ["python3", "app.py"]
