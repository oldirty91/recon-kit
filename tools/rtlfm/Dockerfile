FROM debian:bookworm-slim

# Install rtl_fm (rtl-sdr), aplay (alsa-utils), Python, and Flask from Debian packages
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    rtl-sdr \
    alsa-utils \
    python3 \
    python3-flask \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# App lives here
WORKDIR /opt/app

# Copy our Flask app with inline HTML UI
COPY app.py /opt/app/app.py

EXPOSE 8080

CMD ["python3", "app.py"]
