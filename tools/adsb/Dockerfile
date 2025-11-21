FROM python:3.11-slim

# Install build deps + rtl-sdr runtime + ncurses for dump1090
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential \
        pkg-config \
        ca-certificates \
        libusb-1.0-0-dev \
        librtlsdr-dev \
        rtl-sdr \
        libncurses-dev \
    && rm -rf /var/lib/apt/lists/*

# Build dump1090 (FlightAware) from source
RUN git clone https://github.com/flightaware/dump1090.git /opt/dump1090 && \
    cd /opt/dump1090 && \
    make -j"$(nproc)" && \
    install -m 0755 dump1090 /usr/local/bin/dump1090 && \
    rm -rf /opt/dump1090

WORKDIR /opt/adsb

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Use the locally-built dump1090 binary by default
ENV PORT=8080 \
    DUMP1090_BIN=/usr/local/bin/dump1090 \
    RTL_DEVICE_INDEX=0 \
    JSON_DIR=/run/dump1090 \
    HOME_LAT=0.0 \
    HOME_LON=0.0 \
    STALE_SEC=60 \
    GPS_URL=http://gps:8085/position

EXPOSE 8080

CMD ["python", "app.py"]
