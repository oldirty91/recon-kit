FROM python:3.11-slim

WORKDIR /opt/fft_viewer

# System deps (if you want, can add more later)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Where to find the hub inside the compose network
ENV HUB_URL=http://hub:8080
ENV HUB_HOST=hub
ENV FFT_PORT=8090

CMD ["python", "app.py"]
