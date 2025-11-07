FROM python:3.11-slim

WORKDIR /opt/fft_viewer

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

ENV HUB_URL=http://hub:8080
ENV HUB_HOST=hub
ENV FFT_PORT=8090

CMD ["python", "app.py"]
