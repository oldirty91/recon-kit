FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install ALSA tools so we can play to /dev/snd
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        alsa-utils \
        libasound2 \
        libasound2-dev \
        build-essential && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /opt/mixer

# Your main mixer script is app.py in this project
COPY app.py /opt/mixer/app.py

# Default HTTP port
ENV HTTP_PORT=8087

EXPOSE 8087

CMD ["python", "/opt/mixer/app.py"]
