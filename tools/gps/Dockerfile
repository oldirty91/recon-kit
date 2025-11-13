FROM python:3.11-slim

# Install minimal OS deps (optional but nice to have)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY app.py .

# Ensure unbuffered logs
ENV PYTHONUNBUFFERED=1

CMD ["python", "-u", "app.py"]
