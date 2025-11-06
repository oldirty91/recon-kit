FROM python:3.11-slim

# Install rtl-sdr tools (rtl_tcp, rtl_test, etc.) and usb utilities
RUN apt-get update && \
    apt-get install -y --no-install-recommends rtl-sdr usbutils && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt/recon_kit

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["python", "hub.py"]
