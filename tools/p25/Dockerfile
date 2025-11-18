# FROM debian:bullseye

# ENV DEBIAN_FRONTEND=noninteractive

# # ---- Base deps: GNU Radio 3.8 + build tooling + RTL-SDR ----
# RUN apt-get update && \
#     apt-get install -y \
#         git \
#         sudo \
#         python3 \
#         python3-pip \
#         python3-numpy \
#         python3-scipy \
#         gnuradio \
#         gnuradio-dev \
#         g++ \
#         cmake \
#         libboost-all-dev \
#         libuhd-dev \
#         rtl-sdr \
#         libusb-1.0-0 \
#         libitpp-dev \
#         libpcap0.8-dev \
#         libsndfile1-dev \
#         && apt-get clean && \
#         rm -rf /var/lib/apt/lists/*

# # Quick sanity: should be 3.8.x here
# RUN gnuradio-config-info --version || echo "gnuradio-config-info missing?"

# # ---- Pull OP25 and build for GNU Radio 3.8 (branch gr38) ----
# WORKDIR /opt
# RUN git clone https://github.com/boatbod/op25.git

# WORKDIR /opt/op25
# # branch specifically for gnuradio 3.8 or earlier
# RUN git checkout gr38

# # Run the official installer; it will:
# #  - build gr-op25
# #  - build gr-op25_repeater
# #  - install python bindings (gnuradio.op25)
# RUN chmod +x ./install.sh && ./install.sh

# # Make sure newly installed libs are visible
# RUN ldconfig

# # ---- Runtime layout ----
# RUN mkdir -p /config \
#     && mkdir -p /var/log/op25


# # ---- Our HTTP control + mini GUI ----
# WORKDIR /opt/p25
# COPY app.py /opt/p25/app.py

# RUN pip3 install --no-cache-dir flask

# EXPOSE 9005 8765

# # Default command: run the Flask control server
# CMD ["python3", "app.py"]

FROM debian:bullseye

ENV DEBIAN_FRONTEND=noninteractive

# ---- Base deps: GNU Radio 3.8-ish + build tooling + RTL-SDR ----
RUN apt-get update && \
    apt-get install -y \
        git \
        sudo \
        python3 \
        python3-pip \
        python3-numpy \
        python3-scipy \
        python3-dev \
        gnuradio \
        gnuradio-dev \
        gr-osmosdr \
        g++ \
        cmake \
        libboost-all-dev \
        libuhd-dev \
        rtl-sdr \
        libusb-1.0-0 \
        libitpp-dev \
        libpcap0.8-dev \
        libsndfile1-dev \
        swig \
        && apt-get clean && \
        rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir waitress

# Just to see what we actually got
RUN gnuradio-config-info --version || echo "gnuradio-config-info missing?"

# ---- Pull and install OP25 (boatbod) ----
WORKDIR /opt
RUN git clone https://github.com/boatbod/op25.git

WORKDIR /opt/op25
# branch for pre-3.9 gnuradio
RUN git checkout gr38

# Run the official installer; this is supposed to build:
#  - gr-op25 (core)
#  - gr-op25_repeater (rx.py, web UI, etc)
RUN chmod +x ./install.sh && ./install.sh

# Make sure libs are visible
RUN ldconfig

# ---- Runtime layout ----
RUN mkdir -p /config \
    && mkdir -p /var/log/op25

# We'll mount ./tools/p25/config -> /config from docker-compose
# so we don't COPY trunk.tsv/riscon.tsv here.

# ---- Our HTTP control + mini GUI ----
WORKDIR /opt/p25
COPY app.py /opt/p25/app.py

RUN pip3 install --no-cache-dir flask

EXPOSE 9005 8765

CMD ["python3", "app.py"]
