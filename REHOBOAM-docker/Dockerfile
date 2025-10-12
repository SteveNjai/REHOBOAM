# ---------------------------
# Base image
# ---------------------------
FROM ubuntu:24.04

# Non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Africa/Nairobi

# ---------------------------
# Install system dependencies
# ---------------------------
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-pip \
        wget \
        curl \
        xvfb \
        git \
        ca-certificates \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ---------------------------
# Set working directory
# ---------------------------
WORKDIR /opt/rehoboam

# ---------------------------
# Copy local REHOBOAM repository
# ---------------------------
COPY REHOBOAM/ /opt/rehoboam/

# ---------------------------
# Create Linux-compatible Python virtual environment
# ---------------------------
WORKDIR /opt/rehoboam/rehoboam-tools
RUN python3 -m venv venv && \
    . venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir mt5linux  # Add this for MT5 Linux wrapper

# ---------------------------
# Install MetaTrader 5 for Linux
# ---------------------------
WORKDIR /opt/rehoboam
RUN wget -q https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5linux.sh && \
    chmod +x mt5linux.sh && \
    ./mt5linux.sh || true && \
    rm -f mt5linux.sh

# ---------------------------
# Install Windows Python under Wine (for MT5 Python integration)
# ---------------------------
# Download and install Python 3.12 Windows installer using Wine (matches Ubuntu Python version)
# Wine prefix from mt5linux.sh is /root/.mt5
RUN cd /tmp && \
    wget -q https://www.python.org/ftp/python/3.12.7/python-3.12.7-amd64.exe && \
    xvfb-run -a WINEPREFIX=/root/.mt5 wine python-3.12.7-amd64.exe /quiet InstallAllUsers=1 PrependPath=1 TargetDir=C:\\Python312 && \
    rm python-3.12.7-amd64.exe

# Install MetaTrader5 and rpyc in the Wine Python environment (required for the bridge)
RUN WINEPREFIX=/root/.mt5 wine /root/.mt5/drive_c/Python312/python.exe -m pip install --upgrade pip && \
    WINEPREFIX=/root/.mt5 wine /root/.mt5/drive_c/Python312/python.exe -m pip install MetaTrader5 rpyc

# ---------------------------
# Make runner script executable
# ---------------------------
RUN chmod +x rehoboam-tools/rehoboam_tools_runner.sh && \
    find /opt/rehoboam -type f -name "*.pyc" -delete && \
    find /opt/rehoboam -type d -name "__pycache__" -exec rm -rf {} + && \
    apt-get autoremove -y && apt-get clean && \
    rm -rf /tmp/* /var/tmp/*

# ---------------------------
# Final working directory & entrypoint
# ---------------------------
WORKDIR /opt/rehoboam/rehoboam-tools
# Use a custom entrypoint script to start MT5, the mt5linux server, and then your runner
COPY entrypoint.sh /opt/rehoboam/rehoboam-tools/entrypoint.sh
RUN chmod +x entrypoint.sh
ENTRYPOINT ["/opt/rehoboam/rehoboam-tools/entrypoint.sh"]