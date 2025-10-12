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
    pip install --no-cache-dir -r requirements.txt

# ---------------------------
# Install MetaTrader 5 for Linux
# ---------------------------
WORKDIR /opt/rehoboam
RUN wget -q https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5linux.sh && \
    chmod +x mt5linux.sh && \
    ./mt5linux.sh || true && \
    rm -f mt5linux.sh

# ---------------------------
# Link MT5 Python packages to your venv
# ---------------------------
# This assumes MT5 installs Python packages under /opt/mt5/terminal64/...
# Adjust path if your mt5linux.sh installs elsewhere
RUN MT5_PYTHON_PATH=$(find /opt -type d -name "site-packages" | grep mt5 | head -n 1) && \
    echo $MT5_PYTHON_PATH && \
    ln -s $MT5_PYTHON_PATH /opt/rehoboam/rehoboam-tools/venv/lib/python3.12/site-packages/mt5

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
CMD ["/bin/bash", "-c", "source venv/bin/activate && ./rehoboam_tools_runner.sh"]
