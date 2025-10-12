# ---------------------------
# Base image
# ---------------------------
FROM ubuntu:24.04

# Non-interactive mode (no prompts)
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Africa/Nairobi

# ---------------------------
# System dependencies (combined for fewer layers)
# ---------------------------
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        python3-pip \
        git \
        wget \
        curl \
        xvfb \
        ca-certificates \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ---------------------------
# Set working directory
# ---------------------------
WORKDIR /opt/rehoboam

# ---------------------------
# Clone your GitHub repository
# ---------------------------
RUN git clone --depth 1 https://github.com/SteveNjai/REHOBOAM.git .

# ---------------------------
# Create Linux-compatible virtual environment
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
# Permissions & cleanup
# ---------------------------
RUN chmod +x rehoboam-tools/rehoboam_tools_runner.sh && \
    find /opt/rehoboam -type f -name "*.pyc" -delete && \
    find /opt/rehoboam -type d -name "__pycache__" -exec rm -rf {} + && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /tmp/* /var/tmp/*

# ---------------------------
# Final working directory & entrypoint
# ---------------------------
WORKDIR /opt/rehoboam/rehoboam-tools
CMD ["/bin/bash", "-c", "source venv/bin/activate && ./rehoboam_tools_runner.sh"]
