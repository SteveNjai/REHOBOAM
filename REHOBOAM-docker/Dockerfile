# ============================================
# REHOBOAM Docker Image
# Ubuntu 24.04 + MetaTrader 5 + Python venv
# Production-grade, secure, reproducible
# ============================================

# ---------- Stage 1: Build Environment ----------
FROM ubuntu:24.04 AS builder

# Environment configuration
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Africa/Nairobi \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Install pinned system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3=3.12.* \
    python3-venv=3.12.* \
    python3-pip=24.* \
    wget=1.21.* \
    curl=7.88.* \
    git=1:2.* \
    xvfb=2:21.* \
    && rm -rf /var/lib/apt/lists/*

# Working directory
WORKDIR /opt/rehoboam

# Clone REHOBOAM repository from GitHub
RUN git clone --depth=1 https://github.com/SteveNjai/REHOBOAM.git .

# Create Python virtual environment & install requirements
RUN python3 -m venv rehoboam-tools/Scripts && \
    . rehoboam-tools/Scripts/activate && \
    pip install --no-cache-dir -r rehoboam-tools/requirements.txt

# Install MetaTrader 5 (official Linux installer)
RUN wget -q https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5linux.sh && \
    chmod +x mt5linux.sh && \
    ./mt5linux.sh && \
    rm -f mt5linux.sh

# ---------- Stage 2: Runtime Environment ----------
FROM ubuntu:24.04

# Environment configuration
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=Africa/Nairobi \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Install only runtime essentials (no dev tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3=3.12.* \
    python3-venv=3.12.* \
    xvfb=2:21.* \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m rehoboam && mkdir -p /opt/rehoboam && chown -R rehoboam:rehoboam /opt/rehoboam
USER rehoboam

# Copy built project & MT5 installation from builder
COPY --from=builder /opt/rehoboam /opt/rehoboam
COPY --from=builder /root/.metatrader5 /home/rehoboam/.metatrader5

# Working directory
WORKDIR /opt/rehoboam

# Ensure permissions
RUN chmod +x rehoboam-tools/rehoboam_tools_runner.sh

# Health check (optional)
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD [ "bash", "-c", "pgrep -f rehoboam_tools_runner.sh >/dev/null || exit 1" ]

# Default entrypoint
ENTRYPOINT ["bash", "-c", "source rehoboam-tools/Scripts/activate && ./rehoboam-tools/rehoboam_tools_runner.sh"]
