#!/bin/bash

# Activate venv
source venv/bin/activate

# Start MT5 terminal in background (headless via xvfb; adjust path if needed)
xvfb-run -a WINEPREFIX=/root/.mt5 wine '/root/.mt5/drive_c/Program Files/MetaTrader 5/terminal64.exe' &
MT5_PID=$!

# Wait a bit for MT5 to start (adjust sleep if needed)
sleep 10

# Start mt5linux server in background (uses default port 18812; adjust --port if needed)
python -m mt5linux '/root/.mt5/drive_c/Python312/python.exe' &
SERVER_PID=$!

# Run your original runner script
./rehoboam_tools_runner.sh

# Cleanup (optional: kill processes on exit)
kill $MT5_PID $SERVER_PID