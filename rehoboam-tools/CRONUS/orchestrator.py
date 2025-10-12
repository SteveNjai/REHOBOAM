#--------------ORCHESTARTOR.py----------
#-this code orchestrates the CRONUSautomated trading bot
#-----reuirements are in the rehoboam-tools folder

import threading
import subprocess
import os
import sys
import time
import signal
import logging

# === Logging Setup ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('orchestrate_cronus.log')
    ]
)
logger = logging.getLogger(__name__)

# === Global Variables ===
SYMBOLS_FILE = 'symbols.txt'
CRONUS_SCRIPT = 'cronus.py'
threads = []

# === Signal Handler for Graceful Shutdown ===
def signal_handler(sig, frame):
    logger.info("Received shutdown signal. Terminating all threads...")
    for thread in threads:
        if thread.is_alive():
            logger.info(f"Stopping thread for {thread.pair}")
            # Subprocess termination will be handled by cronus.py's signal handler
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# === Function to Run Cronus for a Symbol Pair ===
def run_cronus(symbol_a, symbol_b):
    pair = f"{symbol_a}-{symbol_b}"
    logger.info(f"Starting cronus.py for pair {pair}")
    try:
        # Execute cronus_trading.py with symbol_a and symbol_b as arguments
        process = subprocess.Popen(
            [sys.executable, CRONUS_SCRIPT, symbol_a, symbol_b],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        # Capture output for logging
        stdout, stderr = process.communicate()
        if stdout:
            logger.info(f"Output from {pair}: {stdout.strip()}")
        if stderr:
            logger.error(f"Error from {pair}: {stderr.strip()}")
        logger.info(f"cronus.py for {pair} exited with code {process.returncode}")
    except Exception as e:
        logger.error(f"Failed to run cronus.py for {pair}: {str(e)}")

# === Main Orchestration Logic ===
def main():
    # Check if cronus_trading.py exists
    if not os.path.isfile(CRONUS_SCRIPT):
        logger.error(f"{CRONUS_SCRIPT} not found in current directory")
        sys.exit(1)

    # Read symbols.txt
    try:
        with open(SYMBOLS_FILE, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        logger.error(f"{SYMBOLS_FILE} not found")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error reading {SYMBOLS_FILE}: {str(e)}")
        sys.exit(1)

    # Parse symbol pairs and start threads
    for line in lines:
        line = line.strip()
        if not line or ',' not in line:
            logger.warning(f"Skipping invalid line: {line}")
            continue
        try:
            symbol_a, symbol_b = line.split(',')
            symbol_a, symbol_b = symbol_a.strip(), symbol_b.strip()
            if not symbol_a or not symbol_b:
                logger.warning(f"Empty symbol in line: {line}")
                continue
            # Create and start a thread for this pair
            thread = threading.Thread(
                target=run_cronus,
                args=(symbol_a, symbol_b),
                name=f"Cronus-{symbol_a}-{symbol_b}"
            )
            thread.pair = f"{symbol_a}-{symbol_b}"  # For logging during shutdown
            threads.append(thread)
            thread.start()
            logger.info(f"Started thread for pair {symbol_a}-{symbol_b}")
        except ValueError:
            logger.warning(f"Invalid format in line: {line}")
            continue
        except Exception as e:
            logger.error(f"Error processing line {line}: {str(e)}")
            continue

    # Wait for all threads to complete (they won't unless interrupted, as cronus runs indefinitely)
    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        signal_handler(None, None)

if __name__ == "__main__":
    logger.info("Starting Cronus orchestration")
    main()
    logger.info("All threads completed")
