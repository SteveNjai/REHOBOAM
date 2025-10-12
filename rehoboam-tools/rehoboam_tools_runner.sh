#!/bin/bash

# ===============================
# REHOBOAM Script Runner (Linux)
# ===============================

# Get the project directory (the folder where this script resides)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$PROJECT_DIR/venv/bin/python"
SCRIPT_DIR="$PROJECT_DIR"
SYMBOL_FILE="$PROJECT_DIR/oracle-spread/symbols.txt"

# Function to pause (like Windows 'pause')
pause() {
    read -rp "Press Enter to return to menu..."
}

while true; do
    clear
    echo "===================================="
    echo "      REHOBOAM Script Runner"
    echo "===================================="
    echo "1. Run zscore_analyzer.py (show zscore statistics)"
    echo "2. Run fetch_mt5.py (fetch history for rehoboam_optimizer tool)"
    echo "3. Run run_optimize.py (optimize on history data for rehoboam_optimizer)"
    echo "4. Run Oracle-Spread (simulate PnL for each zscore pair)"
    echo "5. Exit"
    echo "------------------------------------"
    read -rp "Enter your choice (1-5): " choice

    case $choice in
        1)
            echo "Running zscore_analyzer.py..."
            cd "$SCRIPT_DIR/zscore_analyzer" || exit
            "$PYTHON" zscore_analyzer.py
            pause
            ;;
        2)
            echo "Running fetch_mt5.py..."
            cd "$SCRIPT_DIR/rehoboam_optimizer" || exit
            "$PYTHON" fetch_mt5.py
            pause
            ;;
        3)
            echo "Running run_optimize.py..."
            cd "$SCRIPT_DIR/rehoboam_optimizer" || exit
            "$PYTHON" run_optimize.py
            pause
            ;;
        4)
            echo "Running Oracle-Spread..."
            cd "$SCRIPT_DIR/oracle-spread" || exit

            if [[ ! -f "$SYMBOL_FILE" ]]; then
                echo "Error: symbols.txt not found at $SYMBOL_FILE"
                pause
                continue
            fi

            while IFS=',' read -r SYMBOL1 SYMBOL2; do
                SYMBOL1=$(echo "$SYMBOL1" | xargs)
                SYMBOL2=$(echo "$SYMBOL2" | xargs)
                echo "Launching pair: $SYMBOL1 , $SYMBOL2"
                echo "------------------------------------"

                # Run each pair in its own terminal window
                gnome-terminal -- bash -c "cd '$SCRIPT_DIR/oracle-spread' && '$PYTHON' oracle-spread.py '$SYMBOL1' '$SYMBOL2'; exec bash" &
            done < "$SYMBOL_FILE"

            echo "All oracle-spread instances launched in parallel windows."
            pause
            ;;
        5)
            echo "Exiting..."
            exit 0
            ;;
        *)
            echo "Invalid choice. Please enter a number from 1 to 5."
            pause
            ;;
    esac
done
