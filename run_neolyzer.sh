#!/bin/bash
# NEOlyzer Launcher
# Automatically finds and uses the correct Python environment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use virtual environment Python
if [ -f "./venv/bin/python" ]; then
    exec ./venv/bin/python src/neolyzer.py "$@"
else
    echo "Error: Virtual environment not found. Run ./install.sh first."
    exit 1
fi
