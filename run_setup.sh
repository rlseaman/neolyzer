#!/bin/bash
# NEOlyzer Setup Launcher

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "./venv/bin/python" ]; then
    exec ./venv/bin/python scripts/setup_database.py "$@"
else
    echo "Error: Virtual environment not found. Run ./install.sh first."
    exit 1
fi
