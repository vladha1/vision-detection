#!/usr/bin/env bash
# Quick launcher — activates venv if present, then passes all args to detect.py
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

python detect.py "$@"
