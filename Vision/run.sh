#!/usr/bin/env bash
# Quick launcher — activates venv and starts detection
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/venv/bin/activate"
exec python3 "$SCRIPT_DIR/detect.py" "$@"
