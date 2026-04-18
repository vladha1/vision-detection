#!/usr/bin/env bash
# Quick launcher — activates venv and starts detection
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/venv/bin/activate"
export NNPACK_SUPPORTED=0
export TF_CPP_MIN_LOG_LEVEL=3
exec python3 -W ignore "$SCRIPT_DIR/detect.py" "$@" 2>/dev/null
