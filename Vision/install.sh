#!/usr/bin/env bash
# Vision Detection — Ubuntu install script
# Mac Mini 2011, Ubuntu 20.04/22.04, CPU-only
set -euo pipefail

echo "=== Vision Detection Setup ==="

# ── System packages ────────────────────────────────────────────────────────────
echo "[1/4] Installing system packages …"
sudo apt-get update -qq
sudo apt-get install -y \
    python3 python3-pip python3-venv \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 \
    v4l-utils \
    --no-install-recommends

# ── Check USB camera ───────────────────────────────────────────────────────────
echo "[2/4] Checking USB camera …"
echo "Video devices:"
ls /dev/video* 2>/dev/null || echo "  (none found — plug in USB camera and re-run)"
v4l2-ctl --list-devices 2>/dev/null || true

# ── Python virtualenv ─────────────────────────────────────────────────────────
echo "[3/4] Creating Python virtual environment …"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --upgrade pip --quiet

# Install CPU-only PyTorch FIRST — prevents pip from pulling in CUDA/nvidia packages
echo "  Installing CPU-only PyTorch …"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --quiet

pip install -r "$SCRIPT_DIR/requirements.txt"

echo "[4/4] Pre-downloading YOLOv8n weights …"
python3 -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" 2>&1 | tail -3

echo ""
echo "=== Setup complete! ==="
echo ""
echo "Usage (from this directory):"
echo "  ./run.sh                        # local display (needs monitor/X11)"
echo "  ./run.sh --web                  # headless: open http://<this-ip>:5000"
echo "  ./run.sh --web --port 8080      # custom port"
echo "  ./run.sh -c 1                   # if USB cam is /dev/video1"
echo "  ./run.sh --no-pose              # skip body skeleton"
echo "  ./run.sh --no-hands             # skip gesture detection"
echo "  ./run.sh --conf 0.35            # lower confidence threshold"
echo ""
echo "Find camera index:"
echo "  ls /dev/video*"
echo "  v4l2-ctl --list-devices"
