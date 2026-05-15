#!/usr/bin/env bash
set -e
echo "[install] Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo "[install] Done.  Activate with:  source .venv/bin/activate"
