#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
import sys

if sys.version_info[:2] < (3, 10):
    raise SystemExit("VoxCPM2 API requires Python 3.10 or newer for local setup.")
PY

python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,voxcpm]"
