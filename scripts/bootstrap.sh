#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
import sys

if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit("VoxCPM2 requires Python 3.10, 3.11, or 3.12 for local setup.")
PY

python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,voxcpm]"
