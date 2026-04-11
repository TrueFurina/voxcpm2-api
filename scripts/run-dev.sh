#!/usr/bin/env bash
set -euo pipefail

. .venv/bin/activate
exec voxcpm2-api
