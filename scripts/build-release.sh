#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_DIR="$ROOT_DIR/dist"
TAURI_DIR="$ROOT_DIR/voxcpm2-ui/src-tauri"

cd "$ROOT_DIR"

mkdir -p "$DIST_DIR"
rm -f \
  "$DIST_DIR"/voxcpm2_api-* \
  "$DIST_DIR"/voxcpm2_compat-* \
  "$DIST_DIR"/VoxCPM2-ui-macos-*.zip

python3 -m build
python3 -m build packages/voxcpm2-compat --outdir "$DIST_DIR"
python3 -m twine check \
  "$DIST_DIR"/voxcpm2_api-* \
  "$DIST_DIR"/voxcpm2_compat-*

if command -v cargo >/dev/null 2>&1 && cargo tauri --version >/dev/null 2>&1; then
  VERSION_TAG="v$(python3 - <<'PY'
from pathlib import Path
import tomllib

data = tomllib.loads(Path("pyproject.toml").read_text())
print(data["project"]["version"])
PY
)"
  ARCH="$(uname -m)"
  (
    cd "$TAURI_DIR"
    cargo tauri build --bundles app
    ditto -c -k --sequesterRsrc --keepParent \
      "target/release/bundle/macos/VoxCPM2.app" \
      "$DIST_DIR/VoxCPM2-ui-macos-${VERSION_TAG}-${ARCH}.zip"
  )
else
  echo "Skipping Tauri bundle: cargo tauri is not installed." >&2
fi
