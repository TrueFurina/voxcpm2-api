#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This script only builds DMG artifacts on macOS." >&2
  exit 1
fi

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 <path-to-app-bundle> <output-dmg> [volume-name]" >&2
  exit 1
fi

APP_PATH="$1"
OUTPUT_DMG="$2"
VOLUME_NAME="${3:-$(basename "${APP_PATH%.app}")}"

if [[ ! -d "$APP_PATH" ]]; then
  echo "App bundle not found: $APP_PATH" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT_DMG")"
rm -f "$OUTPUT_DMG"

hdiutil create \
  -ov \
  -format UDZO \
  -volname "$VOLUME_NAME" \
  -srcfolder "$APP_PATH" \
  "$OUTPUT_DMG"

echo "DMG ready at $OUTPUT_DMG"
