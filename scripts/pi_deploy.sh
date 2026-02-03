#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   PI_HOST=pi.local PI_USER=pi ./scripts/pi_deploy.sh
# Optional:
#   PI_PATH=/home/pi/squark

: "${PI_HOST:?Set PI_HOST (e.g. pi.local)}"
: "${PI_USER:=pi}"
: "${PI_PATH:=/home/${PI_USER}/squark}"

BIN="target/aarch64-unknown-linux-gnu/release/squark-app"

if [[ ! -f "$BIN" ]]; then
  echo "Binary not found: $BIN" >&2
  echo "Build it first (see README.md)." >&2
  exit 1
fi

ssh "${PI_USER}@${PI_HOST}" "mkdir -p '${PI_PATH}'"
scp "$BIN" "${PI_USER}@${PI_HOST}:${PI_PATH}/squark-app"
ssh "${PI_USER}@${PI_HOST}" "chmod +x '${PI_PATH}/squark-app'"

echo "Deployed to ${PI_USER}@${PI_HOST}:${PI_PATH}/squark-app"
