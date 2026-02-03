#!/usr/bin/env bash
set -euo pipefail

: "${PI_HOST:?Set PI_HOST (e.g. pi.local)}"
: "${PI_USER:=pi}"
: "${PI_PATH:=/home/${PI_USER}/squark}"

ssh "${PI_USER}@${PI_HOST}" "${PI_PATH}/squark-app"

