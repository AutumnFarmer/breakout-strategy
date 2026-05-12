#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-aiwork-server}"
REMOTE_DIR="${REMOTE_DIR:-/opt/a-breakout-screener}"

rsync -az --delete \
  --exclude ".venv" \
  --exclude ".git" \
  --exclude "data/cache" \
  --exclude "outputs" \
  --exclude ".env" \
  ./ "$HOST:$REMOTE_DIR/"

ssh "$HOST" "cd '$REMOTE_DIR' && bash deploy/install_server.sh"
