#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-aiwork-server}"
REMOTE_DIR="${REMOTE_DIR:-/opt/a-breakout-screener}"

ssh "$HOST" "export PATH=\"\$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:\$PATH\"; cd '$REMOTE_DIR' && pwd && uv run a-breakout --config config.toml --env-file .env doctor --skip-network && crontab -l | grep 'cd $REMOTE_DIR && ./run_daily.sh' && tail -80 cron.log 2>/dev/null || true"
