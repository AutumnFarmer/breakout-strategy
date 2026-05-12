#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/opt/a-breakout-screener"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CRON_TIME="${CRON_TIME:-20 8 * * 1-5}"
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

cd "$APP_DIR"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv sync --frozen || uv sync
chmod +x "$APP_DIR/run_daily.sh"
chmod +x "$APP_DIR/publish_site.sh"

if [ ! -f "$APP_DIR/config.toml" ]; then
  cp "$APP_DIR/config.example.toml" "$APP_DIR/config.toml"
fi

if [ ! -f "$APP_DIR/.env" ]; then
  cat > "$APP_DIR/.env" <<'EOF'
A_BREAKOUT_EMAIL_ENABLED=true
A_BREAKOUT_EMAIL_METHOD=codex_gmail
MAIL_TO=zhangmc895@gmail.com
EOF
  chmod 600 "$APP_DIR/.env"
fi

touch "$APP_DIR/cron.log"
CRON_LINE="$CRON_TIME cd $APP_DIR && ./run_daily.sh >> $APP_DIR/cron.log 2>&1"
TMP_CRON="$(mktemp)"
crontab -l 2>/dev/null | grep -v "cd $APP_DIR && ./run_daily.sh" > "$TMP_CRON" || true
echo "$CRON_LINE" >> "$TMP_CRON"
crontab "$TMP_CRON"
rm -f "$TMP_CRON"

bash "$APP_DIR/deploy/install_web.sh" || true

echo "Installed $APP_DIR"
echo "Cron: $CRON_LINE"
