#!/usr/bin/env bash
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/a-breakout-screener}"
SITE_PATH="${STOCKS_SITE_PATH:-/stocks}"
CADDYFILE="${CADDYFILE:-/etc/caddy/Caddyfile}"

cd "$APP_DIR"
mkdir -p "$APP_DIR/site"
chmod +x "$APP_DIR/publish_site.sh"

if ! command -v caddy >/dev/null 2>&1 || [ ! -f "$CADDYFILE" ]; then
  echo "Caddy 未安装或未找到 $CADDYFILE，跳过网站配置"
  exit 0
fi

python3 - "$CADDYFILE" "$APP_DIR/site" "$SITE_PATH" <<'PY'
from pathlib import Path
import sys

caddyfile = Path(sys.argv[1])
site_root = sys.argv[2]
site_path = sys.argv[3].rstrip("/")
text = caddyfile.read_text()
marker = "# a-breakout-screener stocks site"
block = f"""
	{marker}
	handle_path {site_path}* {{
		root * {site_root}
		file_server
	}}
"""

if marker in text:
    start = text.index(f"\t{marker}")
    end = text.index("\n\treverse_proxy", start)
    text = text[:start] + block.lstrip("\n") + text[end + 1:]
elif "\n\treverse_proxy 127.0.0.1:8080" in text:
    text = text.replace("\n\treverse_proxy 127.0.0.1:8080", "\n" + block + "\treverse_proxy 127.0.0.1:8080", 1)
else:
    raise SystemExit("未找到可插入的网站块位置")

caddyfile.write_text(text)
PY

caddy validate --config "$CADDYFILE"
caddy reload --config "$CADDYFILE"
echo "网站已配置: $SITE_PATH/"
