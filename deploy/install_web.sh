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
sub2api_block = """
	# Sub2API is intentionally not a catch-all. Keep it on explicit app/API paths
	# so new static directories on painlife.cloud never fall through to Sub2API.
	@sub2api {
		path /
		path /api/*
		path /assets/*
		path /logo.png
		path /setup
		path /home
		path /login
		path /register
		path /email-verify
		path /auth/*
		path /forgot-password
		path /reset-password
		path /key-usage
		path /dashboard
		path /keys
		path /usage
		path /redeem
		path /profile
		path /subscriptions
		path /purchase
		path /orders*
		path /payment/*
		path /custom/*
		path /admin*
	}

	handle @sub2api {
		reverse_proxy 127.0.0.1:8080
	}

	handle {
		respond "not found" 404
	}
"""

def find_matching_brace(src: str, open_pos: int) -> int:
    depth = 0
    for idx in range(open_pos, len(src)):
        char = src[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return idx
    raise SystemExit("Caddyfile 中 handle_path 块未闭合")

if marker in text:
    start = text.index(f"\t{marker}")
    handle_start = text.index("\n\thandle_path", start)
    open_pos = text.index("{", handle_start)
    end = find_matching_brace(text, open_pos) + 1
    if end < len(text) and text[end] == "\n":
        end += 1
    text = text[:start] + block.lstrip("\n") + text[end:]
elif "\n\t@sub2api {" in text:
    insert_at = text.index("\n\t@sub2api {")
    text = text[:insert_at] + "\n" + block + text[insert_at:]
elif "\n\thandle {" in text:
    insert_at = text.index("\n\thandle {")
    text = text[:insert_at] + "\n" + block + text[insert_at:]
elif "\n\treverse_proxy 127.0.0.1:8080" in text:
    text = text.replace(
        "\n\treverse_proxy 127.0.0.1:8080",
        "\n" + block + sub2api_block,
        1,
    )
else:
    raise SystemExit("未找到可插入的网站块位置")

if "\n\t@sub2api {" not in text:
    text = text.replace("\n\thandle {\n\t\treverse_proxy 127.0.0.1:8080\n\t}", sub2api_block)
caddyfile.write_text(text)
PY

caddy validate --config "$CADDYFILE"
if systemctl list-unit-files caddy-sub2api.service >/dev/null 2>&1; then
  systemctl reload caddy-sub2api.service
elif systemctl list-unit-files caddy.service >/dev/null 2>&1; then
  systemctl reload caddy.service
else
  caddy reload --config "$CADDYFILE"
fi
echo "网站已配置: $SITE_PATH/"
