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

def indented(raw: str, indent: str) -> str:
    return "\n".join((indent + line if line else "") for line in raw.strip("\n").splitlines()) + "\n"

def stocks_block(indent: str) -> str:
    return indented(
        f"""
{marker}
handle_path {site_path}* {{
	root * {site_root}
	file_server
}}
""",
        indent,
    )

def sub2api_block(indent: str) -> str:
    return indented(
        """
# Sub2API frontend is disabled on the public site. Only backend APIs are exposed.
@sub2api_api {
	path /api
	path /api/*
	path /v1
	path /v1/*
}

handle @sub2api_api {
	reverse_proxy 127.0.0.1:8080
}

handle {
	respond "not found" 404
}
""",
        indent,
    )

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

def line_start_at(src: str, pos: int) -> int:
    return src.rfind("\n", 0, pos) + 1

def line_indent_at(src: str, pos: int) -> str:
    start = line_start_at(src, pos)
    return src[start:pos]

def replace_sub2api_block(src: str) -> str:
    comment_pos = src.find("# Sub2API")
    if comment_pos < 0:
        return src
    start = line_start_at(src, comment_pos)
    indent = line_indent_at(src, comment_pos)
    matcher_pos = src.find(f"\n{indent}@sub2api", start)
    if matcher_pos < 0:
        return src
    fallback_start = src.find(f"\n{indent}handle {{", matcher_pos)
    if fallback_start >= 0:
        end = find_matching_brace(src, src.index("{", fallback_start)) + 1
    else:
        handle_pos = src.find(f"\n{indent}handle @sub2api", matcher_pos)
        if handle_pos < 0:
            return src
        end = find_matching_brace(src, src.index("{", handle_pos)) + 1
    if end < len(src) and src[end] == "\n":
        end += 1
    return src[:start] + sub2api_block(indent) + src[end:]

if marker in text:
    marker_pos = text.index(marker)
    start = line_start_at(text, marker_pos)
    indent = line_indent_at(text, marker_pos)
    handle_start = text.index(f"\n{indent}handle_path", marker_pos)
    open_pos = text.index("{", handle_start)
    end = find_matching_brace(text, open_pos) + 1
    if end < len(text) and text[end] == "\n":
        end += 1
    text = text[:start] + stocks_block(indent) + text[end:]
elif "\n\t\t@sub2api_api {" in text:
    insert_at = text.index("\n\t\t@sub2api_api {")
    text = text[:insert_at] + "\n" + stocks_block("\t\t") + text[insert_at:]
elif "\n\t@sub2api_api {" in text:
    insert_at = text.index("\n\t@sub2api_api {")
    text = text[:insert_at] + "\n" + stocks_block("\t") + text[insert_at:]
elif "\n\t@sub2api {" in text:
    insert_at = text.index("\n\t@sub2api {")
    text = text[:insert_at] + "\n" + stocks_block("\t") + text[insert_at:]
elif "\n\thandle {" in text:
    insert_at = text.index("\n\thandle {")
    text = text[:insert_at] + "\n" + stocks_block("\t") + text[insert_at:]
elif "\n\treverse_proxy 127.0.0.1:8080" in text:
    text = text.replace(
        "\n\treverse_proxy 127.0.0.1:8080",
        "\n" + stocks_block("\t") + sub2api_block("\t"),
        1,
    )
else:
    raise SystemExit("未找到可插入的网站块位置")

text = replace_sub2api_block(text)
if "\n\t@sub2api_api {" not in text and "\n\t\t@sub2api_api {" not in text:
    text = text.replace("\n\thandle {\n\t\treverse_proxy 127.0.0.1:8080\n\t}", sub2api_block("\t"))
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
