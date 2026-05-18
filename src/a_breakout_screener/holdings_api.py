from __future__ import annotations

import argparse
from datetime import date, datetime
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.parse import unquote, urlparse
from uuid import uuid4


DEFAULT_DATA_FILE = Path("data/holdings/positions.json")


def load_positions(data_file: Path) -> list[dict[str, Any]]:
    if not data_file.exists():
        return []
    try:
        payload = json.loads(data_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(payload, dict):
        positions = payload.get("positions", [])
    else:
        positions = payload
    if not isinstance(positions, list):
        return []
    return [item for item in positions if isinstance(item, dict)]


def save_positions(data_file: Path, positions: list[dict[str, Any]]) -> None:
    data_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "positions": positions,
    }
    with NamedTemporaryFile("w", encoding="utf-8", dir=data_file.parent, delete=False) as fp:
        tmp_path = Path(fp.name)
        json.dump(payload, fp, ensure_ascii=False, indent=2)
        fp.write("\n")
    tmp_path.replace(data_file)


def create_position(data: dict[str, Any]) -> dict[str, Any]:
    code = str(data.get("code") or "").strip()
    name = str(data.get("name") or "").strip()
    if not code:
        raise ValueError("code is required")
    buy_price = _positive_float(data.get("buy_price"), "buy_price")
    quantity = _positive_float(data.get("quantity"), "quantity")
    buy_date = str(data.get("buy_date") or date.today().isoformat()).strip()
    return {
        "id": uuid4().hex,
        "code": code,
        "name": name,
        "buy_price": buy_price,
        "quantity": quantity,
        "buy_date": buy_date,
        "note": str(data.get("note") or "").strip(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def _positive_float(value: object, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive number") from exc
    if number <= 0:
        raise ValueError(f"{field_name} must be a positive number")
    return number


class HoldingsHandler(BaseHTTPRequestHandler):
    data_file = DEFAULT_DATA_FILE

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/")
        if path.endswith("/health"):
            self._send_json({"ok": True})
            return
        if path.endswith("/holdings"):
            self._send_json({"positions": load_positions(self.data_file)})
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/")
        if not path.endswith("/holdings"):
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            data = self._read_json()
            position = create_position(data)
            positions = load_positions(self.data_file)
            positions.append(position)
            save_positions(self.data_file, positions)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        self._send_json(position, HTTPStatus.CREATED)

    def do_DELETE(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/")
        marker = "/holdings/"
        if marker not in path:
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        position_id = unquote(path.rsplit(marker, 1)[-1]).strip()
        positions = load_positions(self.data_file)
        next_positions = [item for item in positions if str(item.get("id")) != position_id]
        if len(next_positions) == len(positions):
            self._send_json({"error": "position not found"}, HTTPStatus.NOT_FOUND)
            return
        save_positions(self.data_file, next_positions)
        self._send_json({"ok": True})

    def log_message(self, format: str, *args: object) -> None:
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length") or 0)
        if length <= 0:
            raise ValueError("request body is required")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("request body must be JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status.value)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("cache-control", "no-store")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(host: str, port: int, data_file: Path) -> None:
    handler = type("ConfiguredHoldingsHandler", (HoldingsHandler,), {"data_file": data_file})
    server = ThreadingHTTPServer((host, port), handler)
    print(f"holdings api listening on {host}:{port}, data={data_file}", flush=True)
    server.serve_forever()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A-share holdings persistence API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    args = parser.parse_args(argv)
    run_server(args.host, args.port, args.data_file.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
