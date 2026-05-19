from __future__ import annotations

import argparse
from datetime import date, datetime
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.parse import unquote, urlparse
from uuid import uuid4

import pandas as pd

from .data import fetch_realtime_spot, fetch_spot


DEFAULT_DATA_FILE = Path("data/holdings/positions.json")
QUOTE_CACHE_TTL_SECONDS = 20
_QUOTE_CACHE_LOCK = threading.Lock()
_QUOTE_CACHE: dict[str, Any] = {
    "expires_at": 0.0,
    "quotes": {},
    "meta": {"status": "empty", "source": "akshare_spot_em"},
    "last_success_quotes": {},
    "last_success_meta": {},
}


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


def positions_with_latest_quotes(positions: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    codes = {str(item.get("code") or "").strip().zfill(6) for item in positions if item.get("code")}
    if not codes:
        return list(positions), {
            "status": "empty",
            "source": "akshare_spot_em",
            "requested": 0,
            "matched": 0,
        }
    quotes, meta = _cached_quote_map()
    enriched: list[dict[str, Any]] = []
    for item in positions:
        row = dict(item)
        code = str(row.get("code") or "").strip().zfill(6)
        quote = quotes.get(code)
        if quote:
            row["latest_price"] = quote["latest_price"]
            row["latest_name"] = quote.get("name") or row.get("name") or ""
            row["price_time"] = quote["price_time"]
            row["price_source"] = quote["price_source"]
            row["pct_change"] = quote.get("pct_change")
            row["amount"] = quote.get("amount")
        enriched.append(row)

    found = sum(1 for item in enriched if item.get("latest_price"))
    return enriched, {**meta, "requested": len(codes), "matched": found}


def _cached_quote_map() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    now = time.time()
    with _QUOTE_CACHE_LOCK:
        if now < float(_QUOTE_CACHE.get("expires_at") or 0):
            return dict(_QUOTE_CACHE.get("quotes") or {}), dict(_QUOTE_CACHE.get("meta") or {})

    fetched_at = datetime.now().isoformat(timespec="seconds")
    try:
        quotes = _quote_map_from_spot(
            _fetch_realtime_spot_with_retry(),
            fetched_at=fetched_at,
            source="akshare_spot_em",
        )
        meta = {
            "status": "ok",
            "source": "akshare_spot_em",
            "fetched_at": fetched_at,
            "ttl_seconds": QUOTE_CACHE_TTL_SECONDS,
            "quote_count": len(quotes),
        }
    except Exception as exc:  # pragma: no cover - network/data-source fallback path
        quotes, meta = _quote_fallback(exc, fetched_at)

    with _QUOTE_CACHE_LOCK:
        _QUOTE_CACHE["expires_at"] = now + QUOTE_CACHE_TTL_SECONDS
        _QUOTE_CACHE["quotes"] = quotes
        _QUOTE_CACHE["meta"] = meta
        if meta.get("status") == "ok":
            _QUOTE_CACHE["last_success_quotes"] = quotes
            _QUOTE_CACHE["last_success_meta"] = meta
    return quotes, meta


def _fetch_realtime_spot_with_retry() -> pd.DataFrame:
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            return fetch_realtime_spot()
        except Exception as exc:  # pragma: no cover - network/data-source fallback path
            last_exc = exc
            if attempt == 0:
                time.sleep(1.0)
    assert last_exc is not None
    raise last_exc


def _quote_fallback(exc: Exception, fetched_at: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    with _QUOTE_CACHE_LOCK:
        last_quotes = dict(_QUOTE_CACHE.get("last_success_quotes") or {})
        last_meta = dict(_QUOTE_CACHE.get("last_success_meta") or {})
    if last_quotes:
        return last_quotes, {
            **last_meta,
            "status": "stale",
            "fetched_at": fetched_at,
            "stale_price_time": last_meta.get("fetched_at"),
            "ttl_seconds": QUOTE_CACHE_TTL_SECONDS,
            "error": str(exc),
        }

    try:
        fallback_at = datetime.now().isoformat(timespec="seconds")
        quotes = _quote_map_from_spot(
            fetch_spot(),
            fetched_at=fallback_at,
            source="latest_spot_fallback",
        )
        return quotes, {
            "status": "fallback",
            "source": "latest_spot_fallback",
            "fetched_at": fallback_at,
            "ttl_seconds": QUOTE_CACHE_TTL_SECONDS,
            "quote_count": len(quotes),
            "error": str(exc),
        }
    except Exception as fallback_exc:  # pragma: no cover - network/data-source fallback path
        return {}, {
            "status": "error",
            "source": "akshare_spot_em",
            "fetched_at": fetched_at,
            "ttl_seconds": QUOTE_CACHE_TTL_SECONDS,
            "error": f"{exc}; fallback failed: {fallback_exc}",
        }


def _quote_map_from_spot(spot: pd.DataFrame, fetched_at: str, source: str = "akshare_spot_em") -> dict[str, dict[str, Any]]:
    if spot.empty:
        return {}
    quotes: dict[str, dict[str, Any]] = {}
    df = spot.copy()
    df["code"] = df["code"].astype(str).str.extract(r"(\d{6})", expand=False).str.zfill(6)
    for _, row in df.dropna(subset=["code"]).iterrows():
        code = str(row["code"]).zfill(6)
        latest = _optional_float(row.get("latest"))
        if latest is None or latest <= 0:
            continue
        quotes[code] = {
            "code": code,
            "name": str(row.get("name") or ""),
            "latest_price": round(latest, 4),
            "pct_change": _optional_float(row.get("pct_change")),
            "amount": _optional_float(row.get("amount")),
            "price_time": fetched_at,
            "price_source": source,
        }
    return quotes


def _optional_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


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
            positions, quote_meta = positions_with_latest_quotes(load_positions(self.data_file))
            self._send_json({"positions": positions, "quote": quote_meta})
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
