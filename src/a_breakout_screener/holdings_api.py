from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse
from uuid import uuid4

import pandas as pd

from .ai_analysis import generate_single_stock_analysis
from .config import load_config
from .data import _call_tushare, fetch_history, fetch_realtime_spot, fetch_spot


DEFAULT_DATA_FILE = Path("data/holdings/positions.json")
QUOTE_CACHE_TTL_SECONDS = 20
SEARCH_CACHE_TTL_SECONDS = 300
STOCK_UNIVERSE_CACHE_NAME = "stock_universe.csv"
_QUOTE_CACHE_LOCK = threading.Lock()
_QUOTE_CACHE: dict[str, Any] = {
    "expires_at": 0.0,
    "quotes": {},
    "meta": {"status": "empty", "source": "akshare_spot_em"},
    "last_success_quotes": {},
    "last_success_meta": {},
}
_SEARCH_CACHE_LOCK = threading.Lock()
_SEARCH_CACHE: dict[str, Any] = {"expires_at": 0.0, "items": []}


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


def search_stocks(query: str, limit: int = 12) -> list[dict[str, Any]]:
    query = query.strip()
    if not query:
        return []
    normalized = query.zfill(6) if query.isdigit() else query
    normalized_name_query = _normalize_search_text(query)
    items = _cached_stock_universe()
    matches: list[tuple[int, dict[str, Any]]] = []
    query_lower = query.lower()
    for item in items:
        code = str(item.get("code") or "")
        name = str(item.get("name") or "")
        name_lower = name.lower()
        normalized_name = _normalize_search_text(name)
        score: int | None = None
        if code == normalized:
            score = 0
        elif query.isdigit() and code.startswith(query):
            score = 1
        elif query.isdigit() and query in code:
            score = 2
        elif name == query:
            score = 3
        elif name.startswith(query):
            score = 4
        elif query_lower and query_lower in name_lower:
            score = 5
        elif normalized_name_query and normalized_name_query in normalized_name:
            score = 6
        elif normalized_name_query and _is_subsequence(normalized_name_query, normalized_name):
            score = 7
        if score is not None:
            matches.append((score, item))
    matches.sort(key=lambda pair: (pair[0], str(pair[1].get("code") or "")))
    return [item for _, item in matches[: max(1, limit)]]


def load_kline_payload(code: str, data_file: Path) -> dict[str, Any]:
    raw_code = str(code or "").strip()
    if not raw_code:
        raise ValueError("code is required")
    code = raw_code.zfill(6)
    config = load_config(Path("config.toml"))
    end_date = date.today()
    start_date = end_date - timedelta(days=max(365, config.screener.history_days))
    history = fetch_history(
        symbol=code,
        start_date=start_date,
        end_date=end_date,
        cache_dir=config.paths.cache_dir,
        force_refresh=False,
        allow_truncated_start=True,
    )
    if history.empty:
        raise ValueError(f"{code} kline history is unavailable")
    universe = {item["code"]: item for item in _cached_stock_universe()}
    stock = universe.get(code, {"code": code, "name": code})
    latest_close = _optional_float(history.sort_values("date").iloc[-1].get("close")) or 0.0
    item = {
        "code": code,
        "name": stock.get("name") or code,
        "signalType": "K线",
        "signalReason": "",
        "primaryTag": "搜索结果",
        "latestClose": round(latest_close, 4),
        "circMv": 0,
        "score": 0,
        "growthScore": 0,
        "tags": [],
        "strategyData": False,
        "tradeAction": "仅查看K线",
        "hint": "搜索结果，不代表策略候选",
    }
    return {
        "item": item,
        "history": _history_records(history),
        "source": "cache_or_data_source",
    }


def _history_records(history: pd.DataFrame) -> list[dict[str, Any]]:
    df = history.sort_values("date").tail(900).copy()
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        records.append(
            {
                "date": pd.Timestamp(row["date"]).date().isoformat(),
                "open": _optional_float(row.get("open")),
                "high": _optional_float(row.get("high")),
                "low": _optional_float(row.get("low")),
                "close": _optional_float(row.get("close")),
                "volume": _optional_float(row.get("volume")),
            }
        )
    return records


def _cached_stock_universe() -> list[dict[str, Any]]:
    now = time.time()
    with _SEARCH_CACHE_LOCK:
        if now < float(_SEARCH_CACHE.get("expires_at") or 0):
            return list(_SEARCH_CACHE.get("items") or [])
        config = load_config(Path("config.toml"))
        items = _read_stock_universe_cache(config.paths.cache_dir)
        if not items:
            items = _stock_items_from_tushare_basic()
            if items:
                _write_stock_universe_cache(config.paths.cache_dir, items)
        if not items:
            try:
                spot = fetch_spot()
                items = _stock_items_from_spot(spot)
                if not items:
                    raise RuntimeError("stock universe is empty")
                _write_stock_universe_cache(config.paths.cache_dir, items)
            except Exception:  # pragma: no cover - network/data-source fallback path
                items = _stock_items_from_hist_cache(config.paths.cache_dir)
        _SEARCH_CACHE["expires_at"] = now + SEARCH_CACHE_TTL_SECONDS
        _SEARCH_CACHE["items"] = items
        return list(items)


def _stock_items_from_spot(spot: pd.DataFrame) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if spot.empty:
        return items
    df = spot.copy()
    df["code"] = df["code"].astype(str).str.extract(r"(\d{6})", expand=False).str.zfill(6)
    for _, row in df.dropna(subset=["code"]).iterrows():
        code = str(row.get("code") or "").zfill(6)
        latest = _optional_float(row.get("latest"))
        items.append(
            {
                "code": code,
                "name": str(row.get("name") or code),
                "latest": round(latest, 4) if latest is not None else None,
                "pct_change": _optional_float(row.get("pct_change")),
            }
        )
    return items


def _stock_items_from_hist_cache(cache_dir: Path) -> list[dict[str, Any]]:
    hist_dir = cache_dir / "hist"
    if not hist_dir.is_dir():
        return []
    return [{"code": path.stem.zfill(6), "name": path.stem.zfill(6)} for path in sorted(hist_dir.glob("*.csv"))]


def _stock_items_from_tushare_basic() -> list[dict[str, Any]]:
    try:
        from .tushare_client import get_tushare_pro

        pro = get_tushare_pro()
        raw = _call_tushare(
            "stock_basic",
            lambda: pro.stock_basic(exchange="", list_status="L", fields="ts_code,symbol,name"),
        )
    except Exception:  # pragma: no cover - external data-source fallback path
        return []
    if raw is None or raw.empty:
        return []
    df = raw.copy()
    if "symbol" in df.columns:
        df["code"] = df["symbol"].astype(str).str.extract(r"(\d{6})", expand=False).str.zfill(6)
    else:
        df["code"] = df["ts_code"].astype(str).str.extract(r"(\d{6})", expand=False).str.zfill(6)
    df["name"] = df["name"].fillna(df["code"]).astype(str)
    return [
        {"code": str(row["code"]).zfill(6), "name": str(row["name"]), "latest": None, "pct_change": None}
        for _, row in df.dropna(subset=["code"]).iterrows()
    ]


def _read_stock_universe_cache(cache_dir: Path) -> list[dict[str, Any]]:
    path = cache_dir / STOCK_UNIVERSE_CACHE_NAME
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path, dtype={"code": str})
    except Exception:
        return []
    if "code" not in df.columns or "name" not in df.columns:
        return []
    df["code"] = df["code"].astype(str).str.extract(r"(\d{6})", expand=False).str.zfill(6)
    df["name"] = df["name"].fillna(df["code"]).astype(str)
    items: list[dict[str, Any]] = []
    for _, row in df.dropna(subset=["code"]).iterrows():
        latest = _optional_float(row.get("latest"))
        items.append(
            {
                "code": str(row["code"]).zfill(6),
                "name": str(row["name"]),
                "latest": round(latest, 4) if latest is not None else None,
                "pct_change": _optional_float(row.get("pct_change")),
            }
        )
    return items


def _write_stock_universe_cache(cache_dir: Path, items: list[dict[str, Any]]) -> None:
    if not items:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / STOCK_UNIVERSE_CACHE_NAME
    pd.DataFrame(items).to_csv(path, index=False, encoding="utf-8-sig")


def _normalize_search_text(value: str) -> str:
    return "".join(str(value or "").lower().split())


def _is_subsequence(needle: str, haystack: str) -> bool:
    if not needle:
        return False
    position = 0
    for char in haystack:
        if char == needle[position]:
            position += 1
            if position == len(needle):
                return True
    return False


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
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        if path.endswith("/health"):
            self._send_json({"ok": True})
            return
        if path.endswith("/holdings"):
            positions, quote_meta = positions_with_latest_quotes(load_positions(self.data_file))
            self._send_json({"positions": positions, "quote": quote_meta})
            return
        if path.endswith("/stock-search"):
            params = parse_qs(parsed.query)
            query = (params.get("q") or [""])[0]
            limit = int((params.get("limit") or ["12"])[0] or 12)
            self._send_json({"items": search_stocks(query, limit=limit)})
            return
        if path.endswith("/kline"):
            params = parse_qs(parsed.query)
            code = (params.get("code") or [""])[0]
            try:
                self._send_json(load_kline_payload(code, self.data_file))
            except ValueError as exc:
                self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            except Exception as exc:  # pragma: no cover - network/data-source fallback path
                self._send_json({"error": f"K线数据暂不可用：{exc}"}, HTTPStatus.BAD_GATEWAY)
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/")
        if path.endswith("/stock-ai"):
            self._handle_stock_ai()
            return
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

    def _handle_stock_ai(self) -> None:
        try:
            data = self._read_json()
            stock = data.get("stock")
            if not isinstance(stock, dict):
                raise ValueError("stock is required")
            code = str(stock.get("code") or "").strip()
            if not code:
                raise ValueError("stock.code is required")
            analysis = generate_single_stock_analysis(load_config(Path("config.toml")).ai_analysis, stock)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        except Exception as exc:  # pragma: no cover - external AI gateway variance
            self._send_json({"error": f"AI分析暂不可用：{exc}"}, HTTPStatus.BAD_GATEWAY)
            return
        self._send_json(
            {
                "code": code,
                "analysis": analysis,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            }
        )

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
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args(argv)
    if args.env_file:
        _load_env_file(Path(args.env_file))
    _apply_network_env(load_config(Path("config.toml")))
    run_server(args.host, args.port, args.data_file.expanduser().resolve())
    return 0


def _load_env_file(path: Path) -> None:
    path = path.expanduser()
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _apply_network_env(config) -> None:
    if config.network.disable_system_proxy:
        os.environ.setdefault("NO_PROXY", "*")
        os.environ.setdefault("no_proxy", "*")


if __name__ == "__main__":
    raise SystemExit(main())
