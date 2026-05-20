from __future__ import annotations

import pytest
import pandas as pd

from a_breakout_screener import holdings_api
from a_breakout_screener.holdings_api import (
    create_position,
    load_positions,
    positions_with_latest_quotes,
    save_positions,
    search_stocks,
)


def test_holdings_save_and_load_round_trip(tmp_path) -> None:
    data_file = tmp_path / "positions.json"
    position = create_position(
        {
            "code": "603912",
            "name": "佳力图",
            "buy_price": "11.57",
            "quantity": "200",
            "buy_date": "2026-05-18",
            "note": "test",
        }
    )

    save_positions(data_file, [position])
    loaded = load_positions(data_file)

    assert loaded[0]["code"] == "603912"
    assert loaded[0]["buy_price"] == 11.57
    assert loaded[0]["quantity"] == 200
    assert loaded[0]["buy_date"] == "2026-05-18"


def test_holdings_rejects_invalid_position() -> None:
    with pytest.raises(ValueError):
        create_position({"code": "603912", "buy_price": 0, "quantity": 100})


def test_holdings_enriches_positions_with_latest_quotes(monkeypatch) -> None:
    holdings_api._QUOTE_CACHE["expires_at"] = 0
    holdings_api._QUOTE_CACHE["quotes"] = {}
    holdings_api._QUOTE_CACHE["meta"] = {"status": "empty", "source": "akshare_spot_em"}
    holdings_api._QUOTE_CACHE["last_success_quotes"] = {}
    holdings_api._QUOTE_CACHE["last_success_meta"] = {}

    monkeypatch.setattr(
        holdings_api,
        "fetch_realtime_spot",
        lambda: pd.DataFrame(
            [
                {
                    "code": "603912",
                    "name": "佳力图",
                    "latest": 12.34,
                    "pct_change": 1.2,
                    "amount": 123456789,
                }
            ]
        ),
    )
    position = create_position({"code": "603912", "buy_price": 11.57, "quantity": 200})

    enriched, meta = positions_with_latest_quotes([position])

    assert meta["status"] == "ok"
    assert meta["requested"] == 1
    assert meta["matched"] == 1
    assert enriched[0]["latest_price"] == 12.34
    assert enriched[0]["price_source"] == "akshare_spot_em"
    assert enriched[0]["price_time"]


def test_stock_search_supports_code_and_name_fuzzy(monkeypatch) -> None:
    holdings_api._SEARCH_CACHE["expires_at"] = 0
    holdings_api._SEARCH_CACHE["items"] = []
    monkeypatch.setattr(holdings_api, "_read_stock_universe_cache", lambda cache_dir: [])
    monkeypatch.setattr(holdings_api, "_stock_items_from_tushare_basic", lambda: [])
    monkeypatch.setattr(holdings_api, "_write_stock_universe_cache", lambda cache_dir, items: None)

    monkeypatch.setattr(
        holdings_api,
        "fetch_spot",
        lambda: pd.DataFrame(
            [
                {"code": "000001", "name": "平安银行", "latest": 12.34, "pct_change": 1.2},
                {"code": "600519", "name": "贵州茅台", "latest": 1600.0, "pct_change": -0.5},
                {"code": "603912", "name": "佳力图", "latest": 11.57, "pct_change": 3.4},
            ]
        ),
    )

    by_code = search_stocks("6005", limit=5)
    by_name = search_stocks("平安", limit=5)
    by_spaced_name = search_stocks("平 银", limit=5)

    assert by_code[0]["code"] == "600519"
    assert by_name[0]["code"] == "000001"
    assert by_spaced_name[0]["code"] == "000001"
