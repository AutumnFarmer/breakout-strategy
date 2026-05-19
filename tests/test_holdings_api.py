from __future__ import annotations

import pytest
import pandas as pd

from a_breakout_screener import holdings_api
from a_breakout_screener.holdings_api import create_position, load_positions, positions_with_latest_quotes, save_positions


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
