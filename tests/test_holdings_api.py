from __future__ import annotations

import pytest

from a_breakout_screener.holdings_api import create_position, load_positions, save_positions


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
