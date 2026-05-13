from __future__ import annotations

import pandas as pd

from a_breakout_screener.config import ScreenerConfig
from a_breakout_screener.scoring import evaluate_stock


def test_evaluate_stock_accepts_fresh_weekly_breakout() -> None:
    history = _sample_history(latest_close=12.5, latest_volume=2_000_000)
    candidate = evaluate_stock("000001", "平安银行", history, ScreenerConfig(min_history_rows=120))

    assert candidate is not None
    assert candidate.code == "000001"
    assert candidate.zone_low < candidate.zone_mid < candidate.zone_upper
    assert candidate.resistance == candidate.zone_upper
    assert candidate.resistance_touches >= 3
    assert candidate.span_weeks >= 20
    assert 0 <= candidate.breakout_pct <= 0.12
    assert candidate.signal_type in {"A", "B"}
    assert candidate.volume_ratio > 1
    assert candidate.score > 50


def test_evaluate_stock_rejects_overextended_breakout() -> None:
    history = _sample_history(latest_close=14.5, latest_volume=2_000_000)
    candidate = evaluate_stock("000001", "平安银行", history, ScreenerConfig(min_history_rows=120, max_extension=0.12))

    assert candidate is None


def test_evaluate_stock_rejects_when_trend_ma_has_insufficient_history() -> None:
    history = _sample_history(latest_close=12.2, latest_volume=2_000_000)
    candidate = evaluate_stock(
        "000001",
        "平安银行",
        history,
        ScreenerConfig(min_history_rows=120, ma_trend_period=900),
    )

    assert candidate is None


def _sample_history(latest_close: float, latest_volume: int) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=820)
    rows = []
    for idx, trade_date in enumerate(dates):
        close = 8.8 + idx * 0.002
        high = close + 0.16
        low = close - 0.18
        if idx in {120, 245, 370, 520}:
            high = 11.8
            close = 11.55
            low = 11.1
        volume = 1_000_000
        rows.append(
            {
                "date": trade_date,
                "open": close - 0.05,
                "close": close,
                "high": high,
                "low": low,
                "volume": volume,
                "amount": volume * close,
            }
        )
    rows[-1]["open"] = latest_close * 0.98
    rows[-1]["close"] = latest_close
    rows[-1]["high"] = latest_close * 1.01
    rows[-1]["low"] = latest_close * 0.97
    rows[-1]["volume"] = latest_volume
    rows[-1]["amount"] = latest_volume * latest_close
    return pd.DataFrame(rows)
