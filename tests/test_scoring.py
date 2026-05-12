from __future__ import annotations

import pandas as pd

from a_breakout_screener.config import ScreenerConfig
from a_breakout_screener.scoring import evaluate_stock


def test_evaluate_stock_accepts_fresh_weekly_breakout() -> None:
    history = _sample_history(latest_close=12.2, latest_volume=2_000_000)
    candidate = evaluate_stock("000001", "平安银行", history, ScreenerConfig(min_history_rows=120))

    assert candidate is not None
    assert candidate.code == "000001"
    assert candidate.resistance >= 11.5
    assert 0 <= candidate.breakout_pct <= 0.12
    assert candidate.volume_ratio > 1
    assert candidate.score > 50


def test_evaluate_stock_rejects_overextended_breakout() -> None:
    history = _sample_history(latest_close=14.5, latest_volume=2_000_000)
    candidate = evaluate_stock("000001", "平安银行", history, ScreenerConfig(min_history_rows=120, max_extension=0.12))

    assert candidate is None


def _sample_history(latest_close: float, latest_volume: int) -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-01", periods=260)
    rows = []
    for idx, trade_date in enumerate(dates):
        close = 9.5 + idx * 0.006
        high = close + 0.18
        low = close - 0.18
        if 90 <= idx <= 170 and idx % 20 == 0:
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
