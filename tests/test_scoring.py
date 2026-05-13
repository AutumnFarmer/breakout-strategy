from __future__ import annotations

import pandas as pd

from a_breakout_screener.config import ScreenerConfig
from a_breakout_screener.scoring import _activity_series, calc_resistance, evaluate_stock


def test_evaluate_stock_accepts_fresh_weekly_breakout() -> None:
    history = _sample_history(latest_close=12.04, latest_volume=2_000_000)
    candidate = evaluate_stock("000001", "平安银行", history, ScreenerConfig(min_history_rows=120))

    assert candidate is not None
    assert candidate.code == "000001"
    assert candidate.resistance >= 11.5
    assert 0 <= candidate.breakout_pct <= 0.12
    assert candidate.volume_ratio > 1
    assert candidate.score > 50
    assert candidate.signal_type in {"A", "B", "C", "D"}
    assert candidate.trade_stop_loss == candidate.resistance * 0.97
    assert "建议仓位约" not in candidate.position_hint


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
        ScreenerConfig(min_history_rows=120, ma_trend_period=300),
    )

    assert candidate is None


def test_calc_resistance_excludes_recent_weeks() -> None:
    week_ends = pd.date_range("2025-10-24", periods=30, freq="W-FRI")
    weekly = pd.DataFrame(
        {
            "week_end": week_ends,
            "open": 9.8,
            "high": 10.0,
            "low": 9.5,
            "close": 9.9,
            "volume": 1_000_000,
            "amount": 10_000_000,
        }
    )
    weekly.loc[weekly.index[-5:], "high"] = 13.0
    weekly.loc[weekly.index[-5:], "close"] = 12.8

    latest = pd.Timestamp("2026-05-15")
    excluded = calc_resistance(weekly, latest, lookback_weeks=52, exclude_recent_weeks=4, top_k=4)
    included = calc_resistance(weekly, latest, lookback_weeks=52, exclude_recent_weeks=0, top_k=4)

    assert excluded is not None
    assert included is not None
    assert excluded.resistance < 11
    assert included.resistance > 12


def test_activity_series_falls_back_to_volume_when_amount_units_are_mixed() -> None:
    daily = pd.DataFrame(
        {
            "volume": [10_000] * 15 + [20_000] * 5,
            "amount": [30_000_000] * 15 + [20_000] * 5,
        }
    )

    activity = _activity_series(daily)

    assert activity.equals(daily["volume"])


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
