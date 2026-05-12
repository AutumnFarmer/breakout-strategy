from __future__ import annotations

import pandas as pd

from a_breakout_screener.backtest import _backtest_signal_date, _prepare_history
from a_breakout_screener.config import AppConfig, ScreenerConfig


def test_prepare_history_uses_configured_trend_period() -> None:
    history = pd.DataFrame(
        {
            "date": pd.bdate_range("2026-01-01", periods=20),
            "open": range(1, 21),
            "high": range(2, 22),
            "low": range(0, 20),
            "close": range(1, 21),
            "volume": [1_000_000] * 20,
            "amount": [10_000_000] * 20,
        }
    )

    prepared = _prepare_history("000001", "平安银行", history, ma_trend_period=5).history

    assert prepared["ma_trend"].iloc[-1] == 18


def test_backtest_skips_stock_missing_signal_date() -> None:
    dates = pd.bdate_range("2026-01-01", periods=12)
    missing_signal = dates[6]
    traded_dates = dates.delete(6)
    history = pd.DataFrame(
        {
            "date": traded_dates,
            "open": [10.0] * len(traded_dates),
            "high": [10.5] * len(traded_dates),
            "low": [9.5] * len(traded_dates),
            "close": [10.0] * len(traded_dates),
            "volume": [1_000_000] * len(traded_dates),
            "amount": [100_000_000] * len(traded_dates),
        }
    )
    prepared = _prepare_history("000001", "平安银行", history)
    config = AppConfig(screener=ScreenerConfig(min_history_rows=2, min_amount=1, min_price=1))

    trades, checked, passed = _backtest_signal_date(
        signal_ts=missing_signal,
        prepared_histories=(prepared,),
        config=config,
        holding_days=(1,),
        max_top_n=1,
        max_holding=1,
    )

    assert trades == []
    assert checked == 0
    assert passed == 0
