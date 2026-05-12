from __future__ import annotations

import pandas as pd

from a_breakout_screener.backtest import _prepare_history


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
