from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from a_breakout_screener.backtest import (
    _backtest_signal_date,
    _long_hold_exit,
    _prepare_history,
    _run_first_signal_backtest,
    _summarize_long_hold,
)
from a_breakout_screener.config import AppConfig, ScreenerConfig
from a_breakout_screener.models import Candidate


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


def test_long_hold_exit_uses_stop_loss_price() -> None:
    history = pd.DataFrame(
        {
            "date": pd.bdate_range("2026-01-01", periods=4),
            "low": [10.0, 8.0, 4.8, 4.5],
            "close": [10.0, 8.5, 4.7, 4.6],
        }
    )

    exit_pos, reason, exit_price = _long_hold_exit(
        history=history,
        entry_pos=0,
        end_ts=pd.Timestamp("2026-01-06"),
        entry_price=10.0,
        stop_loss_fraction=0.5,
    )

    assert exit_pos == 2
    assert reason == "stop_loss"
    assert exit_price == pytest.approx(5.0)


def test_summarize_long_hold_totals_portfolio_result() -> None:
    trades = pd.DataFrame(
        [
            {"invested": 1000.0, "exit_value": 1200.0, "pnl": 200.0, "return_pct": 20.0, "exit_reason": "period_end"},
            {"invested": 1000.0, "exit_value": 500.0, "pnl": -500.0, "return_pct": -50.0, "exit_reason": "stop_loss"},
        ]
    )

    summary = _summarize_long_hold(
        trades,
        long_hold_filter_rows=[{"passed": 3}, {"passed": 0}],
        top_n=10,
        capital_per_trade=1000,
        stop_loss_pct=50,
        start_date="2025-05-13",
        end_date="2026-05-13",
    )
    row = summary.iloc[0]

    assert row["trades"] == 2
    assert row["total_invested"] == 2000
    assert row["ending_value"] == 1700
    assert row["total_pnl"] == -300
    assert row["total_return_pct"] == pytest.approx(-15)
    assert row["stopped_trades"] == 1


def test_first_signal_backtest_buys_same_stock_once(monkeypatch) -> None:
    dates = pd.bdate_range("2026-01-01", periods=8)
    history = pd.DataFrame(
        {
            "date": dates,
            "open": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            "high": [11.0] * 8,
            "low": [9.5] * 8,
            "close": [10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5],
            "volume": [1_000_000] * 8,
            "amount": [100_000_000] * 8,
        }
    )
    prepared = _prepare_history("000001", "平安银行", history)
    config = AppConfig(screener=ScreenerConfig(min_history_rows=1, min_amount=1, min_price=1))

    def always_signal(prepared, pos, config, resistance=None):
        return Candidate(
            code=prepared.code,
            name=prepared.name,
            latest_close=10.0,
            resistance=9.0,
            breakout_pct=0.03,
            volume_ratio=2.0,
            signal_type="B",
            score=80.0,
        )

    monkeypatch.setattr("a_breakout_screener.backtest._evaluate_prepared_at_pos", always_signal)
    monkeypatch.setattr(
        "a_breakout_screener.backtest.calc_resistance",
        lambda *args, **kwargs: SimpleNamespace(resistance=10.2),
    )

    trades, filters = _run_first_signal_backtest(
        trading_dates=[pd.Timestamp(item) for item in dates],
        prepared_histories=(prepared,),
        config=config,
        lookback_days=30,
        capital_per_trade=1000,
        stop_loss_pct=30,
    )

    assert len(trades) == 1
    assert trades[0]["code"] == "000001"
    assert trades[0]["invested"] == 1000
    assert filters[-1]["held_unique_codes"] == 1
