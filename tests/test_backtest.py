from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from a_breakout_screener.backtest import (
    _backtest_signal_date,
    _is_backtest_week_confirmed,
    _load_backtest_calendar,
    _long_hold_exit,
    _previous_week_already_broke_out,
    _prepare_history,
    _render_html,
    _run_first_signal_backtest,
    _run_first_signal_executable_backtest,
    _summarize_first_signal_executable,
    _summarize_long_hold,
    calc_lot_position,
    run_first_signal_backtest,
    run_first_signal_executable_backtest,
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


def test_prepare_history_prefers_amount_for_activity_ratio() -> None:
    dates = pd.bdate_range("2026-01-01", periods=25)
    history = pd.DataFrame(
        {
            "date": dates,
            "open": [10.0] * 25,
            "high": [11.0] * 25,
            "low": [9.5] * 25,
            "close": [10.0] * 25,
            "volume": [1_000_000] * 25,
            "amount": [100_000_000] * 24 + [300_000_000],
        }
    )

    prepared = _prepare_history("000001", "平安银行", history).history

    assert prepared.iloc[-1]["bt_activity_source"] == "amount"
    assert prepared.iloc[-1]["bt_volume_ratio"] == pytest.approx(3.0)


def test_backtest_week_confirmation_uses_common_trading_dates() -> None:
    week_calendar = pd.DataFrame(
        {
            "cal_date": pd.to_datetime(["2026-05-11", "2026-05-13", "2026-05-15"]),
            "is_open": [True, True, True],
        }
    )
    short_week_calendar = pd.DataFrame(
        {
            "cal_date": pd.to_datetime(["2026-05-04", "2026-05-07"]),
            "is_open": [True, True],
        }
    )

    assert not _is_backtest_week_confirmed(pd.Timestamp("2026-05-13"), week_calendar)
    assert _is_backtest_week_confirmed(pd.Timestamp("2026-05-15"), week_calendar)
    assert _is_backtest_week_confirmed(pd.Timestamp("2026-05-07"), short_week_calendar)
    assert not _is_backtest_week_confirmed(pd.Timestamp("2026-05-07"), pd.DataFrame())


def test_backtest_calendar_derives_from_cached_trading_dates_without_network(tmp_path) -> None:
    config = AppConfig(paths=SimpleNamespace(cache_dir=tmp_path))
    trading_dates = [pd.Timestamp("2026-05-11"), pd.Timestamp("2026-05-13"), pd.Timestamp("2026-05-15")]

    calendar = _load_backtest_calendar(config, trading_dates)

    assert calendar["cal_date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-05-11", "2026-05-13", "2026-05-15"]
    assert calendar["is_open"].tolist() == [True, True, True]
    assert not _is_backtest_week_confirmed(pd.Timestamp("2026-05-13"), calendar)
    assert _is_backtest_week_confirmed(pd.Timestamp("2026-05-15"), calendar)


def test_previous_week_breakout_blocks_repeated_a_confirmation() -> None:
    weekly = pd.DataFrame(
        {
            "week_end": pd.to_datetime(["2026-04-24", "2026-05-01"]),
            "close": [9.8, 10.3],
        }
    )

    assert _previous_week_already_broke_out(weekly, pd.Timestamp("2026-05-07"), 10.0, 0.02)
    assert not _previous_week_already_broke_out(weekly, pd.Timestamp("2026-05-01"), 10.0, 0.02)


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


def test_calc_lot_position_rounds_down_to_whole_lots() -> None:
    shares, invested = calc_lot_position(18.0, 100, 5000)

    assert shares == 200
    assert invested == pytest.approx(3600)


def test_calc_lot_position_skips_when_one_lot_exceeds_budget() -> None:
    shares, invested = calc_lot_position(60.0, 100, 5000)

    assert shares == 0
    assert invested == 0


def test_calc_lot_position_includes_minimum_buy_fee_in_budget() -> None:
    shares, invested = calc_lot_position(10.0, 100, 1000, fee_rate=0.0003, min_fee=5)

    assert shares == 0
    assert invested == 0

    shares, invested = calc_lot_position(10.0, 100, 1005, fee_rate=0.0003, min_fee=5)

    assert shares == 100
    assert invested == pytest.approx(1000)


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

    def always_signal(prepared, pos, config, resistance=None, **kwargs):
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
        "a_breakout_screener.backtest._calc_configured_resistance",
        lambda *args, **kwargs: SimpleNamespace(
            resistance=10.2,
            zone_low=9.9,
            zone_mid=10.0,
            zone_upper=10.2,
            touches=3,
            cluster_size=3,
            span_weeks=20,
            first_touch_date=None,
            last_touch_date=None,
        ),
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


def test_executable_first_signal_limits_daily_buys(monkeypatch, tmp_path) -> None:
    dates = pd.bdate_range("2026-01-01", periods=5)
    prepared_histories = tuple(_sample_prepared(f"{idx:06d}", dates) for idx in range(1, 11))
    config = AppConfig(
        screener=ScreenerConfig(min_history_rows=1, min_amount=1, min_price=1),
        paths=SimpleNamespace(cache_dir=tmp_path),
    )

    def signal_on_second_day(prepared, pos, config, resistance=None, **kwargs):
        if pos != 1:
            return None
        score = 100 - int(prepared.code)
        return _candidate(prepared.code, prepared.name, score=score, signal_type="A")

    monkeypatch.setattr("a_breakout_screener.backtest._evaluate_prepared_at_pos", signal_on_second_day)

    trades, filters = _run_first_signal_executable_backtest(
        trading_dates=[pd.Timestamp(item) for item in dates],
        prepared_histories=prepared_histories,
        config=config,
        lookback_days=30,
        lot_size=100,
        max_capital_per_trade=1000,
        min_capital_per_trade=0,
        max_buys_per_day=3,
        max_theme_buys_per_day=99,
        slippage_bps=0,
        fee_bps=0,
        stop_loss_pct=30,
    )

    assert len(trades) == 3
    assert all(trade["shares"] % 100 == 0 for trade in trades)
    assert max(int(row["new_buys"]) for row in filters) == 3
    assert sum(int(row["skipped_daily_limit"]) for row in filters) == 7


def test_executable_first_signal_limits_theme_buys(monkeypatch, tmp_path) -> None:
    dates = pd.bdate_range("2026-01-01", periods=5)
    prepared_histories = tuple(_sample_prepared(f"{idx:06d}", dates) for idx in range(1, 6))
    config = AppConfig(
        screener=ScreenerConfig(min_history_rows=1, min_amount=1, min_price=1),
        paths=SimpleNamespace(cache_dir=tmp_path),
    )

    def same_theme_signal(prepared, pos, config, resistance=None, **kwargs):
        if pos != 1:
            return None
        return _candidate(prepared.code, prepared.name, score=90.0, signal_type="A", tags=("AI",))

    monkeypatch.setattr("a_breakout_screener.backtest._evaluate_prepared_at_pos", same_theme_signal)

    trades, filters = _run_first_signal_executable_backtest(
        trading_dates=[pd.Timestamp(item) for item in dates],
        prepared_histories=prepared_histories,
        config=config,
        lookback_days=30,
        lot_size=100,
        max_capital_per_trade=1000,
        min_capital_per_trade=0,
        max_buys_per_day=10,
        max_theme_buys_per_day=2,
        slippage_bps=0,
        fee_bps=0,
        stop_loss_pct=30,
    )

    assert len(trades) == 2
    assert {trade["primary_tag"] for trade in trades} == {"AI"}
    assert sum(int(row["skipped_theme_limit"]) for row in filters) == 3
    assert any(bool(row["theme_limit_applied"]) for row in filters)


def test_executable_first_signal_limits_unknown_theme_bucket(monkeypatch, tmp_path) -> None:
    dates = pd.bdate_range("2026-01-01", periods=5)
    prepared_histories = tuple(_sample_prepared(f"{idx:06d}", dates) for idx in range(1, 4))
    config = AppConfig(
        screener=ScreenerConfig(min_history_rows=1, min_amount=1, min_price=1),
        paths=SimpleNamespace(cache_dir=tmp_path),
    )

    def untagged_signal(prepared, pos, config, resistance=None, **kwargs):
        if pos != 1:
            return None
        return _candidate(prepared.code, prepared.name, score=90.0, signal_type="A")

    monkeypatch.setattr("a_breakout_screener.backtest._evaluate_prepared_at_pos", untagged_signal)

    trades, filters = _run_first_signal_executable_backtest(
        trading_dates=[pd.Timestamp(item) for item in dates],
        prepared_histories=prepared_histories,
        config=config,
        lookback_days=30,
        lot_size=100,
        max_capital_per_trade=1000,
        min_capital_per_trade=0,
        max_buys_per_day=10,
        max_theme_buys_per_day=1,
        slippage_bps=0,
        fee_bps=0,
        stop_loss_pct=30,
    )

    assert len(trades) == 1
    assert trades[0]["primary_tag"] == "UNKNOWN"
    assert sum(int(row["skipped_theme_limit"]) for row in filters) == 2


def test_executable_first_signal_does_not_buy_c1_by_default(monkeypatch, tmp_path) -> None:
    dates = pd.bdate_range("2026-01-01", periods=5)
    prepared = _sample_prepared("000001", dates)
    config = AppConfig(
        screener=ScreenerConfig(min_history_rows=1, min_amount=1, min_price=1),
        paths=SimpleNamespace(cache_dir=tmp_path),
    )

    def c1_signal(prepared, pos, config, resistance=None, **kwargs):
        if pos != 1:
            return None
        return _candidate(prepared.code, prepared.name, score=90.0, signal_type="C1")

    monkeypatch.setattr("a_breakout_screener.backtest._evaluate_prepared_at_pos", c1_signal)

    trades, filters = _run_first_signal_executable_backtest(
        trading_dates=[pd.Timestamp(item) for item in dates],
        prepared_histories=(prepared,),
        config=config,
        lookback_days=30,
        lot_size=100,
        max_capital_per_trade=1000,
        min_capital_per_trade=0,
        max_buys_per_day=10,
        max_theme_buys_per_day=1,
        slippage_bps=0,
        fee_bps=0,
        stop_loss_pct=30,
    )

    assert trades == []
    assert sum(int(row["skipped_unbuyable_signal_type"]) for row in filters) == 1


def test_executable_first_signal_does_not_buy_b_by_default(monkeypatch, tmp_path) -> None:
    dates = pd.bdate_range("2026-01-01", periods=5)
    prepared = _sample_prepared("000001", dates)
    config = AppConfig(
        screener=ScreenerConfig(min_history_rows=1, min_amount=1, min_price=1),
        paths=SimpleNamespace(cache_dir=tmp_path),
    )

    def b_signal(prepared, pos, config, resistance=None, **kwargs):
        if pos != 1:
            return None
        return _candidate(prepared.code, prepared.name, score=90.0, signal_type="B")

    monkeypatch.setattr("a_breakout_screener.backtest._evaluate_prepared_at_pos", b_signal)

    trades, filters = _run_first_signal_executable_backtest(
        trading_dates=[pd.Timestamp(item) for item in dates],
        prepared_histories=(prepared,),
        config=config,
        lookback_days=30,
        lot_size=100,
        max_capital_per_trade=1000,
        min_capital_per_trade=0,
        max_buys_per_day=10,
        max_theme_buys_per_day=1,
        slippage_bps=0,
        fee_bps=0,
        stop_loss_pct=30,
    )

    assert trades == []
    assert sum(int(row["raw_signal_B"]) for row in filters) == 1
    assert sum(int(row["raw_observation_candidates"]) for row in filters) == 1
    assert sum(int(row["skipped_unbuyable_signal_type"]) for row in filters) == 1


def test_executable_first_signal_slippage_and_fees_reduce_return(monkeypatch, tmp_path) -> None:
    dates = pd.bdate_range("2026-01-01", periods=5)
    prepared = _sample_prepared("000001", dates, entry_open=10.0, exit_close=12.0)
    config = AppConfig(
        screener=ScreenerConfig(min_history_rows=1, min_amount=1, min_price=1),
        paths=SimpleNamespace(cache_dir=tmp_path),
    )

    def one_signal(prepared, pos, config, resistance=None, **kwargs):
        if pos != 1:
            return None
        return _candidate(prepared.code, prepared.name, score=90.0, signal_type="A")

    monkeypatch.setattr("a_breakout_screener.backtest._evaluate_prepared_at_pos", one_signal)

    no_cost_trades, _ = _run_first_signal_executable_backtest(
        trading_dates=[pd.Timestamp(item) for item in dates],
        prepared_histories=(prepared,),
        config=config,
        lookback_days=30,
        lot_size=100,
        max_capital_per_trade=5000,
        min_capital_per_trade=0,
        max_buys_per_day=3,
        max_theme_buys_per_day=2,
        slippage_bps=0,
        fee_bps=0,
        stop_loss_pct=30,
    )
    cost_trades, filters = _run_first_signal_executable_backtest(
        trading_dates=[pd.Timestamp(item) for item in dates],
        prepared_histories=(prepared,),
        config=config,
        lookback_days=30,
        lot_size=100,
        max_capital_per_trade=5000,
        min_capital_per_trade=0,
        max_buys_per_day=3,
        max_theme_buys_per_day=2,
        slippage_bps=10,
        fee_bps=3,
        min_fee=5,
        sell_tax_bps=5,
        stop_loss_pct=30,
    )

    assert cost_trades[0]["return_pct"] < no_cost_trades[0]["return_pct"]
    assert cost_trades[0]["buy_fee"] == pytest.approx(5)
    assert cost_trades[0]["sell_fee"] > cost_trades[0]["buy_fee"]
    assert cost_trades[0]["activity_ratio"] == pytest.approx(2.0)
    summary = _summarize_first_signal_executable(
        pd.DataFrame(cost_trades),
        first_signal_filter_rows=filters,
        trading_dates=[pd.Timestamp(item) for item in dates],
        lot_size=100,
        max_capital_per_trade=5000,
        min_capital_per_trade=0,
        max_buys_per_day=3,
        max_theme_buys_per_day=2,
        slippage_bps=10,
        fee_bps=3,
        min_fee=5,
        sell_tax_bps=5,
        stop_loss_pct=30,
        start_date="2026-01-01",
        end_date="2026-01-07",
    )
    assert "peak_capital_used" in summary.columns
    assert summary.iloc[0]["peak_capital_used"] > 0


def test_executable_first_signal_counts_one_lot_too_expensive(monkeypatch, tmp_path) -> None:
    dates = pd.bdate_range("2026-01-01", periods=5)
    prepared = _sample_prepared("000001", dates, entry_open=60.0, exit_close=72.0)
    config = AppConfig(
        screener=ScreenerConfig(min_history_rows=1, min_amount=1, min_price=1),
        paths=SimpleNamespace(cache_dir=tmp_path),
    )

    def one_signal(prepared, pos, config, resistance=None, **kwargs):
        if pos != 1:
            return None
        return _candidate(prepared.code, prepared.name, score=90.0, signal_type="A")

    monkeypatch.setattr("a_breakout_screener.backtest._evaluate_prepared_at_pos", one_signal)

    trades, filters = _run_first_signal_executable_backtest(
        trading_dates=[pd.Timestamp(item) for item in dates],
        prepared_histories=(prepared,),
        config=config,
        lookback_days=30,
        lot_size=100,
        max_capital_per_trade=5000,
        min_capital_per_trade=0,
        max_buys_per_day=3,
        max_theme_buys_per_day=2,
        slippage_bps=0,
        fee_bps=0,
        stop_loss_pct=30,
    )

    assert trades == []
    assert sum(int(row["skipped_one_lot_too_expensive"]) for row in filters) == 1


def test_executable_first_signal_limits_total_capital(monkeypatch, tmp_path) -> None:
    dates = pd.bdate_range("2026-01-01", periods=5)
    prepared_histories = tuple(_sample_prepared(f"{idx:06d}", dates) for idx in range(1, 4))
    config = AppConfig(
        screener=ScreenerConfig(min_history_rows=1, min_amount=1, min_price=1),
        paths=SimpleNamespace(cache_dir=tmp_path),
    )

    def one_day_signal(prepared, pos, config, resistance=None, **kwargs):
        if pos != 1:
            return None
        return _candidate(prepared.code, prepared.name, score=90.0, signal_type="A")

    monkeypatch.setattr("a_breakout_screener.backtest._evaluate_prepared_at_pos", one_day_signal)

    trades, filters = _run_first_signal_executable_backtest(
        trading_dates=[pd.Timestamp(item) for item in dates],
        prepared_histories=prepared_histories,
        config=config,
        lookback_days=30,
        lot_size=100,
        max_capital_per_trade=1000,
        min_capital_per_trade=0,
        max_buys_per_day=10,
        max_theme_buys_per_day=99,
        slippage_bps=0,
        fee_bps=0,
        stop_loss_pct=30,
        max_total_capital=1500,
    )

    assert len(trades) == 1
    assert trades[0]["capital_reserved"] == pytest.approx(1000)
    assert sum(int(row["skipped_total_capital_limit"]) for row in filters) == 2


def test_research_first_signal_public_output_keeps_legacy_files(monkeypatch, tmp_path) -> None:
    dates = pd.bdate_range("2026-01-01", periods=5)
    history = _sample_history(dates)
    config = AppConfig(
        screener=ScreenerConfig(min_history_rows=1, min_amount=1, min_price=1),
        paths=SimpleNamespace(cache_dir=tmp_path / "cache", output_dir=tmp_path / "outputs"),
    )

    monkeypatch.setattr("a_breakout_screener.backtest._load_histories", lambda cache_dir, symbols=None: {"000001": history})
    monkeypatch.setattr("a_breakout_screener.backtest._load_names_from_latest_daily", lambda cache_dir: {"000001": "平安银行"})
    monkeypatch.setattr("a_breakout_screener.backtest._load_backtest_calendar", lambda config, trading_dates: pd.DataFrame())
    monkeypatch.setattr(
        "a_breakout_screener.backtest._run_first_signal_backtest",
        lambda **kwargs: ([], [{"signal_date": "2026-01-02", "passed": 0, "new_buys": 0}]),
    )

    result = run_first_signal_backtest(config, lookback_days=30)

    assert result.output_dir.name != "first_signal_executable"
    assert result.trades_path.name == "first_signal_trades.csv"
    assert result.summary_path.name == "first_signal_summary.csv"
    assert result.filters_path.name == "first_signal_filters.csv"
    assert result.html_path.name == "backtest_report.html"
    assert result.trades_path.exists()
    assert result.summary_path.exists()
    assert result.filters_path.exists()
    assert result.html_path.exists()
    assert not (result.output_dir / "first_signal_executable_trades.csv").exists()


def test_executable_first_signal_public_output_uses_isolated_files(monkeypatch, tmp_path) -> None:
    dates = pd.bdate_range("2026-01-01", periods=5)
    history = _sample_history(dates)
    config = AppConfig(
        screener=ScreenerConfig(min_history_rows=1, min_amount=1, min_price=1),
        paths=SimpleNamespace(cache_dir=tmp_path / "cache", output_dir=tmp_path / "outputs"),
    )

    monkeypatch.setattr("a_breakout_screener.backtest._load_histories", lambda cache_dir, symbols=None: {"000001": history})
    monkeypatch.setattr("a_breakout_screener.backtest._load_names_from_latest_daily", lambda cache_dir: {"000001": "平安银行"})
    monkeypatch.setattr("a_breakout_screener.backtest._load_backtest_calendar", lambda config, trading_dates: pd.DataFrame())
    monkeypatch.setattr(
        "a_breakout_screener.backtest._run_first_signal_executable_backtest",
        lambda **kwargs: ([], [{"signal_date": "2026-01-02", "passed": 0, "raw_candidates": 0}]),
    )

    result = run_first_signal_executable_backtest(config, lookback_days=30)

    assert result.output_dir.name == "first_signal_executable"
    assert result.trades_path.name == "first_signal_executable_trades.csv"
    assert result.summary_path.name == "first_signal_executable_summary.csv"
    assert result.filters_path.name == "first_signal_executable_filters.csv"
    assert result.html_path.name == "first_signal_executable_report.html"
    assert result.trades_path.exists()
    assert result.summary_path.exists()
    assert result.filters_path.exists()
    assert result.html_path.exists()
    assert not (result.output_dir.parent / "first_signal_trades.csv").exists()
    assert not (result.output_dir.parent / "first_signal_summary.csv").exists()
    assert not (result.output_dir.parent / "first_signal_filters.csv").exists()
    assert not (result.output_dir.parent / "backtest_report.html").exists()


def test_research_html_uses_research_first_signal_wording() -> None:
    html = _render_html(
        pd.DataFrame(),
        pd.DataFrame(),
        signal_days=1,
        stock_count=1,
        long_hold_summary_df=pd.DataFrame(),
        long_hold_trades_df=pd.DataFrame(),
        first_signal_summary_df=pd.DataFrame(),
        first_signal_trades_df=pd.DataFrame(),
    )

    assert "固定金额研究口径" in html
    assert "单只买入" in html
    assert "first-signal executable backtest" not in html
    assert "整手买入" not in html
    assert "每日限流" not in html
    assert "峰值资金占用" not in html


def test_executable_html_uses_executable_first_signal_wording() -> None:
    summary = pd.DataFrame(
        [
            {
                "start_date": "2026-01-01",
                "end_date": "2026-01-07",
                "capital_per_trade": 5000.0,
                "max_capital_per_trade": 5000.0,
                "lot_size": 100,
                "max_buys_per_day": 3,
                "max_theme_buys_per_day": 2,
                "slippage_bps": 10.0,
                "fee_bps": 3.0,
                "min_fee": 5.0,
                "sell_tax_bps": 5.0,
                "buy_signal_types": "A",
                "stop_loss_pct": 30.0,
                "signal_days": 1,
                "days_with_candidates": 1,
                "raw_first_signal_candidates": 1,
                "buyable_first_signal_candidates": 1,
                "observation_first_signal_candidates": 0,
                "trades": 1,
                "total_invested": 1000.0,
                "ending_value": 1100.0,
                "total_pnl": 100.0,
                "total_return_pct": 10.0,
                "win_rate_pct": 100.0,
                "stopped_trades": 0,
                "stop_loss_rate_pct": 0.0,
                "peak_capital_used": 1000.0,
            }
        ]
    )

    trades = pd.DataFrame(
        [
            {
                "signal_date": "2026-01-02",
                "entry_date": "2026-01-05",
                "exit_date": "2026-01-07",
                "rank": 1,
                "code": "000001",
                "name": "平安银行",
                "score": 90.0,
                "entry_price": 10.0,
                "exit_price": 11.0,
                "shares": 100,
                "buy_fee": 1.0,
                "sell_fee": 1.1,
                "capital_reserved": 1001.0,
                "pnl": 97.9,
                "return_pct": 9.79,
            }
        ]
    )

    html = _render_html(
        pd.DataFrame(),
        pd.DataFrame(),
        signal_days=1,
        stock_count=1,
        long_hold_summary_df=pd.DataFrame(),
        long_hold_trades_df=pd.DataFrame(),
        first_signal_summary_df=summary,
        first_signal_trades_df=trades,
        report_mode="first_signal_executable",
    )

    assert "first-signal executable backtest" in html
    assert "整手买入" in html
    assert "每日限流" in html
    assert "峰值资金占用" in html
    assert "max_buys_per_day" in html
    assert "lot_size" in html
    assert "buy_signal_types" in html
    assert "sell_tax_bps" in html
    assert "peak_capital_used" in html
    assert "undefined日" not in html
    assert "capital_reserved" in html


def _sample_history(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": dates,
            "open": [10.0, 10.0, 10.0, 11.0, 11.5],
            "high": [11.0, 11.0, 11.0, 12.5, 12.5],
            "low": [9.5, 9.5, 9.5, 10.5, 11.0],
            "close": [10.0, 10.5, 11.0, 11.5, 12.0],
            "volume": [1_000_000] * len(dates),
            "amount": [100_000_000] * len(dates),
        }
    )


def _sample_prepared(
    code: str,
    dates: pd.DatetimeIndex,
    entry_open: float = 10.0,
    exit_close: float = 12.0,
) -> object:
    history = pd.DataFrame(
        {
            "date": dates,
            "open": [10.0, 10.0, entry_open, 11.0, 11.5],
            "high": [11.0, 11.0, max(entry_open, 11.0), 12.5, 12.5],
            "low": [9.5, 9.5, min(entry_open, 10.0), 10.5, 11.0],
            "close": [10.0, 10.5, 11.0, 11.5, exit_close],
            "volume": [1_000_000] * len(dates),
            "amount": [100_000_000] * len(dates),
        }
    )
    return _prepare_history(code, code, history)


def _candidate(
    code: str,
    name: str,
    score: float,
    signal_type: str,
    tags: tuple[str, ...] = (),
) -> Candidate:
    return Candidate(
        code=code,
        name=name,
        latest_close=10.5,
        resistance=10.0,
        breakout_pct=0.05,
        volume_ratio=2.0,
        signal_type=signal_type,
        score=score,
        tags=tags,
    )
