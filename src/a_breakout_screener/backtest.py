from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .config import AppConfig
from .data import _normalize_history
from .models import Candidate
from .scoring import _activity_series, _to_weekly, _weekly_span_mean, assess_candidate, calc_resistance


@dataclass(frozen=True)
class BacktestResult:
    output_dir: Path
    trades_path: Path
    summary_path: Path
    html_path: Path
    long_hold_trades_path: Path
    long_hold_summary_path: Path
    signal_days: int
    stock_count: int
    trade_count: int


@dataclass(frozen=True)
class PreparedHistory:
    code: str
    name: str
    history: pd.DataFrame
    weekly: pd.DataFrame


def run_backtest(
    config: AppConfig,
    days: int = 252,
    top_ns: tuple[int, ...] = (10, 20, 30),
    holding_days: tuple[int, ...] = (5, 10, 20),
    symbols: set[str] | None = None,
    long_hold_days: int = 365,
    long_hold_top_n: int = 10,
    capital_per_trade: float = 1000.0,
    stop_loss_pct: float = 50.0,
) -> BacktestResult:
    histories = _load_histories(config.paths.cache_dir, symbols=symbols)
    if not histories:
        raise RuntimeError(f"没有可回测的历史缓存: {config.paths.cache_dir / 'hist'}")

    names = _load_names_from_latest_daily(config.paths.cache_dir)
    trading_dates = _common_trading_dates(histories)
    if len(trading_dates) < config.screener.min_history_rows + max(holding_days) + 2:
        raise RuntimeError("历史数据太少，无法回测")
    prepared_histories = {
        code: _prepare_history(code, names.get(code, code), history, config.screener.ma_trend_period)
        for code, history in histories.items()
    }

    max_holding = max(holding_days)
    signal_dates = trading_dates[config.screener.min_history_rows : -max_holding - 1]
    signal_dates = signal_dates[-days:]
    max_top_n = max(top_ns)

    trades: list[dict[str, Any]] = []
    filter_rows: list[dict[str, Any]] = []
    total = len(signal_dates)
    prepared_list = tuple(prepared_histories.values())
    max_workers = max(1, config.screener.max_workers)
    if max_workers == 1 or total <= 1:
        for idx, signal_ts in enumerate(signal_dates, start=1):
            day_trades, checked, passed = _backtest_signal_date(
                signal_ts=signal_ts,
                prepared_histories=prepared_list,
                config=config,
                holding_days=holding_days,
                max_top_n=max_top_n,
                max_holding=max_holding,
            )
            trades.extend(day_trades)
            filter_rows.append({"signal_date": signal_ts.date().isoformat(), "total_checked": checked, "passed": passed})
            if idx == 1 or idx % 20 == 0 or idx == total:
                print(f"回测进度: {idx}/{total} 个信号日", flush=True)
    else:
        completed: list[tuple[int, tuple[list[dict[str, Any]], int, int]]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _backtest_signal_date,
                    signal_ts=signal_ts,
                    prepared_histories=prepared_list,
                    config=config,
                    holding_days=holding_days,
                    max_top_n=max_top_n,
                    max_holding=max_holding,
                ): idx
                for idx, signal_ts in enumerate(signal_dates, start=1)
            }
            for done_count, future in enumerate(as_completed(futures), start=1):
                completed.append((futures[future], future.result()))
                if done_count == 1 or done_count % 20 == 0 or done_count == total:
                    print(f"回测进度: {done_count}/{total} 个信号日", flush=True)
        for idx, (day_trades, checked, passed) in sorted(completed, key=lambda item: item[0]):
            trades.extend(day_trades)
            filter_rows.append({"signal_date": signal_dates[idx - 1].date().isoformat(), "total_checked": checked, "passed": passed})

    output_dir = config.paths.output_dir / "backtest" / date.today().isoformat()
    output_dir.mkdir(parents=True, exist_ok=True)
    trades_df = pd.DataFrame(trades)
    summary_df = _summarize(trades_df, top_ns=top_ns, holding_days=holding_days)
    long_hold_start_date = (trading_dates[-1] - pd.Timedelta(days=long_hold_days)).date().isoformat()
    long_hold_end_date = trading_dates[-1].date().isoformat()
    long_hold_trades, long_hold_filter_rows = _run_long_hold_backtest(
        trading_dates=trading_dates,
        prepared_histories=prepared_list,
        config=config,
        lookback_days=long_hold_days,
        top_n=long_hold_top_n,
        capital_per_trade=capital_per_trade,
        stop_loss_pct=stop_loss_pct,
    )
    long_hold_trades_df = pd.DataFrame(long_hold_trades)
    long_hold_summary_df = _summarize_long_hold(
        long_hold_trades_df,
        long_hold_filter_rows=long_hold_filter_rows,
        top_n=long_hold_top_n,
        capital_per_trade=capital_per_trade,
        stop_loss_pct=stop_loss_pct,
        start_date=long_hold_start_date,
        end_date=long_hold_end_date,
    )

    trades_path = output_dir / "backtest_trades.csv"
    summary_path = output_dir / "backtest_summary.csv"
    filters_path = output_dir / "backtest_filters.csv"
    long_hold_trades_path = output_dir / "long_hold_trades.csv"
    long_hold_summary_path = output_dir / "long_hold_summary.csv"
    long_hold_filters_path = output_dir / "long_hold_filters.csv"
    html_path = output_dir / "backtest_report.html"
    trades_df.to_csv(trades_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(filter_rows).to_csv(filters_path, index=False, encoding="utf-8-sig")
    long_hold_trades_df.to_csv(long_hold_trades_path, index=False, encoding="utf-8-sig")
    long_hold_summary_df.to_csv(long_hold_summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(long_hold_filter_rows).to_csv(long_hold_filters_path, index=False, encoding="utf-8-sig")
    html_path.write_text(
        _render_html(
            summary_df,
            trades_df,
            total,
            len(histories),
            long_hold_summary_df=long_hold_summary_df,
            long_hold_trades_df=long_hold_trades_df,
        ),
        encoding="utf-8",
    )

    return BacktestResult(
        output_dir=output_dir,
        trades_path=trades_path,
        summary_path=summary_path,
        html_path=html_path,
        long_hold_trades_path=long_hold_trades_path,
        long_hold_summary_path=long_hold_summary_path,
        signal_days=total,
        stock_count=len(histories),
        trade_count=len(trades) + len(long_hold_trades),
    )


def _backtest_signal_date(
    signal_ts: pd.Timestamp,
    prepared_histories: tuple[PreparedHistory, ...],
    config: AppConfig,
    holding_days: tuple[int, ...],
    max_top_n: int,
    max_holding: int,
) -> tuple[list[dict[str, Any]], int, int]:
    total_checked = 0
    daily_candidates: list[tuple[Candidate, pd.DataFrame, int]] = []
    for prepared in prepared_histories:
        history = prepared.history
        pos = history["date"].searchsorted(signal_ts, side="right") - 1
        if pos < config.screener.min_history_rows:
            continue
        if pd.Timestamp(history.iloc[pos]["date"]).normalize() != signal_ts.normalize():
            continue
        if pos + max_holding + 1 >= len(history):
            continue
        latest = history.iloc[pos]
        if _finite_float(latest.get("amount")) < config.screener.min_amount:
            continue
        if _finite_float(latest.get("close")) < config.screener.min_price:
            continue
        total_checked += 1
        candidate = _evaluate_prepared_at_pos(prepared, pos, config)
        if candidate:
            daily_candidates.append((candidate, history, pos))

    daily_candidates.sort(key=lambda item: item[0].score, reverse=True)
    passed = len(daily_candidates)
    trades: list[dict[str, Any]] = []
    for rank, (candidate, history, pos) in enumerate(daily_candidates[:max_top_n], start=1):
        entry_row = history.iloc[pos + 1]
        entry_price = _finite_float(entry_row.get("open"))
        if entry_price <= 0:
            continue
        for holding in holding_days:
            exit_row = history.iloc[pos + holding]
            exit_price = _finite_float(exit_row.get("close"))
            if exit_price <= 0:
                continue
            ret = exit_price / entry_price - 1
            trades.append(
                {
                    "signal_date": signal_ts.date().isoformat(),
                    "entry_date": pd.Timestamp(entry_row["date"]).date().isoformat(),
                    "exit_date": pd.Timestamp(exit_row["date"]).date().isoformat(),
                    "holding_days": holding,
                    "rank": rank,
                    "code": candidate.code,
                    "name": candidate.name,
                    "score": round(candidate.score, 4),
                    "breakout_pct": round(candidate.breakout_pct * 100, 4),
                    "volume_ratio": round(candidate.volume_ratio, 4),
                    "entry_price": round(entry_price, 4),
                    "exit_price": round(exit_price, 4),
                    "return_pct": round(ret * 100, 4),
                }
            )
    return trades, total_checked, passed


def _run_long_hold_backtest(
    trading_dates: list[pd.Timestamp],
    prepared_histories: tuple[PreparedHistory, ...],
    config: AppConfig,
    lookback_days: int,
    top_n: int,
    capital_per_trade: float,
    stop_loss_pct: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(trading_dates) < config.screener.min_history_rows + 3:
        return [], []
    end_ts = trading_dates[-1]
    start_ts = end_ts - pd.Timedelta(days=lookback_days)
    signal_dates = [
        ts for ts in trading_dates
        if ts >= start_ts and ts < end_ts
    ]
    trades: list[dict[str, Any]] = []
    filter_rows: list[dict[str, Any]] = []
    total = len(signal_dates)
    max_workers = max(1, config.screener.max_workers)
    if max_workers == 1 or total <= 1:
        for idx, signal_ts in enumerate(signal_dates, start=1):
            day_trades, checked, passed = _long_hold_signal_date(
                signal_ts=signal_ts,
                end_ts=end_ts,
                prepared_histories=prepared_histories,
                config=config,
                top_n=top_n,
                capital_per_trade=capital_per_trade,
                stop_loss_pct=stop_loss_pct,
            )
            trades.extend(day_trades)
            filter_rows.append({"signal_date": signal_ts.date().isoformat(), "total_checked": checked, "passed": passed})
            if idx == 1 or idx % 20 == 0 or idx == total:
                print(f"长持回测进度: {idx}/{total} 个信号日", flush=True)
    else:
        completed: list[tuple[int, tuple[list[dict[str, Any]], int, int]]] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _long_hold_signal_date,
                    signal_ts=signal_ts,
                    end_ts=end_ts,
                    prepared_histories=prepared_histories,
                    config=config,
                    top_n=top_n,
                    capital_per_trade=capital_per_trade,
                    stop_loss_pct=stop_loss_pct,
                ): idx
                for idx, signal_ts in enumerate(signal_dates, start=1)
            }
            for done_count, future in enumerate(as_completed(futures), start=1):
                completed.append((futures[future], future.result()))
                if done_count == 1 or done_count % 20 == 0 or done_count == total:
                    print(f"长持回测进度: {done_count}/{total} 个信号日", flush=True)
        for idx, (day_trades, checked, passed) in sorted(completed, key=lambda item: item[0]):
            trades.extend(day_trades)
            filter_rows.append({"signal_date": signal_dates[idx - 1].date().isoformat(), "total_checked": checked, "passed": passed})
    return trades, filter_rows


def _long_hold_signal_date(
    signal_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    prepared_histories: tuple[PreparedHistory, ...],
    config: AppConfig,
    top_n: int,
    capital_per_trade: float,
    stop_loss_pct: float,
) -> tuple[list[dict[str, Any]], int, int]:
    total_checked = 0
    daily_candidates: list[tuple[Candidate, pd.DataFrame, int]] = []
    for prepared in prepared_histories:
        history = prepared.history
        pos = history["date"].searchsorted(signal_ts, side="right") - 1
        if pos < config.screener.min_history_rows:
            continue
        if pd.Timestamp(history.iloc[pos]["date"]).normalize() != signal_ts.normalize():
            continue
        if pos + 1 >= len(history):
            continue
        latest = history.iloc[pos]
        if _finite_float(latest.get("amount")) < config.screener.min_amount:
            continue
        if _finite_float(latest.get("close")) < config.screener.min_price:
            continue
        total_checked += 1
        candidate = _evaluate_prepared_at_pos(prepared, pos, config)
        if candidate:
            daily_candidates.append((candidate, history, pos))

    daily_candidates.sort(key=lambda item: item[0].score, reverse=True)
    passed = len(daily_candidates)
    trades: list[dict[str, Any]] = []
    stop_loss_fraction = max(0.0, stop_loss_pct) / 100
    for rank, (candidate, history, pos) in enumerate(daily_candidates[:top_n], start=1):
        entry_pos = pos + 1
        entry_row = history.iloc[entry_pos]
        entry_price = _finite_float(entry_row.get("open"))
        if entry_price <= 0:
            continue
        exit_pos, exit_reason, exit_price = _long_hold_exit(history, entry_pos, end_ts, entry_price, stop_loss_fraction)
        exit_row = history.iloc[exit_pos]
        if exit_price <= 0:
            continue
        shares = capital_per_trade / entry_price
        exit_value = shares * exit_price
        pnl = exit_value - capital_per_trade
        ret = exit_value / capital_per_trade - 1
        trades.append(
            {
                "signal_date": signal_ts.date().isoformat(),
                "entry_date": pd.Timestamp(entry_row["date"]).date().isoformat(),
                "exit_date": pd.Timestamp(exit_row["date"]).date().isoformat(),
                "exit_reason": exit_reason,
                "rank": rank,
                "code": candidate.code,
                "name": candidate.name,
                "score": round(candidate.score, 4),
                "breakout_pct": round(candidate.breakout_pct * 100, 4),
                "volume_ratio": round(candidate.volume_ratio, 4),
                "entry_price": round(entry_price, 4),
                "exit_price": round(exit_price, 4),
                "shares": round(shares, 6),
                "invested": round(capital_per_trade, 2),
                "exit_value": round(exit_value, 2),
                "pnl": round(pnl, 2),
                "return_pct": round(ret * 100, 4),
            }
        )
    return trades, total_checked, passed


def _long_hold_exit(
    history: pd.DataFrame,
    entry_pos: int,
    end_ts: pd.Timestamp,
    entry_price: float,
    stop_loss_fraction: float,
) -> tuple[int, str, float]:
    end_pos = history["date"].searchsorted(end_ts, side="right") - 1
    end_pos = min(max(entry_pos, end_pos), len(history) - 1)
    stop_price = entry_price * (1 - stop_loss_fraction)
    for idx in range(entry_pos, end_pos + 1):
        low = _finite_float(history.iloc[idx].get("low"))
        close = _finite_float(history.iloc[idx].get("close"))
        if stop_loss_fraction > 0 and (low <= stop_price or close <= stop_price):
            return idx, "stop_loss", stop_price
    return end_pos, "period_end", _finite_float(history.iloc[end_pos].get("close"))


def _prepare_history(code: str, name: str, history: pd.DataFrame, ma_trend_period: int = 0) -> PreparedHistory:
    prepared = history.sort_values("date").reset_index(drop=True).copy()
    prepared["ma10"] = prepared["close"].rolling(10).mean()
    prepared["ma20"] = prepared["close"].rolling(20).mean()

    activity = _activity_series(prepared)
    volume_baseline = activity.shift(1).rolling(19).mean()
    prepared["bt_volume_ratio"] = activity / volume_baseline
    prepared["bt_volume_trend"] = activity.rolling(5).mean() / volume_baseline

    prev_close = prepared["close"].shift(1)
    tr = pd.concat(
        [
            prepared["high"] - prepared["low"],
            (prepared["high"] - prev_close).abs(),
            (prepared["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    prepared["bt_atr_pct"] = tr.rolling(14).mean() / prepared["close"]
    prepared["bt_monthly_span_pct"] = _monthly_span_series(prepared)

    if ma_trend_period > 0:
        prepared["ma_trend"] = prepared["close"].rolling(ma_trend_period).mean()
    else:
        prepared["ma_trend"] = float("nan")

    weekly = _to_weekly(prepared)
    return PreparedHistory(code=code, name=name, history=prepared, weekly=weekly)


def _monthly_span_series(daily: pd.DataFrame, months: int = 12, trading_rows: int = 260) -> pd.Series:
    dates = pd.to_datetime(daily["date"], errors="coerce")
    periods = list(dates.dt.to_period("M"))
    highs = pd.to_numeric(daily["high"], errors="coerce").to_numpy()
    lows = pd.to_numeric(daily["low"], errors="coerce").to_numpy()
    values: list[float] = []
    high_indexes: deque[int] = deque()
    low_indexes: deque[int] = deque()
    left = 0

    for idx, period in enumerate(periods):
        min_period = period - (months - 1)
        min_row = max(0, idx - trading_rows + 1)
        while left < min_row or periods[left] < min_period:
            left += 1
        while high_indexes and high_indexes[0] < left:
            high_indexes.popleft()
        while low_indexes and low_indexes[0] < left:
            low_indexes.popleft()

        high = float(highs[idx])
        low = float(lows[idx])
        if math.isfinite(high):
            while high_indexes and highs[high_indexes[-1]] <= high:
                high_indexes.pop()
            high_indexes.append(idx)
        if math.isfinite(low):
            while low_indexes and lows[low_indexes[-1]] >= low:
                low_indexes.pop()
            low_indexes.append(idx)

        if not high_indexes or not low_indexes:
            values.append(0.0)
            continue
        window_low = float(lows[low_indexes[0]])
        if window_low <= 0:
            values.append(0.0)
        else:
            values.append(float(highs[high_indexes[0]]) / window_low - 1)

    return pd.Series(values, index=daily.index)


def _evaluate_prepared_at_pos(prepared: PreparedHistory, pos: int, config: AppConfig) -> Candidate | None:
    params = config.screener
    history = prepared.history
    latest = history.iloc[pos]
    close = _finite_float(latest.get("close"))
    open_ = _finite_float(latest.get("open"))
    if close <= 0:
        return None

    resistance = calc_resistance(
        prepared.weekly,
        latest["date"],
        params.resistance_lookback_weeks,
        exclude_recent_weeks=params.resistance_exclude_recent_weeks,
    )
    if not resistance:
        return None

    breakout_pct = close / resistance.resistance - 1
    ma10 = _finite_float(latest.get("ma10"))
    ma20 = _finite_float(latest.get("ma20"))
    ma_trend = _finite_float(latest.get("ma_trend"), float("nan"))
    volume_ratio = _finite_float(latest.get("bt_volume_ratio"))
    volume_trend = _finite_float(latest.get("bt_volume_trend"))
    monthly_span_pct = _finite_float(latest.get("bt_monthly_span_pct"))
    atr_pct = _finite_float(latest.get("bt_atr_pct"))
    recent_low = _finite_float(history["low"].iloc[max(0, pos - 19) : pos + 1].min())
    weekly_span_mean = _weekly_span_mean(prepared.weekly, params.consolidation_weeks, latest["date"]) if params.consolidation_weeks > 0 else 0.0

    confirmation_closes: list[float] = []
    if params.confirmation_bars > 0:
        start = pos - params.confirmation_bars + 1
        if start >= 0:
            confirmation_closes = [float(c) for c in history["close"].iloc[start : pos + 1]]

    return assess_candidate(
        code=prepared.code,
        name=prepared.name,
        close=close,
        open_=open_,
        resistance=resistance.resistance,
        resistance_touches=resistance.touches,
        resistance_cluster_size=resistance.cluster_size,
        breakout_pct=breakout_pct,
        ma10=ma10,
        ma20=ma20,
        ma_trend=ma_trend,
        volume_ratio=volume_ratio,
        volume_trend=volume_trend,
        monthly_span_pct=monthly_span_pct,
        weekly_span_mean=weekly_span_mean,
        atr_pct=atr_pct,
        recent_low=recent_low,
        first_resistance_date=resistance.first_touch_date,
        last_resistance_date=resistance.last_touch_date,
        latest_trade_date=pd.Timestamp(latest["date"]).date(),
        confirmation_closes=confirmation_closes,
        params=params,
    )


def _finite_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def _load_histories(cache_dir: Path, symbols: set[str] | None = None) -> dict[str, pd.DataFrame]:
    hist_dir = cache_dir / "hist"
    normalized_symbols = {item.zfill(6) for item in symbols} if symbols else None
    histories: dict[str, pd.DataFrame] = {}
    for path in sorted(hist_dir.glob("*.csv")):
        if not path.stem.isdigit() or len(path.stem) != 6:
            continue
        code = path.stem
        if normalized_symbols and code not in normalized_symbols:
            continue
        try:
            history = _normalize_history(pd.read_csv(path))
        except Exception:
            continue
        if history.empty:
            continue
        histories[code] = history.sort_values("date").reset_index(drop=True)
    return histories


def _load_names_from_latest_daily(cache_dir: Path) -> dict[str, str]:
    daily_dir = cache_dir / "tushare_daily"
    paths = sorted(daily_dir.glob("*.csv"))
    if not paths:
        return {}
    try:
        latest = pd.read_csv(paths[-1])
    except Exception:
        return {}
    if "code" not in latest.columns:
        if "ts_code" not in latest.columns:
            return {}
        latest["code"] = latest["ts_code"].astype(str).str.extract(r"(\d{6})", expand=False).str.zfill(6)
    # Tushare daily cache does not carry stock names. Keep hook for future enriched caches.
    return {str(row["code"]).zfill(6): str(row.get("name", row["code"])) for _, row in latest.iterrows()}


def _common_trading_dates(histories: dict[str, pd.DataFrame]) -> list[pd.Timestamp]:
    counts: dict[pd.Timestamp, int] = {}
    for history in histories.values():
        for ts in history["date"].drop_duplicates():
            key = pd.Timestamp(ts).normalize()
            counts[key] = counts.get(key, 0) + 1
    threshold = min(len(histories), max(1, int(len(histories) * 0.5)))
    return sorted(ts for ts, count in counts.items() if count >= threshold)


def _summarize(
    trades_df: pd.DataFrame,
    top_ns: tuple[int, ...],
    holding_days: tuple[int, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if trades_df.empty:
        return pd.DataFrame()
    for top_n in top_ns:
        for holding in holding_days:
            subset = trades_df[(trades_df["rank"] <= top_n) & (trades_df["holding_days"] == holding)].copy()
            if subset.empty:
                continue
            returns = subset["return_pct"].astype(float)
            grouped = subset.groupby("signal_date")["return_pct"].mean().sort_index() / 100
            equity = (1 + grouped).cumprod()
            max_drawdown = _max_drawdown(equity)
            rows.append(
                {
                    "top_n": top_n,
                    "holding_days": holding,
                    "trades": int(len(subset)),
                    "signal_days": int(subset["signal_date"].nunique()),
                    "win_rate_pct": round(float((returns > 0).mean() * 100), 2),
                    "avg_return_pct": round(float(returns.mean()), 3),
                    "median_return_pct": round(float(returns.median()), 3),
                    "best_return_pct": round(float(returns.max()), 3),
                    "worst_return_pct": round(float(returns.min()), 3),
                    "avg_daily_bucket_pct": round(float(grouped.mean() * 100), 3),
                    "compounded_bucket_return_pct": round(float((equity.iloc[-1] - 1) * 100), 3),
                    "max_drawdown_pct": round(float(max_drawdown * 100), 3),
                }
            )
    return pd.DataFrame(rows).sort_values(["holding_days", "top_n"]).reset_index(drop=True)


def _summarize_long_hold(
    trades_df: pd.DataFrame,
    long_hold_filter_rows: list[dict[str, Any]],
    top_n: int,
    capital_per_trade: float,
    stop_loss_pct: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    signal_days = len(long_hold_filter_rows)
    passed_days = sum(1 for row in long_hold_filter_rows if int(row.get("passed", 0)) > 0)
    total_passed = sum(int(row.get("passed", 0)) for row in long_hold_filter_rows)
    base = {
        "start_date": start_date,
        "end_date": end_date,
        "top_n": top_n,
        "capital_per_trade": round(capital_per_trade, 2),
        "stop_loss_pct": round(stop_loss_pct, 2),
        "signal_days": signal_days,
        "days_with_candidates": passed_days,
        "raw_candidates": total_passed,
    }
    if trades_df.empty:
        return pd.DataFrame(
            [
                {
                    **base,
                    "trades": 0,
                    "total_invested": 0.0,
                    "ending_value": 0.0,
                    "total_pnl": 0.0,
                    "total_return_pct": 0.0,
                    "win_rate_pct": 0.0,
                    "stopped_trades": 0,
                    "stop_loss_rate_pct": 0.0,
                    "avg_return_pct": 0.0,
                    "best_return_pct": 0.0,
                    "worst_return_pct": 0.0,
                }
            ]
        )

    invested = trades_df["invested"].astype(float)
    exit_value = trades_df["exit_value"].astype(float)
    pnl = trades_df["pnl"].astype(float)
    returns = trades_df["return_pct"].astype(float)
    stopped = trades_df["exit_reason"].eq("stop_loss")
    total_invested = float(invested.sum())
    ending_value = float(exit_value.sum())
    total_pnl = float(pnl.sum())
    total_return_pct = total_pnl / total_invested * 100 if total_invested > 0 else 0.0
    return pd.DataFrame(
        [
            {
                **base,
                "trades": int(len(trades_df)),
                "total_invested": round(total_invested, 2),
                "ending_value": round(ending_value, 2),
                "total_pnl": round(total_pnl, 2),
                "total_return_pct": round(total_return_pct, 3),
                "win_rate_pct": round(float((returns > 0).mean() * 100), 2),
                "stopped_trades": int(stopped.sum()),
                "stop_loss_rate_pct": round(float(stopped.mean() * 100), 2),
                "avg_return_pct": round(float(returns.mean()), 3),
                "best_return_pct": round(float(returns.max()), 3),
                "worst_return_pct": round(float(returns.min()), 3),
            }
        ]
    )


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    drawdown = equity / peak - 1
    return abs(float(drawdown.min()))


def _render_html(
    summary: pd.DataFrame,
    trades: pd.DataFrame,
    signal_days: int,
    stock_count: int,
    long_hold_summary_df: pd.DataFrame,
    long_hold_trades_df: pd.DataFrame,
) -> str:
    summary_rows = summary.to_dict(orient="records")
    top_trades = trades.sort_values("return_pct", ascending=False).head(20).to_dict(orient="records") if not trades.empty else []
    worst_trades = trades.sort_values("return_pct", ascending=True).head(20).to_dict(orient="records") if not trades.empty else []
    long_hold_summary = long_hold_summary_df.to_dict(orient="records")
    long_hold_top = long_hold_trades_df.sort_values("pnl", ascending=False).head(20).to_dict(orient="records") if not long_hold_trades_df.empty else []
    long_hold_worst = long_hold_trades_df.sort_values("pnl", ascending=True).head(20).to_dict(orient="records") if not long_hold_trades_df.empty else []
    payload = {
        "summary": summary_rows,
        "topTrades": top_trades,
        "worstTrades": worst_trades,
        "longHoldSummary": long_hold_summary,
        "longHoldTopTrades": long_hold_top,
        "longHoldWorstTrades": long_hold_worst,
        "meta": {
            "signalDays": signal_days,
            "stockCount": stock_count,
            "tradeCount": int(len(trades)),
            "longHoldTradeCount": int(len(long_hold_trades_df)),
        },
    }
    return HTML_TEMPLATE.replace("__BACKTEST_DATA__", json.dumps(payload, ensure_ascii=False))


HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>A股突破策略回测</title>
  <style>
    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f7f9; color: #18202b; }
    header { padding: 24px 28px 14px; background: #fff; border-bottom: 1px solid #d9dee8; }
    h1 { margin: 0 0 12px; font-size: 24px; letter-spacing: 0; }
    .stats { display: flex; flex-wrap: wrap; gap: 10px; }
    .stat { min-width: 132px; padding: 10px 12px; border: 1px solid #d9dee8; border-radius: 8px; background: #fbfcfe; }
    .stat span { display: block; color: #667085; font-size: 12px; }
    .stat strong { display: block; margin-top: 4px; font-size: 18px; }
    main { padding: 16px; display: grid; gap: 16px; }
    section { background: #fff; border: 1px solid #d9dee8; border-radius: 8px; box-shadow: 0 10px 24px rgba(24,32,43,.08); overflow: hidden; }
    h2 { margin: 0; padding: 14px 16px; font-size: 16px; border-bottom: 1px solid #d9dee8; letter-spacing: 0; }
    .note { padding: 12px 16px; color: #667085; font-size: 13px; line-height: 1.6; border-bottom: 1px solid #edf0f5; }
    .wrap { overflow: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 9px 10px; border-bottom: 1px solid #edf0f5; text-align: right; white-space: nowrap; }
    th { background: #f9fafc; color: #667085; font-size: 12px; position: sticky; top: 0; }
    td.name, th.name { text-align: left; }
    .pos { color: #d33f49; font-weight: 700; }
    .neg { color: #16835f; font-weight: 700; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } header { padding: 18px 16px 12px; } main { padding: 12px; } }
  </style>
</head>
<body>
  <header>
    <h1>A股突破策略回测</h1>
    <div class="stats">
      <div class="stat"><span>信号日</span><strong id="signalDays"></strong></div>
      <div class="stat"><span>股票数</span><strong id="stockCount"></strong></div>
      <div class="stat"><span>短持交易</span><strong id="tradeCount"></strong></div>
      <div class="stat"><span>长持交易</span><strong id="longHoldTradeCount"></strong></div>
    </div>
  </header>
  <main>
    <section>
      <h2>年度长持口径</h2>
      <div class="note">口径：从一年前开始逐个交易日回放策略信号，每日取 Top10，下一交易日开盘每只买入 1000 元，持有到最新交易日；期间触发 -50% 止损则按止损价卖出。</div>
      <div class="wrap"><table id="longHoldSummaryTable"></table></div>
    </section>
    <section>
      <h2>参数组合表现</h2>
      <div class="note">口径：信号日收盘后选股，下一交易日开盘买入，持有 N 个交易日后收盘卖出；每个信号日内按 TopN 等权平均，仅用于粗筛策略有效性。</div>
      <div class="wrap"><table id="summaryTable"></table></div>
    </section>
    <div class="grid">
      <section>
        <h2>长持盈利最高</h2>
        <div class="wrap"><table id="longHoldTopTrades"></table></div>
      </section>
      <section>
        <h2>长持亏损最大</h2>
        <div class="wrap"><table id="longHoldWorstTrades"></table></div>
      </section>
      <section>
        <h2>收益最高交易</h2>
        <div class="wrap"><table id="topTrades"></table></div>
      </section>
      <section>
        <h2>亏损最大交易</h2>
        <div class="wrap"><table id="worstTrades"></table></div>
      </section>
    </div>
  </main>
  <script>
    const DATA = __BACKTEST_DATA__;
    document.getElementById("signalDays").textContent = DATA.meta.signalDays;
    document.getElementById("stockCount").textContent = DATA.meta.stockCount;
    document.getElementById("tradeCount").textContent = DATA.meta.tradeCount;
    document.getElementById("longHoldTradeCount").textContent = DATA.meta.longHoldTradeCount;
    function fmt(value, digits = 2) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
      return Number(value).toFixed(digits);
    }
    function cell(value, cls = "") {
      const td = document.createElement("td");
      td.textContent = value;
      if (cls) td.className = cls;
      return td;
    }
    function renderSummary(rows) {
      const table = document.getElementById("summaryTable");
      table.innerHTML = "<thead><tr><th>TopN</th><th>持有</th><th>交易数</th><th>胜率</th><th>平均收益</th><th>中位数</th><th>最好</th><th>最差</th><th>按日复利</th><th>最大回撤</th></tr></thead>";
      const body = document.createElement("tbody");
      rows.forEach(row => {
        const tr = document.createElement("tr");
        tr.append(
          cell(row.top_n),
          cell(`${row.holding_days}日`),
          cell(row.trades),
          cell(`${fmt(row.win_rate_pct)}%`),
          signed(row.avg_return_pct),
          signed(row.median_return_pct),
          signed(row.best_return_pct),
          signed(row.worst_return_pct),
          signed(row.compounded_bucket_return_pct),
          cell(`${fmt(row.max_drawdown_pct)}%`, "neg")
        );
        body.appendChild(tr);
      });
      table.appendChild(body);
    }
    function renderLongHoldSummary(rows) {
      const table = document.getElementById("longHoldSummaryTable");
      table.innerHTML = "<thead><tr><th>区间</th><th>TopN</th><th>单只买入</th><th>信号日</th><th>有候选日</th><th>交易数</th><th>总投入</th><th>期末/止损后市值</th><th>总收益</th><th>总收益率</th><th>胜率</th><th>止损数</th><th>止损率</th></tr></thead>";
      const body = document.createElement("tbody");
      rows.forEach(row => {
        const tr = document.createElement("tr");
        tr.append(
          cell(`${row.start_date} ~ ${row.end_date}`),
          cell(row.top_n),
          cell(fmt(row.capital_per_trade, 0)),
          cell(row.signal_days),
          cell(row.days_with_candidates),
          cell(row.trades),
          cell(fmt(row.total_invested, 2)),
          cell(fmt(row.ending_value, 2)),
          money(row.total_pnl),
          signed(row.total_return_pct),
          cell(`${fmt(row.win_rate_pct)}%`),
          cell(row.stopped_trades),
          cell(`${fmt(row.stop_loss_rate_pct)}%`)
        );
        body.appendChild(tr);
      });
      table.appendChild(body);
    }
    function signed(value) {
      return cell(`${fmt(value, 3)}%`, Number(value) >= 0 ? "pos" : "neg");
    }
    function money(value) {
      return cell(fmt(value, 2), Number(value) >= 0 ? "pos" : "neg");
    }
    function renderTrades(id, rows) {
      const table = document.getElementById(id);
      table.innerHTML = "<thead><tr><th>信号日</th><th class='name'>股票</th><th>持有</th><th>排名</th><th>得分</th><th>买入</th><th>卖出</th><th>收益</th></tr></thead>";
      const body = document.createElement("tbody");
      rows.forEach(row => {
        const tr = document.createElement("tr");
        tr.append(
          cell(row.signal_date),
          cell(`${row.code} ${row.name}`, "name"),
          cell(`${row.holding_days}日`),
          cell(row.rank),
          cell(fmt(row.score, 1)),
          cell(fmt(row.entry_price)),
          cell(fmt(row.exit_price)),
          signed(row.return_pct)
        );
        body.appendChild(tr);
      });
      table.appendChild(body);
    }
    function renderLongHoldTrades(id, rows) {
      const table = document.getElementById(id);
      table.innerHTML = "<thead><tr><th>信号日</th><th>买入日</th><th class='name'>股票</th><th>排名</th><th>退出</th><th>买入</th><th>卖出</th><th>投入</th><th>市值</th><th>收益</th><th>收益率</th></tr></thead>";
      const body = document.createElement("tbody");
      rows.forEach(row => {
        const tr = document.createElement("tr");
        tr.append(
          cell(row.signal_date),
          cell(row.entry_date),
          cell(`${row.code} ${row.name}`, "name"),
          cell(row.rank),
          cell(row.exit_reason === "stop_loss" ? "止损" : row.exit_date),
          cell(fmt(row.entry_price)),
          cell(fmt(row.exit_price)),
          cell(fmt(row.invested, 2)),
          cell(fmt(row.exit_value, 2)),
          money(row.pnl),
          signed(row.return_pct)
        );
        body.appendChild(tr);
      });
      table.appendChild(body);
    }
    renderLongHoldSummary(DATA.longHoldSummary);
    renderLongHoldTrades("longHoldTopTrades", DATA.longHoldTopTrades);
    renderLongHoldTrades("longHoldWorstTrades", DATA.longHoldWorstTrades);
    renderSummary(DATA.summary);
    renderTrades("topTrades", DATA.topTrades);
    renderTrades("worstTrades", DATA.worstTrades);
  </script>
</body>
</html>
"""
