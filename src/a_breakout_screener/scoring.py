from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .config import ScreenerConfig
from .models import Candidate


@dataclass(frozen=True)
class ResistanceInfo:
    resistance: float
    touches: int
    first_touch_date: date | None
    last_touch_date: date | None
    cluster_size: int = 0


def evaluate_stock(code: str, name: str, history: pd.DataFrame, params: ScreenerConfig) -> Candidate | None:
    if len(history) < params.min_history_rows:
        return None
    daily = history.sort_values("date").reset_index(drop=True).copy()
    latest = daily.iloc[-1]
    close = float(latest["close"])
    open_ = float(latest["open"])
    if close <= 0:
        return None

    weekly = _to_weekly(daily)
    resistance = calc_resistance(weekly, latest["date"], params.resistance_lookback_weeks)
    if resistance is None or resistance.resistance <= 0:
        return None

    breakout_pct = close / resistance.resistance - 1

    ma10 = float(daily["close"].tail(10).mean())
    ma20 = float(daily["close"].tail(20).mean())
    ma_trend = float(daily["close"].tail(params.ma_trend_period).mean()) if params.ma_trend_period > 0 and len(daily) >= params.ma_trend_period else float("nan")
    volume_ratio = _volume_ratio(daily)
    volume_trend = _volume_trend(daily)
    monthly_span_pct = _monthly_span_pct(daily)
    weekly_span_mean = _weekly_span_mean(weekly, params.consolidation_weeks, latest["date"]) if params.consolidation_weeks > 0 else 0.0
    atr_pct = _calc_atr(daily)

    return assess_candidate(
        code=code,
        name=name,
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
        first_resistance_date=resistance.first_touch_date,
        last_resistance_date=resistance.last_touch_date,
        latest_trade_date=latest["date"].date(),
        confirmation_closes=list(daily["close"].tail(params.confirmation_bars)) if params.confirmation_bars > 0 else [],
        params=params,
    )


def assess_candidate(
    *,
    code: str,
    name: str,
    close: float,
    open_: float,
    resistance: float,
    resistance_touches: int,
    resistance_cluster_size: int,
    breakout_pct: float,
    ma10: float,
    ma20: float,
    ma_trend: float,
    volume_ratio: float,
    volume_trend: float,
    monthly_span_pct: float,
    weekly_span_mean: float,
    atr_pct: float,
    first_resistance_date: date | None,
    last_resistance_date: date | None,
    latest_trade_date: date,
    confirmation_closes: list[float],
    params: ScreenerConfig,
) -> Candidate | None:
    if breakout_pct < params.breakout_buffer or breakout_pct > params.max_extension:
        return None

    if params.confirmation_bars > 0:
        if len(confirmation_closes) < params.confirmation_bars:
            return None
        if any(c < resistance for c in confirmation_closes):
            return None

    if params.ma_trend_period > 0:
        if not np.isfinite(ma_trend):
            return None
        if close < ma_trend:
            return None

    if params.max_open_gap_pct > 0:
        gap_pct = open_ / resistance - 1
        if gap_pct > params.max_open_gap_pct:
            return None

    score = calc_score(
        close=close,
        breakout_pct=breakout_pct,
        ma10=ma10,
        ma20=ma20,
        volume_ratio=volume_ratio,
        volume_trend=volume_trend,
        resistance_touches=resistance_touches,
        resistance_cluster_size=resistance_cluster_size,
        monthly_span_pct=monthly_span_pct,
        weekly_span_mean=weekly_span_mean,
        params=params,
    )
    if score <= 0:
        return None

    buy_zone_low = resistance * 0.995
    buy_zone_high = min(resistance * 1.025, close * 1.01)
    stop_loss = min(resistance * 0.96, ma20 * 0.98)

    return Candidate(
        code=code,
        name=name,
        latest_close=close,
        resistance=resistance,
        breakout_pct=breakout_pct,
        volume_ratio=volume_ratio,
        volume_trend=volume_trend,
        resistance_touches=resistance_touches,
        resistance_cluster_size=resistance_cluster_size,
        monthly_span_pct=monthly_span_pct,
        ma10=ma10,
        ma20=ma20,
        atr_pct=atr_pct,
        score=score,
        first_resistance_date=first_resistance_date,
        last_resistance_date=last_resistance_date,
        latest_trade_date=latest_trade_date,
        buy_zone_low=buy_zone_low,
        buy_zone_high=buy_zone_high,
        stop_loss=stop_loss,
        position_hint=_position_hint(score, breakout_pct, volume_ratio, atr_pct),
    )


def calc_resistance(
    weekly: pd.DataFrame,
    latest_trade_ts: pd.Timestamp,
    lookback_weeks: int,
    touch_tolerance: float = 0.02,
    cluster_tolerance: float = 0.03,
    top_k: int = 8,
) -> ResistanceInfo | None:
    if len(weekly) < 20:
        return None
    latest_week_end = latest_trade_ts.to_period("W-FRI").end_time.normalize()
    prior = weekly[weekly["week_end"] < latest_week_end].tail(lookback_weeks)
    if len(prior) < 12:
        return None

    top_highs = prior["high"].nlargest(top_k).sort_values()
    if top_highs.empty:
        return None

    clusters: list[list[float]] = []
    current: list[float] = []
    for h in top_highs:
        if not current:
            current.append(h)
        elif h <= current[-1] * (1 + cluster_tolerance):
            current.append(h)
        else:
            clusters.append(current)
            current = [h]
    if current:
        clusters.append(current)

    best = max(clusters, key=lambda c: (len(c), np.median(c)))
    if len(best) == 1:
        # Isolated spike — fall back to the overall max high as resistance.
        resistance = float(prior["high"].max())
        return ResistanceInfo(resistance=resistance, touches=1, first_touch_date=None, last_touch_date=None, cluster_size=1)
    resistance = float(np.median(best))
    if not np.isfinite(resistance) or resistance <= 0:
        return None

    near = prior[
        (prior["high"] >= resistance * (1 - touch_tolerance))
        | (prior["close"] >= resistance * (1 - touch_tolerance))
    ]
    if near.empty:
        return ResistanceInfo(resistance=resistance, touches=0, first_touch_date=None, last_touch_date=None, cluster_size=len(best))
    return ResistanceInfo(
        resistance=resistance,
        touches=int(len(near)),
        first_touch_date=near.iloc[0]["week_end"].date(),
        last_touch_date=near.iloc[-1]["week_end"].date(),
        cluster_size=len(best),
    )


def calc_score(
    close: float,
    breakout_pct: float,
    ma10: float,
    ma20: float,
    volume_ratio: float,
    volume_trend: float,
    resistance_touches: int,
    resistance_cluster_size: int,
    monthly_span_pct: float,
    weekly_span_mean: float,
    params: ScreenerConfig,
) -> float:
    if close <= 0 or breakout_pct < params.breakout_buffer or breakout_pct > params.max_extension:
        return 0.0

    price_score = _piecewise(
        breakout_pct,
        [
            (0.00, 22.0),
            (0.02, 30.0),
            (0.05, 24.0),
            (params.max_extension, 8.0),
        ],
    )
    trend_score = 0.0
    if close >= ma10 >= ma20:
        trend_score = 10.0
    elif close >= ma20:
        trend_score = 6.0
    elif close >= ma10:
        trend_score = 4.0

    volume_score = _piecewise(
        min(max(volume_ratio, 0.0), 4.5),
        [
            (0.0, 0.0),
            (1.0, 8.0),
            (1.8, 20.0),
            (3.0, 16.0),
            (4.5, 8.0),
        ],
    )
    volume_trend_score = _piecewise(
        min(max(volume_trend, 0.0), 4.0),
        [
            (0.0, 0.0),
            (1.0, 6.0),
            (1.5, 10.0),
            (3.0, 6.0),
            (4.0, 4.0),
        ],
    )

    raw_resistance_score = min(20.0, max(0.0, resistance_touches - 1) * 4.0)
    resistance_score = raw_resistance_score * 0.5 if resistance_cluster_size == 1 else raw_resistance_score

    # Continuous penalty: linear from 1.0 (0 pts) to 1.8 (8 pts), capped
    span_penalty = 0.0
    if monthly_span_pct > 1.0:
        span_penalty = max(0.0, min(8.0, (monthly_span_pct - 1.0) * 10.0))

    # Consolidation penalty: penalize stocks with wide pre-breakout weekly ranges
    consolidation_penalty = 0.0
    if params.consolidation_weeks > 0 and params.consolidation_max_span > 0 and weekly_span_mean > 0:
        if weekly_span_mean > params.consolidation_max_span:
            excess = weekly_span_mean - params.consolidation_max_span
            consolidation_penalty = max(0.0, min(params.consolidation_penalty_weight, excess * params.consolidation_penalty_weight / params.consolidation_max_span))

    return max(0.0, price_score + trend_score + volume_score + volume_trend_score + resistance_score - span_penalty - consolidation_penalty)


def _to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    indexed = daily.set_index("date").sort_index()
    weekly = indexed.resample("W-FRI").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "amount": "sum",
        }
    )
    weekly = weekly.dropna(subset=["open", "high", "low", "close"]).reset_index()
    weekly = weekly.rename(columns={"date": "week_end"})
    return weekly


def _weekly_span_mean(weekly: pd.DataFrame, weeks: int, latest_ts: pd.Timestamp) -> float:
    prior = weekly[weekly["week_end"] < latest_ts].tail(weeks)
    if len(prior) < 3:
        return 0.0
    spans = (prior["high"] - prior["low"]) / prior["low"]
    return float(spans.mean())


def _volume_ratio(daily: pd.DataFrame) -> float:
    latest_volume = float(daily.iloc[-1].get("volume", 0) or 0)
    baseline = float(daily["volume"].tail(20).iloc[:-1].mean())
    if baseline <= 0 or not np.isfinite(baseline):
        return 0.0
    return latest_volume / baseline


def _volume_trend(daily: pd.DataFrame) -> float:
    tail = daily["volume"].tail(20)
    short = float(tail.iloc[-5:].mean())
    long = float(tail.iloc[:-1].mean())
    if long <= 0 or not np.isfinite(long):
        return 0.0
    return short / long


def _calc_atr(daily: pd.DataFrame, period: int = 14) -> float:
    h = daily["high"].tail(period + 1).reset_index(drop=True)
    l = daily["low"].tail(period + 1).reset_index(drop=True)
    c = daily["close"].tail(period + 1).reset_index(drop=True)
    prev_close = c.shift(1)
    tr = pd.concat(
        [
            h - l,
            (h - prev_close).abs(),
            (l - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = float(tr.tail(period).mean())
    latest_close = float(c.iloc[-1])
    if latest_close <= 0:
        return 0.0
    return atr / latest_close


def _monthly_span_pct(daily: pd.DataFrame, months: int = 12) -> float:
    recent = daily.set_index("date").sort_index().tail(260)
    monthly = recent.resample("ME").agg({"high": "max", "low": "min"}).tail(months)
    if monthly.empty:
        return 0.0
    low = float(monthly["low"].min())
    high = float(monthly["high"].max())
    if low <= 0:
        return 0.0
    return high / low - 1


def _piecewise(x: float, points: list[tuple[float, float]]) -> float:
    points = sorted(points)
    if x <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            ratio = (x - x0) / (x1 - x0) if x1 != x0 else 0.0
            return y0 + ratio * (y1 - y0)
    return points[-1][1]


def _position_hint(score: float, breakout_pct: float, volume_ratio: float, atr_pct: float) -> str:
    size_suffix = ""
    if atr_pct > 0:
        # Suggested position = risk_pct (2%) / (ATR% * 2)
        suggested = round(0.02 / (atr_pct * 2) * 100)
        suggested = max(5, min(100, suggested))
        size_suffix = f"，建议仓位约{suggested}%"
    if score >= 75 and breakout_pct <= 0.05 and volume_ratio >= 1.2:
        return f"强观察：可小仓试探{size_suffix}，等待回踩确认"
    if score >= 60:
        return f"观察：突破有效性待确认{size_suffix}"
    return f"弱观察：只加入自选，不追高{size_suffix}"
