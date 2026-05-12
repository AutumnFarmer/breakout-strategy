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


def evaluate_stock(code: str, name: str, history: pd.DataFrame, params: ScreenerConfig) -> Candidate | None:
    if len(history) < params.min_history_rows:
        return None
    daily = history.sort_values("date").reset_index(drop=True).copy()
    latest = daily.iloc[-1]
    close = float(latest["close"])
    if close <= 0:
        return None

    weekly = _to_weekly(daily)
    resistance = calc_resistance(weekly, latest["date"], params.resistance_lookback_weeks)
    if resistance is None or resistance.resistance <= 0:
        return None

    breakout_pct = close / resistance.resistance - 1
    if breakout_pct < params.breakout_buffer or breakout_pct > params.max_extension:
        return None

    ma10 = float(daily["close"].tail(10).mean())
    ma20 = float(daily["close"].tail(20).mean())
    volume_ratio = _volume_ratio(daily)
    monthly_span_pct = _monthly_span_pct(daily)
    score = calc_score(
        close=close,
        breakout_pct=breakout_pct,
        ma10=ma10,
        ma20=ma20,
        volume_ratio=volume_ratio,
        resistance_touches=resistance.touches,
        monthly_span_pct=monthly_span_pct,
        params=params,
    )
    if score <= 0:
        return None

    buy_zone_low = resistance.resistance * 0.995
    buy_zone_high = min(resistance.resistance * 1.025, close * 1.01)
    stop_loss = min(resistance.resistance * 0.96, ma20 * 0.98)

    return Candidate(
        code=code,
        name=name,
        latest_close=close,
        resistance=resistance.resistance,
        breakout_pct=breakout_pct,
        volume_ratio=volume_ratio,
        resistance_touches=resistance.touches,
        monthly_span_pct=monthly_span_pct,
        ma10=ma10,
        ma20=ma20,
        score=score,
        first_resistance_date=resistance.first_touch_date,
        last_resistance_date=resistance.last_touch_date,
        latest_trade_date=latest["date"].date(),
        buy_zone_low=buy_zone_low,
        buy_zone_high=buy_zone_high,
        stop_loss=stop_loss,
        position_hint=_position_hint(score, breakout_pct, volume_ratio),
    )


def calc_resistance(
    weekly: pd.DataFrame,
    latest_trade_ts: pd.Timestamp,
    lookback_weeks: int,
    touch_tolerance: float = 0.02,
) -> ResistanceInfo | None:
    if len(weekly) < 20:
        return None
    latest_week_end = latest_trade_ts.to_period("W-FRI").end_time.normalize()
    prior = weekly[weekly["week_end"] < latest_week_end].tail(lookback_weeks)
    if len(prior) < 12:
        return None
    resistance = float(prior["high"].max())
    if not np.isfinite(resistance) or resistance <= 0:
        return None
    near = prior[
        (prior["high"] >= resistance * (1 - touch_tolerance))
        | (prior["close"] >= resistance * (1 - touch_tolerance))
    ]
    if near.empty:
        return ResistanceInfo(resistance=resistance, touches=0, first_touch_date=None, last_touch_date=None)
    return ResistanceInfo(
        resistance=resistance,
        touches=int(len(near)),
        first_touch_date=near.iloc[0]["week_end"].date(),
        last_touch_date=near.iloc[-1]["week_end"].date(),
    )


def calc_score(
    close: float,
    breakout_pct: float,
    ma10: float,
    ma20: float,
    volume_ratio: float,
    resistance_touches: int,
    monthly_span_pct: float,
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
        trend_score = 20.0
    elif close >= ma20:
        trend_score = 12.0
    elif close >= ma10:
        trend_score = 8.0

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
    resistance_score = min(20.0, max(0.0, resistance_touches - 1) * 4.0)
    span_penalty = 0.0
    if monthly_span_pct > 1.8:
        span_penalty = 8.0
    elif monthly_span_pct > 1.2:
        span_penalty = 4.0

    return max(0.0, price_score + trend_score + volume_score + resistance_score - span_penalty)


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


def _volume_ratio(daily: pd.DataFrame) -> float:
    latest_volume = float(daily.iloc[-1].get("volume", 0) or 0)
    baseline = float(daily["volume"].tail(20).iloc[:-1].mean())
    if baseline <= 0 or not np.isfinite(baseline):
        return 0.0
    return latest_volume / baseline


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


def _position_hint(score: float, breakout_pct: float, volume_ratio: float) -> str:
    if score >= 75 and breakout_pct <= 0.05 and volume_ratio >= 1.2:
        return "强观察：可小仓试探，等待回踩确认"
    if score >= 60:
        return "观察：突破有效性待确认"
    return "弱观察：只加入自选，不追高"
