from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .config import ScreenerConfig
from .models import Candidate


@dataclass(frozen=True)
class PressureZone:
    zone_low: float
    zone_mid: float
    zone_upper: float
    touches: int
    span_weeks: int
    first_touch_date: date | None
    last_touch_date: date | None
    cluster_size: int = 0

    @property
    def resistance(self) -> float:
        return self.zone_upper


ResistanceInfo = PressureZone


def evaluate_stock(
    code: str,
    name: str,
    history: pd.DataFrame,
    params: ScreenerConfig,
    is_week_confirmed: bool = False,
) -> Candidate | None:
    if len(history) < params.min_history_rows:
        return None
    daily = history.sort_values("date").reset_index(drop=True).copy()
    latest = daily.iloc[-1]
    close = float(latest["close"])
    open_ = float(latest["open"])
    if close <= 0:
        return None

    weekly = _to_weekly(daily)
    pressure_zone = calc_pressure_zone(
        weekly,
        latest["date"],
        lookback_weeks=params.resistance_lookback_weeks,
        exclude_recent_weeks=params.resistance_exclude_recent_weeks,
        pivot_k=params.pivot_k,
        cluster_tolerance=params.cluster_tolerance,
        touch_tolerance=params.touch_tolerance,
        min_touches=params.min_touches,
        min_touch_gap_weeks=params.min_touch_gap_weeks,
        min_span_weeks=params.min_span_weeks,
        effective_breakout_pct=params.effective_breakout_pct,
    )
    if pressure_zone is None or pressure_zone.zone_upper <= 0:
        return None

    breakout_pct = close / pressure_zone.zone_upper - 1

    ma10 = float(daily["close"].tail(10).mean())
    ma20 = float(daily["close"].tail(20).mean())
    ma_trend = float(daily["close"].tail(params.ma_trend_period).mean()) if params.ma_trend_period > 0 and len(daily) >= params.ma_trend_period else float("nan")
    activity_ratio, activity_source = _activity_ratio(daily)
    volume_ratio = activity_ratio
    volume_trend = _activity_trend(daily)
    monthly_span_pct = _monthly_span_pct(daily)
    weekly_span_mean = _weekly_span_mean(weekly, params.consolidation_weeks, latest["date"]) if params.consolidation_weeks > 0 else 0.0
    atr_pct = _calc_atr(daily)
    recent_low = float(daily["low"].tail(20).min())

    return assess_candidate(
        code=code,
        name=name,
        close=close,
        open_=open_,
        resistance=pressure_zone.zone_upper,
        zone_low=pressure_zone.zone_low,
        zone_mid=pressure_zone.zone_mid,
        zone_upper=pressure_zone.zone_upper,
        resistance_touches=pressure_zone.touches,
        resistance_cluster_size=pressure_zone.cluster_size,
        span_weeks=pressure_zone.span_weeks,
        breakout_pct=breakout_pct,
        ma10=ma10,
        ma20=ma20,
        ma_trend=ma_trend,
        volume_ratio=volume_ratio,
        volume_trend=volume_trend,
        activity_source=activity_source,
        monthly_span_pct=monthly_span_pct,
        weekly_span_mean=weekly_span_mean,
        atr_pct=atr_pct,
        recent_low=recent_low,
        first_resistance_date=pressure_zone.first_touch_date,
        last_resistance_date=pressure_zone.last_touch_date,
        latest_trade_date=latest["date"].date(),
        confirmation_closes=list(daily["close"].tail(params.confirmation_bars)) if params.confirmation_bars > 0 else [],
        is_week_confirmed=is_week_confirmed,
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
    zone_low: float | None = None,
    zone_mid: float | None = None,
    zone_upper: float | None = None,
    span_weeks: int = 0,
    recent_low: float = 0.0,
    is_week_confirmed: bool = False,
    activity_source: str = "",
) -> Candidate | None:
    zone_low = resistance if zone_low is None else zone_low
    zone_mid = resistance if zone_mid is None else zone_mid
    zone_upper = resistance if zone_upper is None else zone_upper
    if breakout_pct < params.breakout_buffer or breakout_pct > params.max_extension:
        return None

    if params.confirmation_bars > 0:
        if len(confirmation_closes) < params.confirmation_bars:
            return None
        if any(c < zone_upper for c in confirmation_closes):
            return None

    if params.ma_trend_period > 0:
        if not np.isfinite(ma_trend):
            return None
        if close < ma_trend:
            return None

    if params.max_open_gap_pct > 0:
        gap_pct = open_ / zone_upper - 1
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

    buy_zone_low = zone_upper * (1 + params.effective_breakout_pct)
    buy_zone_high = zone_upper * 1.06
    trade_stop_loss = zone_upper * 0.97
    structure_stop_loss = _structure_stop_loss(
        close=close,
        zone_upper=zone_upper,
        ma20=ma20,
        atr_pct=atr_pct,
        recent_low=recent_low,
    )
    signal_type, signal_reason, trade_action = _classify_signal(
        breakout_pct=breakout_pct,
        volume_ratio=volume_ratio,
        is_week_confirmed=is_week_confirmed,
        params=params,
    )
    if signal_type == "A" and not _passes_a_trade_structure(close=close, open_=open_, ma10=ma10, ma20=ma20):
        signal_type = "B"
        signal_reason = f"{signal_reason}；收盘强度或均线结构不足，降级观察"
        trade_action = "周线突破但收盘/趋势结构未满足A类买入条件，只观察"

    return Candidate(
        code=code,
        name=name,
        latest_close=close,
        resistance=zone_upper,
        breakout_pct=breakout_pct,
        volume_ratio=volume_ratio,
        volume_trend=volume_trend,
        activity_source=activity_source,
        activity_ratio=volume_ratio,
        resistance_touches=resistance_touches,
        resistance_cluster_size=resistance_cluster_size,
        signal_type=signal_type,
        signal_reason=signal_reason,
        zone_low=zone_low,
        zone_mid=zone_mid,
        zone_upper=zone_upper,
        span_weeks=span_weeks,
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
        stop_loss=trade_stop_loss,
        trade_stop_loss=trade_stop_loss,
        structure_stop_loss=structure_stop_loss,
        trade_action=trade_action,
        position_hint=_position_hint(signal_type, signal_reason),
    )


def calc_pressure_zone(
    weekly: pd.DataFrame,
    latest_trade_ts: pd.Timestamp,
    lookback_weeks: int,
    exclude_recent_weeks: int = 4,
    pivot_k: int = 3,
    cluster_tolerance: float = 0.03,
    touch_tolerance: float = 0.02,
    min_touches: int = 3,
    min_touch_gap_weeks: int = 4,
    min_span_weeks: int = 20,
    effective_breakout_pct: float = 0.02,
) -> PressureZone | None:
    if len(weekly) < max(20, pivot_k * 2 + 6):
        return None
    latest_week_end = latest_trade_ts.to_period("W-FRI").end_time.normalize()
    cutoff_week_end = latest_week_end - pd.Timedelta(weeks=max(0, exclude_recent_weeks))
    prior = weekly[weekly["week_end"] < cutoff_week_end].tail(lookback_weeks).reset_index(drop=True)
    if len(prior) < max(12, pivot_k * 2 + 6):
        return None
    prior = prior.copy()
    prior["touch_price"] = prior.apply(_effective_touch_price, axis=1)

    pivots = _pivot_highs(prior, pivot_k=pivot_k)
    if not pivots:
        return None

    clusters: list[list[tuple[float, pd.Timestamp]]] = []
    current: list[tuple[float, pd.Timestamp]] = []
    for price, week_end in sorted(pivots, key=lambda item: item[0]):
        if not current:
            current.append((price, week_end))
            continue
        current_mid = float(np.median([item[0] for item in current]))
        if price <= current_mid * (1 + cluster_tolerance):
            current.append((price, week_end))
        else:
            clusters.append(current)
            current = [(price, week_end)]
    if current:
        clusters.append(current)

    zones: list[PressureZone] = []
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        prices = [item[0] for item in cluster]
        zone_mid = float(np.median(prices))
        if not np.isfinite(zone_mid) or zone_mid <= 0:
            continue
        zone_low = zone_mid * (1 - cluster_tolerance)
        zone_upper = zone_mid * (1 + cluster_tolerance)
        touch_rows = prior[
            (prior["touch_price"] >= zone_low * (1 - touch_tolerance))
            & (prior["touch_price"] <= zone_upper * (1 + touch_tolerance))
            & (prior["close"] <= zone_upper * (1 + effective_breakout_pct))
        ]
        independent = _independent_touches(touch_rows, min_gap_weeks=min_touch_gap_weeks)
        if len(independent) < min_touches:
            continue
        first_touch = pd.Timestamp(independent[0]["week_end"])
        last_touch = pd.Timestamp(independent[-1]["week_end"])
        span_weeks = max(0, int((last_touch - first_touch).days // 7))
        if span_weeks < min_span_weeks:
            continue
        after_first = prior[prior["week_end"] > first_touch]
        closes_above = after_first[after_first["close"] > zone_upper * (1 + effective_breakout_pct)]
        if len(closes_above) >= 2:
            continue
        zones.append(
            PressureZone(
                zone_low=zone_low,
                zone_mid=zone_mid,
                zone_upper=zone_upper,
                touches=len(independent),
                span_weeks=span_weeks,
                first_touch_date=first_touch.date(),
                last_touch_date=last_touch.date(),
                cluster_size=len(cluster),
            )
        )

    if not zones:
        return None
    return max(zones, key=lambda item: (item.touches, item.span_weeks, item.cluster_size, item.zone_mid))


def calc_resistance(
    weekly: pd.DataFrame,
    latest_trade_ts: pd.Timestamp,
    lookback_weeks: int,
    touch_tolerance: float = 0.02,
    cluster_tolerance: float = 0.03,
    top_k: int = 8,
) -> ResistanceInfo | None:
    pivot_k = max(1, min(3, top_k // 2))
    return calc_pressure_zone(
        weekly=weekly,
        latest_trade_ts=latest_trade_ts,
        lookback_weeks=lookback_weeks,
        exclude_recent_weeks=4,
        pivot_k=pivot_k,
        cluster_tolerance=cluster_tolerance,
        touch_tolerance=touch_tolerance,
        min_touches=3,
        min_touch_gap_weeks=4,
        min_span_weeks=20,
        effective_breakout_pct=0.02,
    )


def _pivot_highs(weekly: pd.DataFrame, pivot_k: int) -> list[tuple[float, pd.Timestamp]]:
    highs = pd.to_numeric(weekly["high"], errors="coerce").to_numpy()
    pivots: list[tuple[float, pd.Timestamp]] = []
    for idx in range(pivot_k, len(weekly) - pivot_k):
        high = float(highs[idx])
        if not np.isfinite(high) or high <= 0:
            continue
        left = highs[idx - pivot_k : idx]
        right = highs[idx + 1 : idx + pivot_k + 1]
        if high <= float(np.nanmax(left)) or high <= float(np.nanmax(right)):
            continue
        row = weekly.iloc[idx]
        pivots.append((_adjusted_pivot_price(row), pd.Timestamp(row["week_end"])))
    return pivots


def _adjusted_pivot_price(row: pd.Series) -> float:
    return _effective_touch_price(row)


def _effective_touch_price(row: pd.Series) -> float:
    high = float(row["high"])
    low = float(row["low"])
    open_ = float(row["open"])
    close = float(row["close"])
    top_body = max(open_, close)
    week_range = high - low
    upper_shadow = high - top_body
    if week_range > 0 and upper_shadow / week_range >= 0.45:
        return top_body + upper_shadow * 0.5
    return high


def _independent_touches(touch_rows: pd.DataFrame, min_gap_weeks: int) -> list[pd.Series]:
    touches: list[pd.Series] = []
    last_week_end: pd.Timestamp | None = None
    for _, row in touch_rows.sort_values("week_end").iterrows():
        week_end = pd.Timestamp(row["week_end"])
        if last_week_end is not None and (week_end - last_week_end).days < min_gap_weeks * 7:
            continue
        touches.append(row)
        last_week_end = week_end
    return touches


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
            (params.effective_breakout_pct, 30.0),
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
            (params.strong_volume_ratio, 20.0),
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


def _activity_series(daily: pd.DataFrame) -> tuple[pd.Series, str]:
    amount = pd.to_numeric(daily.get("amount", pd.Series(index=daily.index)), errors="coerce").fillna(0)
    volume = pd.to_numeric(daily.get("volume", pd.Series(index=daily.index)), errors="coerce").fillna(0)
    if len(amount) >= 20 and float(amount.tail(20).mean()) > 0:
        return amount, "amount"
    return volume, "volume"


def _activity_ratio(daily: pd.DataFrame) -> tuple[float, str]:
    activity, source = _activity_series(daily)
    latest_activity = float(activity.iloc[-1])
    baseline = float(activity.tail(20).iloc[:-1].mean())
    if baseline <= 0 or not np.isfinite(baseline):
        return 0.0, source
    return latest_activity / baseline, source


def _activity_trend(daily: pd.DataFrame) -> float:
    activity, _ = _activity_series(daily)
    tail = activity.tail(20)
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


def _structure_stop_loss(
    close: float,
    zone_upper: float,
    ma20: float,
    atr_pct: float,
    recent_low: float,
) -> float:
    candidates = [zone_upper * 0.97]
    if ma20 > 0 and np.isfinite(ma20):
        candidates.append(ma20 * 0.98)
    if recent_low > 0 and np.isfinite(recent_low):
        candidates.append(recent_low * 0.98)
    if atr_pct > 0 and np.isfinite(atr_pct):
        candidates.append(close * (1 - atr_pct * 2))
    valid = [item for item in candidates if item > 0 and item < close]
    return max(valid) if valid else zone_upper * 0.97


def _passes_a_trade_structure(close: float, open_: float, ma10: float, ma20: float) -> bool:
    if close < open_:
        return False
    if ma10 > 0 and np.isfinite(ma10) and close < ma10:
        return False
    if ma10 > 0 and ma20 > 0 and np.isfinite(ma10) and np.isfinite(ma20) and ma10 < ma20:
        return False
    return True


def _classify_signal(
    breakout_pct: float,
    volume_ratio: float,
    is_week_confirmed: bool,
    params: ScreenerConfig,
) -> tuple[str, str, str]:
    if breakout_pct < params.effective_breakout_pct:
        return "D", "突破不足2%，只观察压力区附近反应", "突破不足，不进入交易池"
    if breakout_pct <= 0.06:
        if is_week_confirmed and volume_ratio >= params.strong_volume_ratio:
            return "A", "周线确认且成交额放大", "可交易观察"
        return "B", "日线预警，周线或量能仍需确认", "只预警观察，等待周线确认"
    if breakout_pct <= params.max_buy_extension:
        return "B", "突破偏高但仍在谨慎观察区", "谨慎观察，不主动追高"
    if breakout_pct <= params.max_extension:
        if volume_ratio >= params.strong_volume_ratio:
            return "C1", "强趋势突破，已远离低风险买点，但可进入右尾观察", "强趋势观察，小仓或等回踩"
        return "C2", "距离压力区过远且量能确认不足", "不追"
    return "D", "突破过远，风险收益比失衡", "排除"


def _piecewise(x: float, points: list[tuple[float, float]]) -> float:
    points = sorted(points)
    if x <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            ratio = (x - x0) / (x1 - x0) if x1 != x0 else 0.0
            return y0 + ratio * (y1 - y0)
    return points[-1][1]


def _position_hint(signal_type: str, signal_reason: str) -> str:
    if signal_type == "A":
        return f"A类周线确认：{signal_reason}；单票初始仓位按总资产0.5%-1%控制"
    if signal_type == "B":
        return f"B类日线预警：{signal_reason}；只观察或小仓试错"
    if signal_type == "C1":
        return f"C1类强趋势观察：{signal_reason}；不属于低风险买点，只适合右尾长持池"
    if signal_type in {"C2", "C"}:
        return f"C2类不追：{signal_reason}；等待回踩或重新整理"
    return f"D类排除：{signal_reason}"
