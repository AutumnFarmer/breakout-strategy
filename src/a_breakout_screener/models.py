from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import date


@dataclass(frozen=True)
class Candidate:
    code: str
    name: str
    latest_close: float
    resistance: float
    breakout_pct: float
    volume_ratio: float
    volume_trend: float = 0.0
    resistance_touches: int = 0
    resistance_cluster_size: int = 0
    signal_type: str = "D"
    signal_reason: str = ""
    zone_low: float = 0.0
    zone_mid: float = 0.0
    zone_upper: float = 0.0
    span_weeks: int = 0
    monthly_span_pct: float = 0.0
    ma10: float = 0.0
    ma20: float = 0.0
    atr_pct: float = 0.0
    score: float = 0.0
    first_resistance_date: date | None = None
    last_resistance_date: date | None = None
    latest_trade_date: date = field(default_factory=date.today)
    buy_zone_low: float = 0.0
    buy_zone_high: float = 0.0
    stop_loss: float = 0.0
    trade_stop_loss: float = 0.0
    structure_stop_loss: float = 0.0
    circ_mv: float = 0.0
    total_asset_position: str = "0.5%-1%"
    strategy_position: str = "10%-20%"
    max_loss_asset_pct: str = "0.2%-0.3%"
    trade_action: str = ""
    position_hint: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
    financial_end_date: str = ""
    revenue_yoy: float | None = None
    profit_yoy: float | None = None
    roe: float | None = None
    gross_margin: float | None = None
    debt_to_assets: float | None = None
    growth_score: float = 0.0

    def to_chinese_dict(self) -> dict[str, object]:
        data = {
            "代码": self.code,
            "名称": self.name,
            "信号类型": self.signal_type,
            "信号说明": self.signal_reason,
            "最新收盘": round(self.latest_close, 2),
            "压力区下沿": round(self.zone_low or self.resistance, 2),
            "压力区中枢": round(self.zone_mid or self.resistance, 2),
            "压力区上沿": round(self.zone_upper or self.resistance, 2),
            "阻力位": round(self.resistance, 2),
            "突破幅度%": round(self.breakout_pct * 100, 2),
            "量能比": round(self.volume_ratio, 2),
            "量能趋势": round(self.volume_trend, 2),
            "阻力触达次数": self.resistance_touches,
            "阻力聚类大小": self.resistance_cluster_size,
            "压力跨度周": self.span_weeks,
            "月线跨度%": round(self.monthly_span_pct * 100, 2),
            "ATR%": round(self.atr_pct * 100, 2),
            "MA10": round(self.ma10, 2),
            "MA20": round(self.ma20, 2),
            "得分": round(self.score, 2),
            "首次阻力日期": self.first_resistance_date.isoformat() if self.first_resistance_date else "",
            "最近阻力日期": self.last_resistance_date.isoformat() if self.last_resistance_date else "",
            "最新交易日": self.latest_trade_date.isoformat(),
            "建议买入区": f"{self.buy_zone_low:.2f}-{self.buy_zone_high:.2f}",
            "止损位": round(self.stop_loss, 2),
            "交易止损": round(self.trade_stop_loss or self.stop_loss, 2),
            "结构止损": round(self.structure_stop_loss or self.stop_loss, 2),
            "流通市值(亿)": round(self.circ_mv, 2) if self.circ_mv > 0 else "",
            "总资产建议仓位": self.total_asset_position,
            "策略内建议仓位": self.strategy_position,
            "最大允许亏损": self.max_loss_asset_pct,
            "交易结论": self.trade_action,
            "仓位提示": self.position_hint,
            "题材标签": " / ".join(self.tags),
            "财务期": self.financial_end_date,
            "成长分": round(self.growth_score, 2) if self.growth_score > 0 else "",
            "营收同比%": _round_optional(self.revenue_yoy),
            "净利同比%": _round_optional(self.profit_yoy),
            "ROE%": _round_optional(self.roe),
            "毛利率%": _round_optional(self.gross_margin),
            "资产负债率%": _round_optional(self.debt_to_assets),
        }
        return data

    def with_circ_mv(self, circ_mv: float) -> Candidate:
        return replace(self, circ_mv=circ_mv)

    def with_tags(self, tags: tuple[str, ...] | list[str]) -> Candidate:
        return replace(self, tags=tuple(tags))

    def with_financial_metrics(self, metrics: dict[str, object]) -> Candidate:
        return replace(
            self,
            financial_end_date=str(metrics.get("financial_end_date") or ""),
            revenue_yoy=_optional_float(metrics.get("revenue_yoy")),
            profit_yoy=_optional_float(metrics.get("profit_yoy")),
            roe=_optional_float(metrics.get("roe")),
            gross_margin=_optional_float(metrics.get("gross_margin")),
            debt_to_assets=_optional_float(metrics.get("debt_to_assets")),
            growth_score=float(metrics.get("growth_score") or 0.0),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round_optional(value: float | None) -> float | str:
    return round(value, 2) if value is not None else ""
