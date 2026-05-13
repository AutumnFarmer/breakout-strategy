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
    position_hint: str = ""

    def to_chinese_dict(self) -> dict[str, object]:
        data = {
            "代码": self.code,
            "名称": self.name,
            "最新收盘": round(self.latest_close, 2),
            "信号类型": self.signal_type,
            "信号说明": self.signal_reason,
            "阻力位": round(self.resistance, 2),
            "突破幅度%": round(self.breakout_pct * 100, 2),
            "量能比": round(self.volume_ratio, 2),
            "量能趋势": round(self.volume_trend, 2),
            "阻力触达次数": self.resistance_touches,
            "阻力聚类大小": self.resistance_cluster_size,
            "月线跨度%": round(self.monthly_span_pct * 100, 2),
            "ATR%": round(self.atr_pct * 100, 2),
            "MA10": round(self.ma10, 2),
            "MA20": round(self.ma20, 2),
            "得分": round(self.score, 2),
            "首次阻力日期": self.first_resistance_date.isoformat() if self.first_resistance_date else "",
            "最近阻力日期": self.last_resistance_date.isoformat() if self.last_resistance_date else "",
            "最新交易日": self.latest_trade_date.isoformat(),
            "建议买入区": f"{self.buy_zone_low:.2f}-{self.buy_zone_high:.2f}",
            "交易止损": round(self.trade_stop_loss or self.stop_loss, 2),
            "结构止损": round(self.structure_stop_loss or self.stop_loss, 2),
            "流通市值(亿)": round(self.circ_mv, 2) if self.circ_mv > 0 else "",
            "总资产建议仓位": self.total_asset_position,
            "策略内建议仓位": self.strategy_position,
            "最大允许亏损": self.max_loss_asset_pct,
            "仓位提示": self.position_hint,
        }
        return data

    def with_circ_mv(self, circ_mv: float) -> Candidate:
        return replace(self, circ_mv=circ_mv)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
