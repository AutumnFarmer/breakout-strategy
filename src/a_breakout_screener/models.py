from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date


@dataclass(frozen=True)
class Candidate:
    code: str
    name: str
    latest_close: float
    resistance: float
    breakout_pct: float
    volume_ratio: float
    resistance_touches: int
    monthly_span_pct: float
    ma10: float
    ma20: float
    score: float
    first_resistance_date: date | None
    last_resistance_date: date | None
    latest_trade_date: date
    buy_zone_low: float
    buy_zone_high: float
    stop_loss: float
    position_hint: str

    def to_chinese_dict(self) -> dict[str, object]:
        data = {
            "代码": self.code,
            "名称": self.name,
            "最新收盘": round(self.latest_close, 2),
            "阻力位": round(self.resistance, 2),
            "突破幅度%": round(self.breakout_pct * 100, 2),
            "量能比": round(self.volume_ratio, 2),
            "阻力触达次数": self.resistance_touches,
            "月线跨度%": round(self.monthly_span_pct * 100, 2),
            "MA10": round(self.ma10, 2),
            "MA20": round(self.ma20, 2),
            "得分": round(self.score, 2),
            "首次阻力日期": self.first_resistance_date.isoformat() if self.first_resistance_date else "",
            "最近阻力日期": self.last_resistance_date.isoformat() if self.last_resistance_date else "",
            "最新交易日": self.latest_trade_date.isoformat(),
            "建议买入区": f"{self.buy_zone_low:.2f}-{self.buy_zone_high:.2f}",
            "止损位": round(self.stop_loss, 2),
            "仓位提示": self.position_hint,
        }
        return data

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
