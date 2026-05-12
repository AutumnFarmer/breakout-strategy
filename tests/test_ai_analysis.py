from __future__ import annotations

from datetime import date

from a_breakout_screener.ai_analysis import _extract_text, generate_ai_analysis
from a_breakout_screener.config import AIAnalysisConfig
from a_breakout_screener.models import Candidate


def test_generate_ai_analysis_returns_empty_when_disabled() -> None:
    candidate = Candidate(
        code="300750",
        name="宁德时代",
        latest_close=430,
        resistance=420,
        breakout_pct=0.02,
        volume_ratio=1.5,
        latest_trade_date=date(2026, 5, 12),
    )

    assert generate_ai_analysis(AIAnalysisConfig(enabled=False), [candidate], 1, 0, "2026-05-12") == ""


def test_extract_text_supports_sub2api_message_shape() -> None:
    payload = {
        "content": [
            {"type": "text", "text": "第一段"},
            {"type": "text", "text": "第二段"},
        ]
    }

    assert _extract_text(payload) == "第一段\n第二段"
