from __future__ import annotations

from datetime import date

from a_breakout_screener.models import Candidate
from a_breakout_screener.screener import render_markdown_report, write_outputs


def test_write_outputs_creates_report_files(tmp_path) -> None:
    candidate = Candidate(
        code="000001",
        name="平安银行",
        latest_close=12.2,
        resistance=11.8,
        breakout_pct=0.0339,
        volume_ratio=2.1,
        resistance_touches=4,
        monthly_span_pct=0.52,
        ma10=11.7,
        ma20=11.3,
        score=78.5,
        first_resistance_date=date(2025, 6, 6),
        last_resistance_date=date(2025, 11, 7),
        latest_trade_date=date(2026, 5, 11),
        buy_zone_low=11.74,
        buy_zone_high=12.09,
        stop_loss=11.0,
        position_hint="强观察：可小仓试探，等待回踩确认",
    )

    csv_path, xlsx_path, markdown_path, html_path = write_outputs(
        [candidate],
        {},
        tmp_path,
        1,
        0,
        "2026-05-11",
    )

    assert csv_path.exists()
    assert xlsx_path.exists()
    assert markdown_path.exists()
    assert html_path.exists()
    assert "平安银行" in markdown_path.read_text(encoding="utf-8")
    assert "A股突破选股复核" in html_path.read_text(encoding="utf-8")


def test_render_markdown_report_handles_empty_candidates() -> None:
    report = render_markdown_report([], scanned_count=10, failed_count=1, latest_trade_date="2026-05-11")

    assert "今日没有符合突破条件的候选" in report
    assert "扫描股票数: 10" in report
