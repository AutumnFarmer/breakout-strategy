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
        tags=("银行", "金融科技"),
        financial_end_date="2026-03-31",
        revenue_yoy=12.5,
        profit_yoy=18.2,
        roe=9.8,
        gross_margin=42.1,
        debt_to_assets=58.0,
        growth_score=72.4,
    )

    csv_path, xlsx_path, markdown_path, html_path = write_outputs(
        [candidate],
        {},
        tmp_path,
        1,
        0,
        "2026-05-11",
        ai_analysis="## 总体判断\n测试分析",
    )

    assert csv_path.exists()
    assert xlsx_path.exists()
    assert markdown_path.exists()
    assert html_path.exists()
    assert "平安银行" in markdown_path.read_text(encoding="utf-8")
    html = html_path.read_text(encoding="utf-8")
    assert "A股突破选股复核" in html
    assert "tagFilter" in html
    assert "金融科技" in html
    assert "成长分" in html
    assert "revenueYoy" in html
    assert "AI选股分析员" in html
    assert (tmp_path / "ai_analysis.md").exists()


def test_render_markdown_report_handles_empty_candidates() -> None:
    report = render_markdown_report([], scanned_count=10, failed_count=1, latest_trade_date="2026-05-11")

    assert "今日没有符合突破条件的候选" in report
    assert "Scanned stocks: 10" in report
    assert "Final action: NO_BUY" in report


def test_render_markdown_report_downgrades_a_to_buy_check_without_market_regime() -> None:
    candidate = Candidate(
        code="000001",
        name="平安银行",
        latest_close=12.2,
        resistance=11.8,
        zone_upper=11.8,
        breakout_pct=0.0339,
        volume_ratio=2.1,
        activity_ratio=2.1,
        activity_source="amount",
        resistance_touches=4,
        span_weeks=40,
        signal_type="A",
        score=78.5,
        buy_zone_low=12.04,
        buy_zone_high=12.51,
        trade_stop_loss=11.45,
        trade_action="可交易观察",
    )

    report = render_markdown_report(
        [candidate],
        scanned_count=1,
        failed_count=0,
        latest_trade_date="2026-05-11",
        pool_counts={"A": 1, "B": 0, "C1": 0, "C2": 0, "D": 0},
        ai_analysis="## 总体判断\n测试分析",
    )

    assert "Final action: BUY_CHECK" in report
    assert "LOW_RISK_BUY_CONFIRMED" in report
    assert "是否在买入区" in report
    assert "高开>3%不追" in report
    assert "## 10. AI 复核分析" in report


def test_render_markdown_report_includes_c1_overheat_fields() -> None:
    candidate = Candidate(
        code="000001",
        name="平安银行",
        latest_close=12.2,
        resistance=11.0,
        zone_upper=11.0,
        breakout_pct=0.109,
        volume_ratio=2.1,
        activity_ratio=2.1,
        activity_source="amount",
        signal_type="C1",
        score=78.5,
        recent_5d_pct=0.08,
        recent_10d_pct=0.16,
        consecutive_limit_up_days=2,
        long_upper_shadow=True,
    )

    report = render_markdown_report(
        [candidate],
        scanned_count=1,
        failed_count=0,
        latest_trade_date="2026-05-11",
        pool_counts={"A": 0, "B": 0, "C1": 1, "C2": 0, "D": 0},
    )

    assert "Final action: RIGHT_TAIL_WATCH" in report
    assert "近5日涨幅%" in report
    assert "近10日涨幅%" in report
    assert "|1|平安银行|000001|未标记|12.20|11.00|10.90|2.10|8.00|16.00|是|是|" in report
