from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from .models import Candidate


def write_html_dashboard(
    candidates: list[Candidate],
    history_by_code: dict[str, pd.DataFrame],
    output_dir: Path,
    scanned_count: int,
    failed_count: int,
    latest_trade_date: str,
    ai_analysis: str = "",
    pool_counts: dict[str, int] | None = None,
    full_scan: bool = True,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "breakout_dashboard.html"
    payload = {
        "meta": {
            "latestTradeDate": latest_trade_date,
            "scannedCount": scanned_count,
            "failedCount": failed_count,
            "candidateCount": len(candidates),
            "allCandidateCount": sum((pool_counts or {}).values()) if pool_counts else len(candidates),
            "poolCounts": pool_counts or {},
            "dataVerdict": _data_verdict(scanned_count, failed_count),
            "finalAction": _final_action(scanned_count, failed_count, pool_counts or {}),
            "marketRegime": "NOT_EVALUATED",
            "runTime": datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S"),
            "themeConcentration": _theme_concentration(candidates),
            "fullScan": full_scan,
        },
        "candidates": [_candidate_payload(idx, item) for idx, item in enumerate(candidates, start=1)],
        "history": {
            code: _history_payload(history)
            for code, history in history_by_code.items()
        },
        "aiAnalysis": ai_analysis.strip(),
    }
    path.write_text(
        HTML_TEMPLATE.replace("__DASHBOARD_DATA__", json.dumps(payload, ensure_ascii=False)),
        encoding="utf-8",
    )
    return path


def _candidate_payload(rank: int, item: Candidate) -> dict[str, Any]:
    return {
        "rank": rank,
        "code": item.code,
        "name": item.name,
        "signalType": item.signal_type,
        "signalReason": item.signal_reason,
        "primaryTag": _primary_tag(item),
        "latestClose": round(item.latest_close, 4),
        "resistance": round(item.resistance, 4),
        "zoneLow": round(item.zone_low or item.resistance, 4),
        "zoneMid": round(item.zone_mid or item.resistance, 4),
        "zoneUpper": round(item.zone_upper or item.resistance, 4),
        "spanWeeks": item.span_weeks,
        "breakoutPct": round(item.breakout_pct * 100, 4),
        "volumeRatio": round(item.volume_ratio, 4),
        "volumeTrend": round(item.volume_trend, 4),
        "activitySource": item.activity_source,
        "activityRatio": round(item.activity_ratio or item.volume_ratio, 4),
        "touches": item.resistance_touches,
        "clusterSize": item.resistance_cluster_size,
        "atrPct": round(item.atr_pct * 100, 4),
        "recent5dPct": round(item.recent_5d_pct * 100, 4),
        "recent10dPct": round(item.recent_10d_pct * 100, 4),
        "consecutiveLimitUpDays": item.consecutive_limit_up_days,
        "longUpperShadow": item.long_upper_shadow,
        "circMv": round(item.circ_mv, 2) if item.circ_mv > 0 else 0,
        "score": round(item.score, 4),
        "growthScore": round(item.growth_score, 4) if item.growth_score > 0 else 0,
        "financialEndDate": item.financial_end_date,
        "revenueYoy": _rounded(item.revenue_yoy, 4),
        "profitYoy": _rounded(item.profit_yoy, 4),
        "roe": _rounded(item.roe, 4),
        "grossMargin": _rounded(item.gross_margin, 4),
        "debtToAssets": _rounded(item.debt_to_assets, 4),
        "buyLow": round(item.buy_zone_low, 4),
        "buyHigh": round(item.buy_zone_high, 4),
        "buyZoneStatus": _buy_zone_status(item),
        "nextDayTradeAction": _next_day_trade_action(item),
        "stopLoss": round(item.stop_loss, 4),
        "tradeStopLoss": round(item.trade_stop_loss or item.stop_loss, 4),
        "structureStopLoss": round(item.structure_stop_loss or item.stop_loss, 4),
        "tradeAction": item.trade_action,
        "hint": item.position_hint,
        "tags": list(item.tags),
        "latestTradeDate": item.latest_trade_date.isoformat(),
    }


def _history_payload(history: pd.DataFrame) -> list[dict[str, Any]]:
    if history.empty:
        return []
    df = history.sort_values("date").tail(900).copy()
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        records.append(
            {
                "date": pd.Timestamp(row["date"]).date().isoformat(),
                "open": _rounded(row.get("open")),
                "high": _rounded(row.get("high")),
                "low": _rounded(row.get("low")),
                "close": _rounded(row.get("close")),
                "volume": _rounded(row.get("volume"), 0),
            }
        )
    return records


def _rounded(value: Any, digits: int = 4) -> float | None:
    if pd.isna(value):
        return None
    return round(float(value), digits)


def _data_verdict(scanned_count: int, failed_count: int) -> str:
    if scanned_count <= 0:
        return "DATA_FAILED"
    if failed_count >= scanned_count or failed_count > max(50, int(scanned_count * 0.2)):
        return "DATA_FAILED"
    if failed_count > 0:
        return "WARN"
    return "OK"


def _final_action(scanned_count: int, failed_count: int, pool_counts: dict[str, int]) -> str:
    if _data_verdict(scanned_count, failed_count) == "DATA_FAILED":
        return "DATA_FAILED"
    if pool_counts.get("A", 0) > 0:
        return "BUY_CHECK"
    if pool_counts.get("B", 0) > 0:
        return "WATCH_ONLY"
    if pool_counts.get("C1", 0) > 0:
        return "RIGHT_TAIL_WATCH"
    return "NO_BUY"


def _primary_tag(item: Candidate) -> str:
    return item.tags[0] if item.tags else "未标记"


def _theme_concentration(candidates: list[Candidate]) -> str:
    themes = [_primary_tag(item) for item in candidates if item.signal_type in {"A", "B", "C1"}]
    if not themes:
        return "低"
    top_count = Counter(themes).most_common(1)[0][1]
    ratio = top_count / len(themes)
    if ratio >= 0.35:
        return "高"
    if ratio >= 0.2:
        return "中"
    return "低"


def _buy_zone_status(item: Candidate) -> str:
    if item.buy_zone_low <= 0 or item.buy_zone_high <= 0:
        return "UNKNOWN"
    if item.buy_zone_low <= item.latest_close <= item.buy_zone_high:
        return "YES"
    if item.latest_close > item.buy_zone_high:
        return "ABOVE"
    return "BELOW"


def _next_day_trade_action(item: Candidate) -> str:
    status = _buy_zone_status(item)
    if status == "YES":
        return "低风险复核；次日不高开才考虑"
    if status == "ABOVE":
        return "等回踩，不追"
    if status == "BELOW":
        return "等重新站回买入区"
    return "缺少买入区数据，人工复核"


HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>A股选股看板</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --text: #18202b;
      --muted: #667085;
      --line: #d9dee8;
      --panel: #ffffff;
      --accent: #1f6feb;
      --accent-soft: #e9f1ff;
      --up: #d33f49;
      --down: #16835f;
      --warn: #b76e00;
      --shadow: 0 10px 24px rgba(24, 32, 43, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    header {
      padding: 22px 28px 14px;
      border-bottom: 1px solid var(--line);
      background: #fff;
    }
    h1 {
      margin: 0 0 12px;
      font-size: 24px;
      line-height: 1.25;
      letter-spacing: 0;
    }
    .stats {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    .stat {
      min-width: 120px;
      padding: 9px 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfe;
    }
    .stat span {
      display: block;
      color: var(--muted);
      font-size: 12px;
    }
    .stat strong {
      display: block;
      margin-top: 3px;
      font-size: 18px;
      font-weight: 700;
    }
    .review-shell {
      padding: 16px;
      display: grid;
      grid-template-columns: 1fr;
      gap: 14px;
    }
    .review-section {
      padding: 16px;
      box-shadow: none;
    }
    .review-section h2 {
      margin: 0 0 10px;
      font-size: 17px;
      line-height: 1.35;
      letter-spacing: 0;
    }
    .review-section h3 {
      margin: 16px 0 8px;
      font-size: 14px;
      line-height: 1.35;
      letter-spacing: 0;
    }
    .review-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
    }
    .review-kv {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px 10px;
      background: #fbfcfe;
      min-width: 0;
    }
    .review-kv span {
      display: block;
      color: var(--muted);
      font-size: 12px;
    }
    .review-kv strong {
      display: block;
      margin-top: 4px;
      font-size: 14px;
      overflow-wrap: anywhere;
    }
    .verdict-box {
      border: 1px solid #d7e3f8;
      border-radius: 8px;
      padding: 12px;
      background: #f7fbff;
    }
    .verdict-box strong {
      display: block;
      margin-bottom: 6px;
      color: var(--accent);
      font-size: 16px;
    }
    .review-note {
      margin: 8px 0 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.6;
    }
    .review-table-wrap {
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .review-table-wrap table {
      min-width: 760px;
    }
    .review-table-wrap th {
      position: static;
    }
    .review-table-wrap td,
    .review-table-wrap th {
      padding: 8px 9px;
    }
    .review-table-wrap .left {
      text-align: left;
    }
    .link-button {
      border: 0;
      padding: 0;
      background: transparent;
      color: var(--accent);
      font: inherit;
      font-weight: 700;
      cursor: pointer;
      text-align: left;
    }
    .link-button:hover {
      text-decoration: underline;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      padding: 2px 7px;
      border-radius: 999px;
      border: 1px solid #d7e3f8;
      background: #f2f7ff;
      color: #285b9f;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }
    .pill.warn {
      border-color: #f1d19b;
      background: #fff7e8;
      color: var(--warn);
    }
    .pool-block {
      margin-top: 12px;
      scroll-margin-top: 16px;
    }
    .pool-block.flash {
      outline: 2px solid var(--accent);
      outline-offset: 2px;
      transition: outline-color .2s ease;
    }
    .theme-detail {
      margin-top: 12px;
      padding: 12px;
      border: 1px solid #d7e3f8;
      border-radius: 8px;
      background: #f7fbff;
      scroll-margin-top: 16px;
    }
    .theme-detail[hidden] {
      display: none;
    }
    .theme-detail-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 8px;
    }
    .theme-detail-head h3 {
      margin: 0;
    }
    .stock-search-bar {
      display: flex;
      gap: 8px;
      align-items: center;
      max-width: 560px;
    }
    .stock-search-bar input {
      flex: 1;
      min-width: 180px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 7px 9px;
      font: inherit;
      font-size: 13px;
    }
    .stock-search-results {
      margin-top: 10px;
    }
    .action-buttons {
      display: inline-flex;
      gap: 6px;
      align-items: center;
      white-space: nowrap;
    }
    .action-button {
      padding: 4px 8px;
      border-radius: 6px;
      font-size: 12px;
      line-height: 1.25;
    }
    .action-button.primary {
      border-color: var(--accent);
      background: var(--accent-soft);
      color: var(--accent);
      font-weight: 700;
    }
    .action-button.buy {
      border-color: #f1d19b;
      background: #fff7e8;
      color: var(--warn);
      font-weight: 700;
    }
    .zone-badge {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 76px;
      padding: 3px 8px;
      border-radius: 999px;
      border: 1px solid #d7e3f8;
      background: #f2f7ff;
      color: #285b9f;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }
    .zone-badge.yes {
      border-color: #f1d19b;
      background: #fff7e8;
      color: var(--warn);
    }
    .zone-badge.above {
      border-color: #ffd6dc;
      background: #fff3f5;
      color: var(--up);
    }
    .zone-badge.below,
    .zone-badge.unknown {
      border-color: #d6eadf;
      background: #f2fbf6;
      color: var(--down);
    }
    .stock-detail {
      min-width: 170px;
    }
    .stock-detail summary {
      color: var(--accent);
      font-weight: 700;
      cursor: pointer;
    }
    .stock-detail dl {
      display: grid;
      grid-template-columns: auto minmax(70px, 1fr);
      gap: 4px 10px;
      min-width: 230px;
      margin: 8px 0 0;
      padding: 10px;
      border: 1px solid #edf0f5;
      border-radius: 8px;
      background: #fbfcfe;
    }
    .stock-detail dt {
      color: var(--muted);
      font-size: 12px;
      text-align: left;
      white-space: nowrap;
    }
    .stock-detail dd {
      margin: 0;
      color: var(--text);
      font-size: 12px;
      text-align: right;
      white-space: nowrap;
    }
    .technical-detail {
      margin-top: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      background: #fbfcfe;
    }
    .technical-detail summary {
      color: var(--muted);
      cursor: pointer;
      font-weight: 700;
    }
    .position-positive {
      color: var(--up);
      font-weight: 700;
    }
    .position-negative {
      color: var(--down);
      font-weight: 700;
    }
    .chart-main {
      display: block;
      min-height: auto;
    }
    .chart-main[hidden] {
      display: none;
    }
    .chart-main .chart-panel {
      min-height: 640px;
    }
    .chart-dialog {
      width: min(1180px, calc(100vw - 24px));
      max-height: calc(100vh - 24px);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0;
      background: var(--panel);
      box-shadow: 0 24px 70px rgba(24, 32, 43, 0.24);
      color: var(--text);
      overflow: hidden;
    }
    .chart-dialog::backdrop {
      background: rgba(24, 32, 43, 0.48);
    }
    .chart-dialog .chart-main {
      display: block;
      padding: 0;
      min-height: auto;
    }
    .chart-dialog .chart-panel {
      max-height: calc(100vh - 24px);
      min-height: min(760px, calc(100vh - 24px));
      border: 0;
      border-radius: 8px;
      box-shadow: none;
      overflow: auto;
    }
    .chart-close-button {
      border-color: #ffd6dc;
      background: #fff3f5;
      color: var(--up);
      font-weight: 700;
    }
    .position-dialog {
      width: min(420px, calc(100vw - 32px));
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0;
      box-shadow: var(--shadow);
      color: var(--text);
    }
    .position-dialog::backdrop {
      background: rgba(24, 32, 43, 0.38);
    }
    .stock-ai-dialog {
      width: min(760px, calc(100vw - 28px));
      max-height: calc(100vh - 28px);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 0;
      box-shadow: var(--shadow);
      color: var(--text);
    }
    .stock-ai-dialog::backdrop {
      background: rgba(24, 32, 43, 0.42);
    }
    .stock-ai-body {
      max-height: min(620px, calc(100vh - 150px));
      overflow: auto;
      border: 1px solid #edf0f5;
      border-radius: 8px;
      padding: 10px 12px;
      background: #fbfcfe;
    }
    .position-form {
      padding: 16px;
    }
    .position-form h2 {
      margin: 0 0 12px;
      font-size: 17px;
      line-height: 1.35;
    }
    .field {
      margin-top: 10px;
    }
    .field label {
      display: block;
      margin-bottom: 5px;
      color: var(--muted);
      font-size: 12px;
    }
    .field input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 9px;
      font: inherit;
      font-size: 14px;
    }
    .dialog-actions {
      display: flex;
      justify-content: flex-end;
      gap: 8px;
      margin-top: 14px;
    }
    .ai-report-content {
      color: #2c3440;
      font-size: 13px;
      line-height: 1.65;
    }
    .ai-report-content h3 {
      margin: 12px 0 6px;
    }
    .ai-report-content p {
      margin: 6px 0;
    }
    .ai-report-content ul {
      margin: 6px 0 8px 18px;
      padding: 0;
    }
    .ai-report-content table {
      width: 100%;
      border-collapse: collapse;
      margin: 8px 0 12px;
      font-size: 12px;
    }
    .ai-report-content th,
    .ai-report-content td {
      padding: 7px 8px;
      border: 1px solid #edf0f5;
      text-align: left;
      white-space: normal;
    }
    .ai-report-content th {
      background: #f9fafc;
      color: var(--muted);
      font-weight: 700;
    }
    .ai-report-content code {
      padding: 1px 4px;
      border-radius: 4px;
      background: #eef2f7;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }
    main {
      display: grid;
      grid-template-columns: minmax(390px, 44%) minmax(520px, 1fr);
      gap: 16px;
      padding: 16px;
      min-height: calc(100vh - 112px);
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      min-width: 0;
    }
    .table-panel {
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }
    .section-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
    }
    .section-head h2 {
      margin: 0;
      font-size: 16px;
      line-height: 1.3;
      letter-spacing: 0;
    }
    .filter-tools {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
      min-width: 220px;
    }
    select {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
      border-radius: 6px;
      padding: 6px 28px 6px 9px;
      font: inherit;
      font-size: 12px;
      max-width: 220px;
    }
    .filter-count {
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    .range-controls {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    button {
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
      border-radius: 6px;
      padding: 6px 9px;
      font: inherit;
      font-size: 12px;
      cursor: pointer;
    }
    button.active {
      border-color: var(--accent);
      background: var(--accent-soft);
      color: var(--accent);
      font-weight: 700;
    }
    .table-wrap {
      overflow: auto;
      max-height: calc(100vh - 210px);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      padding: 9px 10px;
      border-bottom: 1px solid #edf0f5;
      text-align: right;
      white-space: nowrap;
    }
    th {
      position: sticky;
      top: 0;
      z-index: 2;
      background: #f9fafc;
      color: var(--muted);
      font-weight: 700;
      font-size: 12px;
    }
    th.name, td.name { text-align: left; }
    tbody tr { cursor: pointer; }
    tbody tr:hover { background: #f8fbff; }
    tbody tr.selected {
      background: var(--accent-soft);
      box-shadow: inset 3px 0 0 var(--accent);
    }
    .score {
      font-weight: 700;
      color: var(--accent);
    }
    .growth {
      font-weight: 700;
      color: #7a4f00;
    }
    .tags {
      display: flex;
      align-items: center;
      gap: 5px;
      flex-wrap: wrap;
      max-width: 260px;
    }
    .tag {
      display: inline-flex;
      align-items: center;
      max-width: 120px;
      padding: 2px 6px;
      border: 1px solid #d7e3f8;
      border-radius: 999px;
      background: #f2f7ff;
      color: #285b9f;
      font-size: 11px;
      line-height: 1.35;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .strong {
      color: var(--up);
      font-weight: 700;
    }
    .chart-panel {
      min-height: 640px;
      display: flex;
      flex-direction: column;
    }
    .chart-title {
      min-width: 0;
    }
    .chart-title h2 {
      overflow-wrap: anywhere;
    }
    .meta-line {
      margin-top: 5px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }
    .chart-box {
      padding: 12px 16px 4px;
      flex: 1;
      min-height: 440px;
    }
    .chart-shell {
      position: relative;
      min-height: 520px;
    }
    canvas {
      display: block;
      width: 100%;
      height: 520px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      cursor: crosshair;
      touch-action: none;
      user-select: none;
    }
    canvas.dragging {
      cursor: grabbing;
    }
    .chart-tooltip {
      position: absolute;
      left: 12px;
      top: 12px;
      min-width: 220px;
      max-width: min(360px, calc(100% - 24px));
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: rgba(255,255,255,.94);
      box-shadow: 0 10px 24px rgba(24, 32, 43, .12);
      color: var(--text);
      font-size: 12px;
      line-height: 1.55;
      pointer-events: none;
      display: none;
      z-index: 4;
      white-space: pre-line;
    }
    .chart-tooltip strong {
      display: block;
      margin-bottom: 2px;
      font-size: 13px;
    }
    .chart-help {
      margin-top: 6px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }
    .detail-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      padding: 12px 16px 16px;
    }
    .detail {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px 10px;
      background: #fbfcfe;
      min-width: 0;
    }
    .detail span {
      display: block;
      color: var(--muted);
      font-size: 12px;
    }
    .detail strong {
      display: block;
      margin-top: 4px;
      font-size: 14px;
      overflow-wrap: anywhere;
    }
    .detail.wide {
      grid-column: 1 / -1;
    }
    .ai-panel {
      margin-top: 16px;
      padding: 14px 16px 16px;
      border-top: 1px solid var(--line);
    }
    .ai-panel h2 {
      margin: 0 0 10px;
      font-size: 16px;
      line-height: 1.3;
      letter-spacing: 0;
    }
    .ai-content {
      color: #2c3440;
      font-size: 13px;
      line-height: 1.65;
    }
    .ai-content h3 {
      margin: 14px 0 6px;
      font-size: 14px;
      line-height: 1.35;
    }
    .ai-content p {
      margin: 6px 0;
    }
    .ai-content ul {
      margin: 6px 0 8px 18px;
      padding: 0;
    }
    .empty {
      padding: 24px;
      color: var(--muted);
    }
    @media (max-width: 980px) {
      header { padding: 18px 16px 12px; }
      .review-shell { padding: 12px; }
      .review-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      main {
        grid-template-columns: 1fr;
        padding: 12px;
      }
      .table-wrap { max-height: 420px; }
      .chart-shell { min-height: 430px; }
      canvas { height: 430px; }
      .detail-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <header>
    <h1>A股选股看板</h1>
    <div class="stats">
      <div class="stat"><span>交易日</span><strong id="statDate"></strong></div>
      <div class="stat"><span>扫描股票</span><strong id="statScanned"></strong></div>
      <div class="stat"><span>全量候选</span><strong id="statAllCandidates"></strong></div>
      <div class="stat"><span>展示数量</span><strong id="statCandidates"></strong></div>
      <div class="stat"><span>A/B/C1/C2/D</span><strong id="statPools"></strong></div>
      <div class="stat"><span>结论</span><strong id="statAction"></strong></div>
      <div class="stat"><span>数据失败</span><strong id="statFailed"></strong></div>
    </div>
  </header>
  <div class="review-shell">
    <section class="review-section">
      <h2>今日概览</h2>
      <div class="review-grid" id="reportDataStatus"></div>
      <div class="verdict-box" id="reportFinalAction"></div>
    </section>
    <section class="review-section">
      <h2>当前持仓</h2>
      <p class="review-note">点候选股后的“已买入”记录买入价和数量，这里会按最新收盘价估算浮盈亏。</p>
      <div id="holdingsStatus" class="review-note"></div>
      <div id="holdingsTable"></div>
    </section>
    <section class="review-section">
      <h2>股票K线搜索</h2>
      <div class="stock-search-bar">
        <input id="stockSearchInput" type="search" placeholder="输入代码或名称，例如 600519、平安、康希">
        <button type="button" id="stockSearchButton" class="active">搜索</button>
      </div>
      <div id="stockSearchStatus" class="review-note"></div>
      <div id="stockSearchResults" class="stock-search-results"></div>
    </section>
    <section class="review-section">
      <h2>候选分布</h2>
      <div class="review-grid" id="reportMarketStatus" hidden></div>
      <div class="pool-block" id="reportPoolCounts"></div>
    </section>
    <section class="review-section">
      <h2>题材/主线分布</h2>
      <div id="reportThemeDistribution"></div>
    </section>
    <section class="review-section">
      <h2>候选池</h2>
      <div id="reportPools"></div>
    </section>
    <section class="review-section">
      <h2>AI 复核分析</h2>
      <p class="review-note">AI选股分析员输出，仅用于辅助判断。</p>
      <div class="ai-report-content" id="aiReportContent"></div>
    </section>
  </div>
  <dialog class="chart-dialog" id="chartDialog">
    <main id="chartMain" class="chart-main">
      <section class="chart-panel">
        <div class="section-head">
          <div class="chart-title">
            <h2 id="chartName">历史K线</h2>
            <div class="meta-line" id="chartMeta"></div>
          </div>
          <div class="range-controls">
            <button type="button" data-range="30">1月</button>
            <button type="button" data-range="60">60日</button>
            <button type="button" data-range="120" class="active">120日</button>
            <button type="button" data-range="250">250日</button>
            <button type="button" data-range="500">2年</button>
            <button type="button" data-range="all">全部</button>
            <button type="button" id="resetView">重置</button>
            <button type="button" id="chartStockAi">AI分析</button>
            <button type="button" id="closeChart" class="chart-close-button">关闭</button>
          </div>
        </div>
        <div class="chart-box">
          <div class="chart-shell">
            <canvas id="klineCanvas" width="1200" height="620"></canvas>
            <div class="chart-tooltip" id="chartTooltip"></div>
          </div>
          <div class="chart-help">拖拽平移，滚轮缩放，移动鼠标查看十字光标和 OHLC。</div>
        </div>
        <div class="detail-grid">
          <div class="detail"><span>信号类型</span><strong id="signalType"></strong></div>
          <div class="detail"><span>交易结论</span><strong id="tradeAction"></strong></div>
          <div class="detail"><span>压力区</span><strong id="pressureZone"></strong></div>
          <div class="detail"><span>压力跨度</span><strong id="spanWeeks"></strong></div>
          <div class="detail"><span>买入区</span><strong id="buyZone"></strong></div>
          <div class="detail"><span>止损位</span><strong id="stopLoss"></strong></div>
          <div class="detail"><span>量能口径</span><strong id="activitySource"></strong></div>
          <div class="detail"><span>阻力触达</span><strong id="touches"></strong></div>
          <div class="detail"><span>聚类大小</span><strong id="cluster"></strong></div>
          <div class="detail"><span>ATR%</span><strong id="atr"></strong></div>
          <div class="detail"><span>选中K线</span><strong id="hoverInfo"></strong></div>
          <div class="detail"><span>财务期</span><strong id="financialDate"></strong></div>
          <div class="detail"><span>成长分</span><strong id="growthScore"></strong></div>
          <div class="detail"><span>营收同比</span><strong id="revenueYoy"></strong></div>
          <div class="detail"><span>净利同比</span><strong id="profitYoy"></strong></div>
          <div class="detail"><span>ROE</span><strong id="roe"></strong></div>
          <div class="detail wide"><span>题材标签</span><strong id="tagDetail"></strong></div>
        </div>
      </section>
    </main>
  </dialog>
  <dialog class="position-dialog" id="positionDialog">
    <form method="dialog" class="position-form" id="positionForm">
      <h2 id="positionTitle">记录买入</h2>
      <input type="hidden" id="positionCode">
      <div class="field">
        <label for="positionPrice">买入价格</label>
        <input id="positionPrice" type="number" step="0.001" min="0" required>
      </div>
      <div class="field">
        <label for="positionQuantity">数量</label>
        <input id="positionQuantity" type="number" step="1" min="1" required>
      </div>
      <div class="field">
        <label for="positionDate">买入日期</label>
        <input id="positionDate" type="date" required>
      </div>
      <div class="field">
        <label for="positionNote">备注</label>
        <input id="positionNote" type="text" placeholder="可选">
      </div>
      <div class="dialog-actions">
        <button type="button" id="cancelPosition">取消</button>
        <button type="submit" class="active">保存</button>
      </div>
    </form>
  </dialog>
  <dialog class="stock-ai-dialog" id="stockAiDialog">
    <form method="dialog" class="position-form" id="stockAiForm">
      <h2 id="stockAiTitle">单股 AI 分析</h2>
      <div class="ai-report-content stock-ai-body" id="stockAiContent"></div>
      <div class="dialog-actions">
        <button type="button" id="closeStockAi">关闭</button>
      </div>
    </form>
  </dialog>
  <script>
    const DASHBOARD = __DASHBOARD_DATA__;
    const MIN_BARS = 24;
    const DEFAULT_BARS = 120;
    const state = {
      code: DASHBOARD.candidates[0]?.code || "",
      visibleStart: 0,
      visibleEnd: 0,
      hoverIndex: null,
      hoverX: null,
      hoverY: null,
      dragging: false,
      dragStartX: 0,
      dragStartStart: 0,
      dragStartEnd: 0,
      activeRange: "120",
      tagFilter: "",
      lastDraw: null,
      positions: [],
      holdingsQuote: null,
      extraItems: {},
      extraHistory: {},
      holdingsApiAvailable: true
    };
    const HOLDINGS_API = "api/holdings";
    const STOCK_AI_API = "api/stock-ai";
    const STOCK_SEARCH_API = "api/stock-search";
    const KLINE_API = "api/kline";

    const fmt = (value, digits = 2) => {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
      return Number(value).toFixed(digits);
    };
    const fmtPct = (value, digits = 1) => {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
      return `${Number(value).toFixed(digits)}%`;
    };

    document.getElementById("statDate").textContent = DASHBOARD.meta.latestTradeDate;
    document.getElementById("statScanned").textContent = DASHBOARD.meta.scannedCount;
    document.getElementById("statAllCandidates").textContent = DASHBOARD.meta.allCandidateCount;
    document.getElementById("statCandidates").textContent = DASHBOARD.meta.candidateCount;
    document.getElementById("statPools").textContent = ["A", "B", "C1", "C2", "D"].map(key => DASHBOARD.meta.poolCounts?.[key] || 0).join("/");
    document.getElementById("statAction").textContent = actionLabel(DASHBOARD.meta.finalAction);
    document.getElementById("statFailed").textContent = DASHBOARD.meta.failedCount;

    function escapeHtml(value) {
      return String(value ?? "-")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function kv(label, value) {
      return `<div class="review-kv"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`;
    }

    function poolCount(key) {
      return DASHBOARD.meta.poolCounts?.[key] || 0;
    }

    function actionSentence(action) {
      if (action === "BUY_CHECK") return "有 A 类候选，先看是否在交易区；偏高或次日高开就不追。";
      if (action === "LOW_RISK_BUY_CONFIRMED") return "有 A 类候选，仍需确认开盘位置和持仓风险。";
      if (action === "RIGHT_TAIL_WATCH") return "只有强趋势右侧信号，适合观察，不作为低风险买点。";
      if (action === "WATCH_ONLY") return "今天只观察，等更明确的确认信号。";
      if (action === "DATA_FAILED") return "今日扫描数据不完整，不做交易判断。";
      return "今天没有明确买点。";
    }

    function actionLabel(action) {
      if (action === "BUY_CHECK") return "有A类，人工复核";
      if (action === "LOW_RISK_BUY_CONFIRMED") return "可买入复核";
      if (action === "RIGHT_TAIL_WATCH") return "右尾观察";
      if (action === "WATCH_ONLY") return "只观察";
      if (action === "DATA_FAILED") return "数据失败";
      return "无买点";
    }

    function marketLabel(regime) {
      if (regime === "STRONG_ATTACK") return "强进攻";
      if (regime === "CAUTIOUS") return "谨慎";
      if (regime === "DEFENSIVE") return "防守";
      return "未接入，保守处理";
    }

    function signalLabel(signalType) {
      if (signalType === "A") return "A 周线确认";
      if (signalType === "B") return "B 日线预警";
      if (signalType === "C1") return "C1 右尾观察";
      if (signalType === "C2") return "C2 不追";
      if (signalType === "D") return "D 排除";
      return signalType || "-";
    }

    function buyZoneLabel(status) {
      if (status === "YES") return "在交易区";
      if (status === "ABOVE") return "偏高，等回踩";
      if (status === "BELOW") return "未到交易区";
      return "待复核";
    }

    function buyZoneClass(status) {
      if (status === "YES") return "yes";
      if (status === "ABOVE") return "above";
      if (status === "BELOW") return "below";
      return "unknown";
    }

    function buyZoneCell(item) {
      const status = item.buyZoneStatus || "UNKNOWN";
      const zone = `${fmt(item.buyLow)}-${fmt(item.buyHigh)}`;
      return `<span class="zone-badge ${buyZoneClass(status)}" title="买入区 ${escapeHtml(zone)}">${escapeHtml(buyZoneLabel(status))}</span>`;
    }

    function suggestedAction(item) {
      if (item.signalType === "A") return item.nextDayTradeAction || item.tradeAction || "人工复核";
      if (item.signalType === "B") return "观察，等周线确认";
      if (item.signalType === "C1") return "右尾观察，不追高";
      if (item.signalType === "C2") return item.tradeAction || "不追，等整理";
      if (item.signalType === "D") return exclusionReason(item);
      if ((item.growthScore || 0) > 0) return "成长观察，等突破确认";
      return item.tradeAction || item.hint || item.signalReason || "-";
    }

    function detailCell(item) {
      const rows = [
        ["收盘", fmt(item.latestClose)],
        ["压力区下沿", fmt(item.zoneLow || item.resistance)],
        ["压力区中枢", fmt(item.zoneMid || item.resistance)],
        ["压力区上沿", fmt(item.zoneUpper || item.resistance)],
        ["突破幅度", `${fmt(item.breakoutPct)}%`],
        ["成交额倍数", fmt(item.activityRatio || item.volumeRatio)],
        ["量能来源", item.activitySource === "amount" ? "成交额" : (item.activitySource || "-")],
        ["触碰次数", item.touches || 0],
        ["跨度周", item.spanWeeks || 0],
        ["技术分", fmt(item.score, 1)],
        ["成长分", item.growthScore > 0 ? fmt(item.growthScore, 1) : "-"],
        ["买入区", `${fmt(item.buyLow)}-${fmt(item.buyHigh)}`],
        ["交易止损", fmt(item.tradeStopLoss || item.stopLoss)],
        ["近5日涨幅%", fmt(item.recent5dPct)],
        ["近10日涨幅%", fmt(item.recent10dPct)],
        ["连续涨停", item.consecutiveLimitUpDays >= 2 ? "是" : "否"],
        ["长上影", item.longUpperShadow ? "是" : "否"],
        ["营收同比%", fmtPct(item.revenueYoy)],
        ["净利同比%", fmtPct(item.profitYoy)],
        ["ROE%", fmtPct(item.roe)],
      ];
      const body = rows.map(([label, value]) => `<dt>${escapeHtml(label)}</dt><dd>${escapeHtml(value)}</dd>`).join("");
      return `<details class="stock-detail"><summary>详情</summary><dl>${body}</dl></details>`;
    }

    function compactRows(items) {
      return items.map((item, index) => [
        escapeHtml(index + 1),
        candidateName(item),
        escapeHtml(item.code),
        escapeHtml(item.primaryTag || "未标记"),
        escapeHtml(signalLabel(item.signalType)),
        buyZoneCell(item),
        escapeHtml(suggestedAction(item)),
        actionButtons(item),
        detailCell(item),
      ]);
    }

    function compactHeaders() {
      return [
        { label: "排名" },
        { label: "股票", left: true },
        { label: "代码" },
        { label: "题材", left: true },
        { label: "类型", left: true },
        { label: "交易区" },
        { label: "建议", left: true },
        { label: "操作", left: true },
        { label: "详情", left: true },
      ];
    }

    function poolItems(signalType, limit = null) {
      const items = DASHBOARD.candidates.filter(item => item.signalType === signalType);
      return limit ? items.slice(0, limit) : items;
    }

    function noChaseItems(limit = 5) {
      return DASHBOARD.candidates
        .filter(item => item.signalType === "C2" || item.signalType === "D")
        .slice(0, limit);
    }

    function growthItems(limit = 10) {
      return [...DASHBOARD.candidates]
        .filter(item => item.growthScore > 0)
        .sort((a, b) => (b.growthScore || 0) - (a.growthScore || 0))
        .slice(0, limit);
    }

    function themeRows(limit = 8) {
      const grouped = new Map();
      DASHBOARD.candidates.forEach(item => {
        if (item.signalType === "D") return;
        const theme = item.primaryTag || "未标记";
        const current = grouped.get(theme) || { A: 0, B: 0, C1: 0, C2: 0, total: 0 };
        if (["A", "B", "C1", "C2"].includes(item.signalType)) {
          current[item.signalType] += 1;
          current.total += 1;
        }
        grouped.set(theme, current);
      });
      return Array.from(grouped.entries())
        .sort((a, b) => b[1].total - a[1].total || b[1].A - a[1].A || b[1].B - a[1].B || a[0].localeCompare(b[0], "zh-CN"))
        .slice(0, limit);
    }

    function itemsByTheme(theme) {
      const signalOrder = { A: 0, B: 1, C1: 2, C2: 3, D: 4 };
      return DASHBOARD.candidates
        .filter(item => (item.primaryTag || "未标记") === theme)
        .sort((a, b) => (signalOrder[a.signalType] ?? 9) - (signalOrder[b.signalType] ?? 9) || (b.score || 0) - (a.score || 0));
    }

    function themeJudgement(row) {
      if (row.A > 0 && row.total >= 2) return "有A类共振，重点复核";
      if (row.total >= 4) return "有共振，重点观察";
      if (row.C1 > 0) return "强趋势但波动偏高";
      if (row.total >= 2) return "扩散线";
      return "孤立信号";
    }

    function exclusionReason(item) {
      const text = [item.signalReason, item.tradeAction, item.hint].filter(Boolean).join(" ");
      if ((item.breakoutPct || 0) < 2 || text.includes("突破不足")) return "突破不足2%";
      if ((item.breakoutPct || 0) > 12 || text.includes("过远") || text.includes("远离") || text.includes("不追")) return "距离压力区过远";
      if ((item.activityRatio || item.volumeRatio || 0) < 1.8 || text.includes("量能") || text.includes("成交")) return "成交额倍数不足1.8";
      if (text.includes("基本面") || text.includes("ROE") || text.includes("净利") || text.includes("营收")) return "基本面风险";
      if (!(item.tags || []).length) return "题材孤立";
      return "交易纪律不满足";
    }

    function renderSimpleTable(headers, rows, emptyText) {
      if (!rows.length) {
        return `<p class="review-note">${escapeHtml(emptyText)}</p>`;
      }
      const head = headers.map(header => `<th class="${header.left ? "left" : ""}">${escapeHtml(header.label)}</th>`).join("");
      const body = rows.map(row => {
        return `<tr>${headers.map((header, idx) => `<td class="${header.left ? "left" : ""}">${row[idx]}</td>`).join("")}</tr>`;
      }).join("");
      return `<div class="review-table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
    }

    function poolJumpButton(label, targetId) {
      return `<button type="button" class="link-button" data-pool-target="${escapeHtml(targetId)}">${escapeHtml(label)}</button>`;
    }

    function themeExpandButton(theme) {
      return `<button type="button" class="link-button" data-theme-key="${escapeHtml(theme)}">${escapeHtml(theme)}</button>`;
    }

    function candidateName(item) {
      return escapeHtml(item.name || "-");
    }

    function actionButtons(item) {
      return `
        <span class="action-buttons">
          <button type="button" class="action-button primary" data-chart-code="${escapeHtml(item.code)}">查看K线</button>
          <button type="button" class="action-button" data-ai-code="${escapeHtml(item.code)}">AI分析</button>
          <button type="button" class="action-button buy" data-buy-code="${escapeHtml(item.code)}">已买入</button>
        </span>
      `;
    }

    function coreRows(items, includeTradeConstraints = false) {
      return items.map((item, index) => {
        const row = [
          escapeHtml(index + 1),
          candidateName(item),
          actionButtons(item),
          escapeHtml(item.code),
          escapeHtml(item.primaryTag || "未标记"),
          escapeHtml(fmt(item.latestClose)),
          escapeHtml(fmt(item.zoneUpper || item.resistance)),
          escapeHtml(fmt(item.breakoutPct)),
          escapeHtml(fmt(item.activityRatio || item.volumeRatio)),
          escapeHtml(item.activitySource || "-"),
          escapeHtml(item.touches || 0),
          escapeHtml(item.spanWeeks || 0),
          escapeHtml(fmt(item.score, 1)),
          escapeHtml(item.growthScore > 0 ? fmt(item.growthScore, 1) : "-"),
          escapeHtml(`${fmt(item.buyLow)}-${fmt(item.buyHigh)}`),
          escapeHtml(fmt(item.tradeStopLoss || item.stopLoss)),
        ];
        if (includeTradeConstraints) {
          row.push(
            escapeHtml(item.buyZoneStatus || "UNKNOWN"),
            escapeHtml("高开>3%不追，>5%放弃；跌回压力区上沿取消"),
            escapeHtml(item.nextDayTradeAction || "-"),
          );
        }
        row.push(escapeHtml(item.tradeAction || item.hint || item.signalReason || "-"));
        return row;
      });
    }

    function inlineMarkdown(text) {
      return escapeHtml(text)
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/\\*\\*([^*]+)\\*\\*/g, "<strong>$1</strong>")
        .replace(/__([^_]+)__/g, "<strong>$1</strong>");
    }

    function splitMarkdownTableRow(line) {
      return line.trim().replace(/^\\|/, "").replace(/\\|$/, "").split("|").map(cell => cell.trim());
    }

    function isMarkdownTableSeparator(line) {
      return /^\\s*\\|?\\s*:?-{3,}:?\\s*(\\|\\s*:?-{3,}:?\\s*)+\\|?\\s*$/.test(line);
    }

    function renderMarkdownTable(lines) {
      const headers = splitMarkdownTableRow(lines[0]);
      const rows = lines.slice(2).map(splitMarkdownTableRow);
      const head = headers.map(cell => `<th>${inlineMarkdown(cell)}</th>`).join("");
      const body = rows.map(row => `<tr>${headers.map((_, idx) => `<td>${inlineMarkdown(row[idx] || "")}</td>`).join("")}</tr>`).join("");
      return `<div class="review-table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div>`;
    }

    function renderMarkdownLike(target, text) {
      if (!text) {
        target.innerHTML = '<p class="review-note">未启用或未生成 AI 复核分析。</p>';
        return;
      }
      const lines = text.split("\\n");
      const parts = [];
      let idx = 0;
      while (idx < lines.length) {
        const line = lines[idx].trim();
        if (!line) {
          idx += 1;
          continue;
        }
        if (line.includes("|") && idx + 1 < lines.length && isMarkdownTableSeparator(lines[idx + 1])) {
          const tableLines = [lines[idx], lines[idx + 1]];
          idx += 2;
          while (idx < lines.length && lines[idx].includes("|") && lines[idx].trim()) {
            tableLines.push(lines[idx]);
            idx += 1;
          }
          parts.push(renderMarkdownTable(tableLines));
          continue;
        }
        const heading = line.match(/^(#{1,4})\\s+(.+)$/);
        if (heading) {
          const level = heading[1].length <= 2 ? "h3" : "h4";
          parts.push(`<${level}>${inlineMarkdown(heading[2])}</${level}>`);
          idx += 1;
          continue;
        }
        if (line.startsWith("- ") || line.startsWith("* ") || /^\\d+\\.\\s+/.test(line)) {
          const ordered = /^\\d+\\.\\s+/.test(line);
          const tag = ordered ? "ol" : "ul";
          const items = [];
          while (idx < lines.length) {
            const itemLine = lines[idx].trim();
            if (ordered && /^\\d+\\.\\s+/.test(itemLine)) {
              items.push(`<li>${inlineMarkdown(itemLine.replace(/^\\d+\\.\\s+/, ""))}</li>`);
              idx += 1;
              continue;
            }
            if (!ordered && (itemLine.startsWith("- ") || itemLine.startsWith("* "))) {
              items.push(`<li>${inlineMarkdown(itemLine.replace(/^[-*]\\s+/, ""))}</li>`);
              idx += 1;
              continue;
            }
            break;
          }
          parts.push(`<${tag}>${items.join("")}</${tag}>`);
          continue;
        }
        const paragraphLines = [line];
        idx += 1;
        while (idx < lines.length) {
          const next = lines[idx].trim();
          if (!next || next.startsWith("#") || next.startsWith("- ") || next.startsWith("* ") || /^\\d+\\.\\s+/.test(next) || (next.includes("|") && idx + 1 < lines.length && isMarkdownTableSeparator(lines[idx + 1]))) {
            break;
          }
          paragraphLines.push(next);
          idx += 1;
        }
        parts.push(`<p>${inlineMarkdown(paragraphLines.join(" "))}</p>`);
      }
      target.innerHTML = parts.join("");
    }

    function renderReportSections() {
      const meta = DASHBOARD.meta || {};
      document.getElementById("reportDataStatus").innerHTML = [
        kv("交易日", meta.latestTradeDate),
        kv("今日结论", actionLabel(meta.finalAction)),
        kv("A/B/C1", `${poolCount("A")}/${poolCount("B")}/${poolCount("C1")}`),
        kv("扫描/失败", `${meta.scannedCount}/${meta.failedCount}`),
      ].join("");
      document.getElementById("reportFinalAction").innerHTML = `
        <strong>${escapeHtml(actionLabel(meta.finalAction))}</strong>
        <p class="review-note">${escapeHtml(actionSentence(meta.finalAction))}</p>
        <details class="technical-detail">
          <summary>数据细节</summary>
          <div class="review-grid">
            ${kv("运行时间", meta.runTime || "-")}
            ${kv("数据状态", meta.dataVerdict || "-")}
            ${kv("全量扫描", meta.fullScan ? "是" : "否")}
            ${kv("展示数量", meta.candidateCount)}
          </div>
        </details>
      `;
      document.getElementById("reportMarketStatus").innerHTML = "";
      document.getElementById("reportPoolCounts").innerHTML = renderSimpleTable(
        [{ label: "类型", left: true }, { label: "数量" }, { label: "看什么", left: true }],
        [
          [poolJumpButton("A 周线确认", "pool-A"), escapeHtml(poolCount("A")), "重点看是否还在交易区"],
          [poolJumpButton("B 日线预警", "pool-B"), escapeHtml(poolCount("B")), "先观察，等周线确认"],
          [poolJumpButton("C1 强趋势", "pool-C1"), escapeHtml(poolCount("C1")), "看题材强度，不追高"],
          [poolJumpButton("C2 / D 不追", "pool-no-chase"), escapeHtml(poolCount("C2") + poolCount("D")), "避开追高和无效信号"],
          [poolJumpButton("成长观察", "pool-growth"), escapeHtml(growthItems(50).length), "跟踪，不作为买入清单"],
        ],
        "暂无候选数量。"
      );
      const themes = themeRows();
      document.getElementById("reportThemeDistribution").innerHTML = renderSimpleTable(
        [
          { label: "题材", left: true },
          { label: "A" },
          { label: "B" },
          { label: "C1" },
          { label: "C2" },
          { label: "合计" },
          { label: "建议", left: true },
        ],
        themes.map(([theme, row]) => [
          themeExpandButton(theme),
          escapeHtml(row.A),
          escapeHtml(row.B),
          escapeHtml(row.C1),
          escapeHtml(row.C2),
          escapeHtml(row.total),
          escapeHtml(themeJudgement(row)),
        ]),
        "无候选，无法判断题材共振。"
      ) + '<div id="themeDetail" class="theme-detail" hidden></div>';

      const headers = compactHeaders();
      document.getElementById("reportPools").innerHTML = [
        `<div class="pool-block" id="pool-A"><h3>A类：周线确认突破池 <span class="pill">${poolCount("A")}只</span></h3>${renderSimpleTable(headers, compactRows(poolItems("A")), "今日 A 类数量：0。")}</div>`,
        `<div class="pool-block" id="pool-B"><h3>B类：日线预警池 <span class="pill warn">${poolCount("B")}只</span></h3>${renderSimpleTable(headers, compactRows(poolItems("B", 10)), "今日 B 类数量：0。")}</div>`,
        `<div class="pool-block" id="pool-C1"><h3>C1类：强趋势右尾观察池 <span class="pill warn">${poolCount("C1")}只</span></h3>${renderSimpleTable(headers, compactRows(poolItems("C1", 5)), "今日 C1 类数量：0。")}</div>`,
        `<div class="pool-block" id="pool-no-chase"><h3>C2 / D：不追池</h3>${renderSimpleTable(headers, compactRows(noChaseItems(5)), "无重点不追标的。")}</div>`,
        `<div class="pool-block" id="pool-growth"><h3>成长观察池</h3>${renderSimpleTable(headers, compactRows(growthItems(10)), "暂无成长分可用的观察对象。")}</div>`,
      ].join("");
      renderMarkdownLike(document.getElementById("aiReportContent"), DASHBOARD.aiAnalysis || "");
      bindReportActions();
    }

    function bindReportActions() {
      document.querySelectorAll("[data-pool-target]").forEach(button => {
        if (button.dataset.bound === "1") return;
        button.dataset.bound = "1";
        button.addEventListener("click", () => scrollToPool(button.dataset.poolTarget));
      });
      document.querySelectorAll("[data-theme-key]").forEach(button => {
        if (button.dataset.bound === "1") return;
        button.dataset.bound = "1";
        button.addEventListener("click", () => showThemeDetail(button.dataset.themeKey));
      });
      document.querySelectorAll("[data-chart-code]").forEach(button => {
        if (button.dataset.bound === "1") return;
        button.dataset.bound = "1";
        button.addEventListener("click", () => openChart(button.dataset.chartCode));
      });
      document.querySelectorAll("[data-buy-code]").forEach(button => {
        if (button.dataset.bound === "1") return;
        button.dataset.bound = "1";
        button.addEventListener("click", () => openPositionDialog(button.dataset.buyCode));
      });
      document.querySelectorAll("[data-ai-code]").forEach(button => {
        if (button.dataset.bound === "1") return;
        button.dataset.bound = "1";
        button.addEventListener("click", () => openStockAi(button.dataset.aiCode));
      });
      document.querySelectorAll("[data-delete-position]").forEach(button => {
        if (button.dataset.bound === "1") return;
        button.dataset.bound = "1";
        button.addEventListener("click", () => deletePosition(button.dataset.deletePosition));
      });
    }

    function scrollToPool(targetId) {
      const target = document.getElementById(targetId || "");
      if (!target) return;
      target.scrollIntoView({ behavior: "smooth", block: "start" });
      target.classList.add("flash");
      window.setTimeout(() => target.classList.remove("flash"), 1200);
    }

    function showThemeDetail(theme) {
      const target = document.getElementById("themeDetail");
      if (!target || !theme) return;
      const items = itemsByTheme(theme);
      target.hidden = false;
      target.innerHTML = `
        <div class="theme-detail-head">
          <h3>${escapeHtml(theme)} 候选明细 <span class="pill">${items.length}只</span></h3>
          <button type="button" class="action-button" data-close-theme-detail="1">收起</button>
        </div>
        ${renderSimpleTable(compactHeaders(), compactRows(items), "该题材暂无候选。")}
      `;
      target.querySelector("[data-close-theme-detail]")?.addEventListener("click", () => {
        target.hidden = true;
      });
      bindReportActions();
      target.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    function isChartOpen() {
      const dialog = document.getElementById("chartDialog");
      return Boolean(dialog?.open);
    }

    async function openChart(code) {
      if (!code) return;
      state.code = code;
      state.hoverIndex = null;
      const dialog = document.getElementById("chartDialog");
      if (dialog && !dialog.open) {
        if (typeof dialog.showModal === "function") {
          dialog.showModal();
        } else {
          dialog.setAttribute("open", "open");
        }
      }
      if (!candidateByCode(code) || !historyFor(code).length) {
        setChartLoading(code);
        try {
          await loadKline(code);
        } catch (error) {
          setChartError(code, error.message || String(error));
          return;
        }
      }
      setVisibleByBars(DEFAULT_BARS, "120");
      requestAnimationFrame(() => {
        renderChart();
      });
    }

    async function loadKline(code) {
      const response = await fetch(`${KLINE_API}?code=${encodeURIComponent(code)}`, { cache: "no-store" });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.error || `HTTP ${response.status}`);
      }
      const item = payload.item || {};
      const normalizedCode = item.code || code;
      state.extraItems[normalizedCode] = item;
      state.extraHistory[normalizedCode] = Array.isArray(payload.history) ? payload.history : [];
      state.code = normalizedCode;
    }

    function setChartLoading(code) {
      document.getElementById("chartName").textContent = `${code} K线加载中`;
      document.getElementById("chartMeta").textContent = "正在从服务器缓存读取历史行情...";
      [
        "signalType", "tradeAction", "pressureZone", "spanWeeks",
        "buyZone", "stopLoss", "activitySource", "touches", "cluster", "atr", "hoverInfo",
        "financialDate", "growthScore", "revenueYoy", "profitYoy", "roe", "tagDetail"
      ].forEach(id => {
        document.getElementById(id).textContent = "-";
      });
    }

    function setChartError(code, message) {
      document.getElementById("chartName").textContent = `${code} K线加载失败`;
      document.getElementById("chartMeta").textContent = message || "K线数据暂不可用";
    }

    function closeChart() {
      const dialog = document.getElementById("chartDialog");
      if (!dialog) return;
      if (typeof dialog.close === "function" && dialog.open) {
        dialog.close();
      } else {
        dialog.removeAttribute("open");
      }
      state.hoverIndex = null;
      state.hoverX = null;
      state.hoverY = null;
    }

    async function loadHoldings() {
      try {
        const response = await fetch(HOLDINGS_API, { cache: "no-store" });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const payload = await response.json();
        state.positions = Array.isArray(payload.positions) ? payload.positions : [];
        state.holdingsQuote = payload.quote || null;
        state.holdingsApiAvailable = true;
      } catch (error) {
        state.positions = [];
        state.holdingsQuote = null;
        state.holdingsApiAvailable = false;
        console.error("failed to load holdings", error);
      }
      renderHoldings();
    }

    function renderHoldings() {
      const status = document.getElementById("holdingsStatus");
      const target = document.getElementById("holdingsTable");
      if (!state.holdingsApiAvailable) {
        status.textContent = "持仓 API 不可用，无法落地记录。请检查 a-breakout-holdings-api 服务。";
        target.innerHTML = "";
        return;
      }
      const quote = state.holdingsQuote || {};
      if (!state.positions.length) {
        status.textContent = "暂无持仓记录。";
      } else if (quote.status === "ok") {
        status.textContent = `已记录 ${state.positions.length} 笔持仓；实时行情 ${quote.matched || 0}/${quote.requested || state.positions.length}，更新时间 ${quote.fetched_at || "-"}。`;
      } else if (quote.status === "stale") {
        status.textContent = `已记录 ${state.positions.length} 笔持仓；实时行情临时不可用，暂用上一份实时价 ${quote.stale_price_time || "-"}。`;
      } else if (quote.status === "fallback") {
        status.textContent = `已记录 ${state.positions.length} 笔持仓；实时行情临时不可用，暂用最新可得行情/收盘价 ${quote.fetched_at || "-"}。`;
      } else {
        status.textContent = `已记录 ${state.positions.length} 笔持仓；实时行情暂不可用，页面将回退到最新收盘价。${quote.error ? "原因：" + quote.error : ""}`;
      }
      const rows = state.positions.map(position => {
        const item = candidateByCode(position.code) || {};
        const quoteLatest = Number(position.latest_price || 0);
        const fallbackLatest = Number(item.latestClose || 0);
        const latest = quoteLatest > 0 ? quoteLatest : fallbackLatest;
        const latestLabel = quoteLatest > 0
          ? fmt(quoteLatest, 2)
          : (fallbackLatest > 0 ? `${fmt(fallbackLatest, 2)} 收盘` : "-");
        const buyPrice = Number(position.buy_price || 0);
        const quantity = Number(position.quantity || 0);
        const cost = buyPrice * quantity;
        const marketValue = latest > 0 ? latest * quantity : 0;
        const pnl = marketValue - cost;
        const pnlPct = cost > 0 && latest > 0 ? pnl / cost * 100 : null;
        const pnlClass = pnl >= 0 ? "position-positive" : "position-negative";
        return [
          escapeHtml(position.buy_date || "-"),
          escapeHtml(position.name || item.name || "-"),
          escapeHtml(position.code || "-"),
          escapeHtml(fmt(buyPrice, 3)),
          escapeHtml(quantity),
          escapeHtml(latestLabel),
          escapeHtml(position.price_time || (fallbackLatest > 0 ? DASHBOARD.meta.latestTradeDate : "-")),
          escapeHtml(position.price_source || (fallbackLatest > 0 ? "latest_close" : "-")),
          escapeHtml(fmt(cost, 2)),
          escapeHtml(marketValue > 0 ? fmt(marketValue, 2) : "-"),
          `<span class="${pnlClass}">${latest > 0 ? escapeHtml(fmt(pnl, 2)) : "-"}</span>`,
          `<span class="${pnlClass}">${pnlPct === null ? "-" : escapeHtml(fmt(pnlPct, 2) + "%")}</span>`,
          `<span class="action-buttons"><button type="button" class="action-button primary" data-chart-code="${escapeHtml(position.code)}">查看K线</button><button type="button" class="action-button" data-ai-code="${escapeHtml(position.code)}">AI分析</button><button type="button" class="action-button" data-delete-position="${escapeHtml(position.id)}">删除</button></span>`,
        ];
      });
      target.innerHTML = renderSimpleTable(
        [
          { label: "买入日期" }, { label: "股票", left: true }, { label: "代码" },
          { label: "买入价" }, { label: "数量" }, { label: "当前价" },
          { label: "价格时间" }, { label: "来源" },
          { label: "成本" }, { label: "市值" }, { label: "浮盈亏" }, { label: "浮盈亏%" },
          { label: "操作", left: true },
        ],
        rows,
        "暂无持仓记录。"
      );
      bindReportActions();
    }

    async function searchStocks() {
      const input = document.getElementById("stockSearchInput");
      const status = document.getElementById("stockSearchStatus");
      const target = document.getElementById("stockSearchResults");
      const query = (input.value || "").trim();
      if (!query) {
        status.textContent = "请输入股票代码或名称。";
        target.innerHTML = "";
        return;
      }
      status.textContent = "正在搜索...";
      target.innerHTML = "";
      try {
        const response = await fetch(`${STOCK_SEARCH_API}?q=${encodeURIComponent(query)}&limit=12`, { cache: "no-store" });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(payload.error || `HTTP ${response.status}`);
        }
        const items = Array.isArray(payload.items) ? payload.items : [];
        status.textContent = items.length ? `找到 ${items.length} 只，支持代码和名称模糊搜索。` : "没有匹配结果。";
        const rows = items.map(item => [
          escapeHtml(item.code || "-"),
          escapeHtml(item.name || "-"),
          escapeHtml(item.latest ? fmt(item.latest) : "-"),
          escapeHtml(item.pct_change === null || item.pct_change === undefined ? "-" : `${fmt(item.pct_change)}%`),
          `<button type="button" class="action-button primary" data-chart-code="${escapeHtml(item.code)}">查看K线</button>`,
        ]);
        target.innerHTML = renderSimpleTable(
          [
            { label: "代码" },
            { label: "名称", left: true },
            { label: "最新价" },
            { label: "涨跌幅" },
            { label: "操作", left: true },
          ],
          rows,
          "没有匹配结果。"
        );
        bindReportActions();
      } catch (error) {
        status.textContent = `搜索失败：${error.message || error}`;
      }
    }

    function stockAiPayload(item) {
      return {
        code: item.code,
        name: item.name,
        signalType: item.signalType,
        signalReason: item.signalReason,
        primaryTag: item.primaryTag,
        tags: item.tags || [],
        latestClose: item.latestClose,
        buyZoneStatus: item.buyZoneStatus,
        suggestedAction: suggestedAction(item),
        zoneLow: item.zoneLow,
        zoneMid: item.zoneMid,
        zoneUpper: item.zoneUpper || item.resistance,
        breakoutPct: item.breakoutPct,
        activityRatio: item.activityRatio || item.volumeRatio,
        activitySource: item.activitySource,
        touches: item.touches,
        spanWeeks: item.spanWeeks,
        score: item.score,
        growthScore: item.growthScore,
        revenueYoy: item.revenueYoy,
        profitYoy: item.profitYoy,
        roe: item.roe,
        debtToAssets: item.debtToAssets,
        buyLow: item.buyLow,
        buyHigh: item.buyHigh,
        tradeStopLoss: item.tradeStopLoss || item.stopLoss,
        recent5dPct: item.recent5dPct,
        recent10dPct: item.recent10dPct,
        longUpperShadow: item.longUpperShadow,
      };
    }

    async function openStockAi(code) {
      const item = candidateByCode(code);
      if (!item) {
        window.alert("未找到这只股票的候选数据，暂不能做单股 AI 分析。");
        return;
      }
      const dialog = document.getElementById("stockAiDialog");
      const title = document.getElementById("stockAiTitle");
      const content = document.getElementById("stockAiContent");
      title.textContent = `单股 AI 分析：${item.code} ${item.name}`;
      content.innerHTML = '<p class="review-note">正在生成单股分析...</p>';
      if (typeof dialog.showModal === "function") {
        dialog.showModal();
      } else {
        dialog.setAttribute("open", "open");
      }
      try {
        const response = await fetch(STOCK_AI_API, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({ stock: stockAiPayload(item) }),
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(payload.error || `HTTP ${response.status}`);
        }
        renderMarkdownLike(content, payload.analysis || "单股 AI 没有返回内容。");
      } catch (error) {
        content.innerHTML = `<p class="review-note">单股 AI 暂不可用：${escapeHtml(error.message || error)}</p>`;
      }
    }

    function openPositionDialog(code) {
      const item = candidateByCode(code);
      if (!item) return;
      const dialog = document.getElementById("positionDialog");
      document.getElementById("positionTitle").textContent = `记录买入：${item.code} ${item.name}`;
      document.getElementById("positionCode").value = item.code;
      document.getElementById("positionPrice").value = item.latestClose || "";
      document.getElementById("positionQuantity").value = "";
      document.getElementById("positionDate").value = new Date().toISOString().slice(0, 10);
      document.getElementById("positionNote").value = "";
      if (typeof dialog.showModal === "function") {
        dialog.showModal();
      } else {
        const price = window.prompt("买入价格", item.latestClose || "");
        if (!price) return;
        const quantity = window.prompt("数量", "");
        if (!quantity) return;
        savePosition({ code: item.code, buy_price: price, quantity, buy_date: new Date().toISOString().slice(0, 10), note: "" });
      }
    }

    async function savePosition(payload) {
      const item = candidateByCode(payload.code);
      const body = {
        code: payload.code,
        name: item?.name || "",
        buy_price: Number(payload.buy_price),
        quantity: Number(payload.quantity),
        buy_date: payload.buy_date,
        note: payload.note || "",
      };
      try {
        const response = await fetch(HOLDINGS_API, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!response.ok) {
          const payload = await response.json().catch(() => ({}));
          throw new Error(payload.error || `HTTP ${response.status}`);
        }
        await loadHoldings();
      } catch (error) {
        window.alert(`持仓保存失败：${error.message || error}`);
      }
    }

    async function deletePosition(positionId) {
      if (!positionId || !window.confirm("删除这条持仓记录？")) return;
      try {
        const response = await fetch(`${HOLDINGS_API}/${encodeURIComponent(positionId)}`, { method: "DELETE" });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        await loadHoldings();
      } catch (error) {
        window.alert(`持仓删除失败：${error.message || error}`);
      }
    }

    function candidateByCode(code) {
      return DASHBOARD.candidates.find(item => item.code === code) || state.extraItems[code];
    }

    function historyFor(code) {
      return (DASHBOARD.history[code] || state.extraHistory[code] || []).filter(row =>
        row.open !== null && row.high !== null && row.low !== null && row.close !== null
      );
    }

    function filteredCandidates() {
      if (!state.tagFilter) return DASHBOARD.candidates;
      return DASHBOARD.candidates.filter(item => (item.tags || []).includes(state.tagFilter));
    }

    function tagOptions() {
      const counts = new Map();
      DASHBOARD.candidates.forEach(item => {
        (item.tags || []).forEach(tag => counts.set(tag, (counts.get(tag) || 0) + 1));
      });
      return Array.from(counts.entries()).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0], "zh-CN"));
    }

    function renderTagFilter() {
      const select = document.getElementById("tagFilter");
      select.textContent = "";
      const all = document.createElement("option");
      all.value = "";
      all.textContent = "全部标签";
      select.appendChild(all);
      tagOptions().forEach(([tag, count]) => {
        const option = document.createElement("option");
        option.value = tag;
        option.textContent = `${tag} (${count})`;
        select.appendChild(option);
      });
      select.value = state.tagFilter;
      document.getElementById("filterCount").textContent =
        `${filteredCandidates().length}/${DASHBOARD.candidates.length} 只`;
    }

    function clamp(value, min, max) {
      return Math.max(min, Math.min(max, value));
    }

    function setActiveRange(value) {
      state.activeRange = value;
      document.querySelectorAll("[data-range]").forEach(item => {
        item.classList.toggle("active", item.dataset.range === value);
      });
    }

    function setVisibleByBars(bars, activeRange = String(bars)) {
      const data = historyFor(state.code);
      const total = data.length;
      const span = bars === "all" ? total : clamp(Number(bars) || DEFAULT_BARS, MIN_BARS, Math.max(MIN_BARS, total));
      state.visibleEnd = total;
      state.visibleStart = Math.max(0, total - span);
      state.hoverIndex = null;
      setActiveRange(activeRange);
    }

    function ensureVisibleWindow() {
      const total = historyFor(state.code).length;
      if (!total) {
        state.visibleStart = 0;
        state.visibleEnd = 0;
        return;
      }
      if (state.visibleEnd <= state.visibleStart || state.visibleEnd > total) {
        setVisibleByBars(state.activeRange === "all" ? "all" : Number(state.activeRange) || DEFAULT_BARS, state.activeRange);
        return;
      }
      const span = clamp(state.visibleEnd - state.visibleStart, MIN_BARS, total);
      state.visibleEnd = clamp(state.visibleEnd, span, total);
      state.visibleStart = state.visibleEnd - span;
    }

    function renderRows() {
      const tbody = document.getElementById("candidateRows");
      tbody.textContent = "";
      const rows = filteredCandidates();
      document.getElementById("filterCount").textContent = `${rows.length}/${DASHBOARD.candidates.length} 只`;
      if (!rows.length) {
        const row = document.createElement("tr");
        const cell = document.createElement("td");
        cell.colSpan = 13;
        cell.className = "empty";
        cell.textContent = DASHBOARD.candidates.length ? "当前标签下没有候选。" : "今日没有符合条件的候选。";
        row.appendChild(cell);
        tbody.appendChild(row);
        return;
      }
      for (const item of rows) {
        const row = document.createElement("tr");
        row.dataset.code = item.code;
        if (item.code === state.code) row.classList.add("selected");
        const cells = [
          item.rank,
          item.signalType || "-",
          `${item.code} ${item.name}`,
          fmt(item.latestClose),
          item.circMv > 0 ? fmt(item.circMv, 1) : "-",
          fmt(item.zoneUpper || item.resistance),
          `${fmt(item.breakoutPct)}%`,
          `${fmt(item.activityRatio || item.volumeRatio)} ${item.activitySource === "amount" ? "额" : "量"}`,
          fmt(item.volumeTrend),
          `${fmt(item.atrPct)}%`,
          fmt(item.score, 1),
          item.growthScore > 0 ? fmt(item.growthScore, 1) : "-",
          ""
        ];
        cells.forEach((value, idx) => {
          const cell = document.createElement("td");
          if (idx === 12) {
            cell.className = "name";
            const tags = document.createElement("div");
            tags.className = "tags";
            (item.tags || []).slice(0, 4).forEach(tag => {
              const tagNode = document.createElement("span");
              tagNode.className = "tag";
              tagNode.title = tag;
              tagNode.textContent = tag;
              tags.appendChild(tagNode);
            });
            if (!(item.tags || []).length) {
              tags.textContent = "-";
            }
            cell.appendChild(tags);
          } else {
            cell.textContent = value;
          }
          if (idx === 1 && item.signalType === "A") cell.classList.add("strong");
          if (idx === 2) cell.className = "name";
          if (idx === 6 && item.breakoutPct > 0) cell.classList.add("strong");
          if (idx === 10) cell.classList.add("score");
          if (idx === 11) cell.classList.add("growth");
          row.appendChild(cell);
        });
        row.addEventListener("click", () => {
          state.code = item.code;
          state.hoverIndex = null;
          setVisibleByBars(DEFAULT_BARS, "120");
          renderRows();
          renderChart();
        });
        tbody.appendChild(row);
      }
    }

    const canvas = document.getElementById("klineCanvas");
    const ctx = canvas.getContext("2d");
    const tooltip = document.getElementById("chartTooltip");

    function setupCanvas() {
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(600, Math.floor(rect.width * dpr));
      canvas.height = Math.max(360, Math.floor(rect.height * dpr));
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function yScale(value, min, max, top, bottom) {
      return top + (max - value) / (max - min) * (bottom - top);
    }

    function priceAtY(y, min, max, top, bottom) {
      return max - (y - top) / (bottom - top) * (max - min);
    }

    function drawLine(y, color, label) {
      const width = canvas.clientWidth;
      const draw = state.lastDraw || { left: 58, right: width - 58 };
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(draw.left, y);
      ctx.lineTo(draw.right, y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = color;
      ctx.font = "12px -apple-system, BlinkMacSystemFont, sans-serif";
      ctx.fillText(label, draw.right - 72, y - 5);
      ctx.restore();
    }

    function drawLabel(text, x, y, fillStyle, align = "right") {
      ctx.save();
      ctx.font = "12px -apple-system, BlinkMacSystemFont, sans-serif";
      const metrics = ctx.measureText(text);
      const paddingX = 6;
      const w = metrics.width + paddingX * 2;
      const h = 20;
      const left = align === "right" ? x - w : x;
      ctx.fillStyle = fillStyle;
      ctx.fillRect(left, y - h / 2, w, h);
      ctx.fillStyle = "#fff";
      ctx.textBaseline = "middle";
      ctx.fillText(text, left + paddingX, y);
      ctx.restore();
    }

    function renderChart() {
      setupCanvas();
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      ctx.clearRect(0, 0, width, height);
      tooltip.style.display = "none";

      const item = candidateByCode(state.code);
      if (!item) {
        document.getElementById("chartName").textContent = "历史K线";
        document.getElementById("chartMeta").textContent = "";
        [
          "signalType", "tradeAction", "pressureZone", "spanWeeks",
          "buyZone", "stopLoss", "activitySource", "touches", "cluster", "atr", "hoverInfo",
          "financialDate", "growthScore", "revenueYoy", "profitYoy", "roe", "tagDetail"
        ].forEach(id => {
          document.getElementById(id).textContent = "-";
        });
        return;
      }
      const allData = historyFor(item.code);
      ensureVisibleWindow();
      const data = allData.slice(state.visibleStart, state.visibleEnd);
      const hasStrategyData = item.strategyData !== false;

      document.getElementById("chartName").textContent = `${item.code} ${item.name}`;
      document.getElementById("chartMeta").textContent = hasStrategyData
        ? `${item.signalType || "-"}类 / 收盘 ${fmt(item.latestClose)} / 市值 ${item.circMv > 0 ? fmt(item.circMv, 1) + '亿' : '-'} / 压力上沿 ${fmt(item.zoneUpper || item.resistance)} / 突破 ${fmt(item.breakoutPct)}% / 技术 ${fmt(item.score, 1)} / 成长 ${item.growthScore > 0 ? fmt(item.growthScore, 1) : '-'} / ${item.hint}`
        : `搜索K线 / 最新收盘 ${fmt(item.latestClose)} / ${item.hint || "非策略候选，仅查看历史走势"}`;
      document.getElementById("signalType").textContent = `${item.signalType || "-"} ${item.signalReason || ""}`.trim();
      document.getElementById("tradeAction").textContent = item.tradeAction || "-";
      document.getElementById("pressureZone").textContent = hasStrategyData ? `${fmt(item.zoneLow || item.resistance)} - ${fmt(item.zoneUpper || item.resistance)}` : "-";
      document.getElementById("spanWeeks").textContent = hasStrategyData && item.spanWeeks ? `${item.spanWeeks} 周` : "-";
      document.getElementById("buyZone").textContent = hasStrategyData ? `${fmt(item.buyLow)} - ${fmt(item.buyHigh)}` : "-";
      document.getElementById("stopLoss").textContent = hasStrategyData ? `${fmt(item.tradeStopLoss || item.stopLoss)} / 结构 ${fmt(item.structureStopLoss || item.stopLoss)}` : "-";
      document.getElementById("activitySource").textContent = hasStrategyData ? `${item.activitySource === "amount" ? "成交额" : "成交量"} / ${fmt(item.activityRatio || item.volumeRatio)}倍` : "-";
      document.getElementById("touches").textContent = hasStrategyData ? `${item.touches} 次` : "-";
      document.getElementById("cluster").textContent = hasStrategyData ? `${item.clusterSize || "-"} 根K线` : "-";
      document.getElementById("atr").textContent = hasStrategyData ? `${fmt(item.atrPct)}%` : "-";
      document.getElementById("financialDate").textContent = item.financialEndDate || "-";
      document.getElementById("growthScore").textContent = item.growthScore > 0 ? fmt(item.growthScore, 1) : "-";
      document.getElementById("revenueYoy").textContent = fmtPct(item.revenueYoy);
      document.getElementById("profitYoy").textContent = fmtPct(item.profitYoy);
      document.getElementById("roe").textContent = fmtPct(item.roe);
      document.getElementById("tagDetail").textContent = (item.tags || []).length ? item.tags.join(" / ") : "-";

      if (!data.length) {
        ctx.fillStyle = "#667085";
        ctx.font = "15px -apple-system, BlinkMacSystemFont, sans-serif";
        ctx.fillText("没有可用K线数据", 32, 48);
        return;
      }

      const left = 58;
      const right = width - 58;
      const top = 22;
      const priceBottom = Math.max(240, height - 128);
      const volumeTop = priceBottom + 18;
      const volumeBottom = height - 26;
      const highs = data.map(row => row.high);
      const lows = data.map(row => row.low);
      if (hasStrategyData) {
        highs.push(item.zoneUpper || item.resistance, item.zoneMid || item.resistance, item.buyHigh, item.stopLoss);
        lows.push(item.zoneLow || item.resistance, item.zoneMid || item.resistance, item.buyLow, item.stopLoss);
      }
      let maxPrice = Math.max(...highs);
      let minPrice = Math.min(...lows);
      const pad = Math.max((maxPrice - minPrice) * 0.08, maxPrice * 0.01, 0.5);
      maxPrice += pad;
      minPrice -= pad;
      const maxVolume = Math.max(...data.map(row => row.volume || 0), 1);
      const step = (right - left) / Math.max(data.length, 1);
      const candleWidth = Math.max(3, Math.min(12, step * 0.58));
      state.lastDraw = {
        left,
        right,
        top,
        priceBottom,
        volumeTop,
        volumeBottom,
        minPrice,
        maxPrice,
        step,
        total: allData.length
      };

      ctx.strokeStyle = "#edf0f5";
      ctx.fillStyle = "#667085";
      ctx.font = "12px -apple-system, BlinkMacSystemFont, sans-serif";
      for (let i = 0; i <= 4; i += 1) {
        const y = top + (priceBottom - top) * i / 4;
        const price = maxPrice - (maxPrice - minPrice) * i / 4;
        ctx.beginPath();
        ctx.moveTo(left, y);
        ctx.lineTo(right, y);
        ctx.stroke();
        ctx.fillText(fmt(price), 6, y + 4);
      }
      ctx.strokeStyle = "#edf0f5";
      ctx.beginPath();
      ctx.moveTo(left, volumeTop);
      ctx.lineTo(right, volumeTop);
      ctx.stroke();

      if (hasStrategyData) {
        const buyY1 = yScale(item.buyHigh, minPrice, maxPrice, top, priceBottom);
        const buyY2 = yScale(item.buyLow, minPrice, maxPrice, top, priceBottom);
        ctx.fillStyle = "rgba(31, 111, 235, 0.08)";
        ctx.fillRect(left, Math.min(buyY1, buyY2), right - left, Math.abs(buyY2 - buyY1));
        const zoneLowY = yScale(item.zoneLow || item.resistance, minPrice, maxPrice, top, priceBottom);
        const zoneUpperY = yScale(item.zoneUpper || item.resistance, minPrice, maxPrice, top, priceBottom);
        ctx.fillStyle = "rgba(183, 110, 0, 0.09)";
        ctx.fillRect(left, Math.min(zoneLowY, zoneUpperY), right - left, Math.abs(zoneUpperY - zoneLowY));
        drawLine(yScale(item.zoneUpper || item.resistance, minPrice, maxPrice, top, priceBottom), "#1f6feb", "压力上沿");
        drawLine(yScale(item.zoneMid || item.resistance, minPrice, maxPrice, top, priceBottom), "#667085", "压力中枢");
        drawLine(yScale(item.stopLoss, minPrice, maxPrice, top, priceBottom), "#b76e00", "止损");
      }

      data.forEach((row, idx) => {
        const x = left + step * idx + step / 2;
        const isUp = row.close >= row.open;
        const color = isUp ? "#d33f49" : "#16835f";
        const yHigh = yScale(row.high, minPrice, maxPrice, top, priceBottom);
        const yLow = yScale(row.low, minPrice, maxPrice, top, priceBottom);
        const yOpen = yScale(row.open, minPrice, maxPrice, top, priceBottom);
        const yClose = yScale(row.close, minPrice, maxPrice, top, priceBottom);
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.moveTo(x, yHigh);
        ctx.lineTo(x, yLow);
        ctx.stroke();
        const bodyY = Math.min(yOpen, yClose);
        const bodyH = Math.max(1, Math.abs(yOpen - yClose));
        ctx.fillRect(x - candleWidth / 2, bodyY, candleWidth, bodyH);

        const vHeight = (row.volume || 0) / maxVolume * (volumeBottom - volumeTop);
        ctx.globalAlpha = 0.42;
        ctx.fillRect(x - candleWidth / 2, volumeBottom - vHeight, candleWidth, vHeight);
        ctx.globalAlpha = 1;
      });

      const first = data[0];
      const last = data[data.length - 1];
      ctx.fillStyle = "#667085";
      ctx.fillText(first.date, left, height - 8);
      ctx.fillText(last.date, Math.max(left, right - 92), height - 8);
      ctx.fillText(`${state.visibleEnd - state.visibleStart}/${allData.length} 根`, Math.max(left, right - 172), 16);

      if (state.hoverIndex !== null && allData[state.hoverIndex]) {
        const row = allData[state.hoverIndex];
        const localIndex = state.hoverIndex - state.visibleStart;
        if (localIndex >= 0 && localIndex < data.length) {
          const x = left + step * localIndex + step / 2;
          const closeY = yScale(row.close, minPrice, maxPrice, top, priceBottom);
          const hoverY = state.hoverY !== null ? clamp(state.hoverY, top, volumeBottom) : closeY;
          const hoverPrice = priceAtY(clamp(hoverY, top, priceBottom), minPrice, maxPrice, top, priceBottom);
          ctx.strokeStyle = "rgba(24, 32, 43, 0.38)";
          ctx.setLineDash([3, 3]);
          ctx.beginPath();
          ctx.moveTo(x, top);
          ctx.lineTo(x, volumeBottom);
          ctx.stroke();
          ctx.beginPath();
          ctx.moveTo(left, hoverY);
          ctx.lineTo(right, hoverY);
          ctx.stroke();
          ctx.setLineDash([]);
          drawLabel(fmt(hoverPrice), right, hoverY, "#18202b");
          const change = row.open ? (row.close / row.open - 1) * 100 : 0;
          const volumeText = row.volume ? Number(row.volume).toLocaleString("zh-CN") : "-";
          document.getElementById("hoverInfo").textContent =
            `${row.date} 开 ${fmt(row.open)} 高 ${fmt(row.high)} 低 ${fmt(row.low)} 收 ${fmt(row.close)}`;
          tooltip.textContent =
            `${item.code} ${item.name} · ${row.date}\n` +
            `开 ${fmt(row.open)}　高 ${fmt(row.high)}　低 ${fmt(row.low)}　收 ${fmt(row.close)}\n` +
            `涨跌 ${change >= 0 ? "+" : ""}${fmt(change)}%　量 ${volumeText}`;
          const rect = canvas.getBoundingClientRect();
          const shellRect = canvas.parentElement.getBoundingClientRect();
          const tipX = clamp(x + rect.left - shellRect.left + 12, 8, shellRect.width - 260);
          const tipY = clamp((state.hoverY ?? top) + rect.top - shellRect.top + 12, 8, shellRect.height - 82);
          tooltip.style.left = `${tipX}px`;
          tooltip.style.top = `${tipY}px`;
          tooltip.style.display = "block";
        }
      } else {
        document.getElementById("hoverInfo").textContent = `${last.date} 收 ${fmt(last.close)}`;
      }
    }

    function indexFromPointer(event) {
      const item = candidateByCode(state.code);
      const data = item ? historyFor(item.code) : [];
      const draw = state.lastDraw;
      if (!data.length || !draw) return null;
      const rect = canvas.getBoundingClientRect();
      const localX = event.clientX - rect.left;
      const localY = event.clientY - rect.top;
      const visibleCount = state.visibleEnd - state.visibleStart;
      const idx = Math.round((localX - draw.left - draw.step / 2) / draw.step);
      return {
        index: clamp(state.visibleStart + idx, state.visibleStart, Math.max(state.visibleStart, state.visibleEnd - 1)),
        x: localX,
        y: localY,
        visibleCount
      };
    }

    function moveWindow(start, end) {
      const total = historyFor(state.code).length;
      const span = clamp(end - start, MIN_BARS, total);
      const nextStart = clamp(start, 0, Math.max(0, total - span));
      state.visibleStart = nextStart;
      state.visibleEnd = nextStart + span;
      if (state.visibleEnd > total) {
        state.visibleEnd = total;
        state.visibleStart = Math.max(0, total - span);
      }
    }

    function zoomAt(event) {
      const data = historyFor(state.code);
      if (!data.length || !state.lastDraw) return;
      event.preventDefault();
      const pointer = indexFromPointer(event);
      if (!pointer) return;
      const total = data.length;
      const currentSpan = state.visibleEnd - state.visibleStart;
      const scale = event.deltaY < 0 ? 0.82 : 1.22;
      const nextSpan = clamp(Math.round(currentSpan * scale), MIN_BARS, total);
      const ratio = clamp((pointer.index - state.visibleStart) / Math.max(1, currentSpan), 0, 1);
      const nextStart = Math.round(pointer.index - nextSpan * ratio);
      moveWindow(nextStart, nextStart + nextSpan);
      setActiveRange("");
      state.hoverIndex = pointer.index;
      state.hoverX = pointer.x;
      state.hoverY = pointer.y;
      renderChart();
    }

    canvas.addEventListener("pointerdown", (event) => {
      const draw = state.lastDraw;
      if (!draw) return;
      state.dragging = true;
      state.dragStartX = event.clientX;
      state.dragStartStart = state.visibleStart;
      state.dragStartEnd = state.visibleEnd;
      canvas.classList.add("dragging");
      canvas.setPointerCapture(event.pointerId);
    });

    canvas.addEventListener("pointermove", (event) => {
      const pointer = indexFromPointer(event);
      if (!pointer) return;
      state.hoverX = pointer.x;
      state.hoverY = pointer.y;
      if (state.dragging && state.lastDraw) {
        const deltaPx = event.clientX - state.dragStartX;
        const barsDelta = -Math.round(deltaPx / state.lastDraw.step);
        moveWindow(state.dragStartStart + barsDelta, state.dragStartEnd + barsDelta);
        setActiveRange("");
        const movedPointer = indexFromPointer(event);
        if (movedPointer) {
          pointer.index = movedPointer.index;
        }
      }
      state.hoverIndex = pointer.index;
      renderChart();
    });

    function endDrag(event) {
      state.dragging = false;
      canvas.classList.remove("dragging");
      if (event.pointerId !== undefined && canvas.hasPointerCapture(event.pointerId)) {
        canvas.releasePointerCapture(event.pointerId);
      }
    }

    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", endDrag);
    canvas.addEventListener("wheel", zoomAt, { passive: false });
    canvas.addEventListener("mouseleave", () => {
      if (!state.dragging) {
        state.hoverIndex = null;
        state.hoverX = null;
        state.hoverY = null;
        renderChart();
      }
    });

    const tagFilterElement = document.getElementById("tagFilter");
    if (tagFilterElement) {
      tagFilterElement.addEventListener("change", (event) => {
        state.tagFilter = event.target.value;
        const rows = filteredCandidates();
        if (!rows.some(item => item.code === state.code)) {
          state.code = rows[0]?.code || "";
          state.hoverIndex = null;
          setVisibleByBars(DEFAULT_BARS, "120");
        }
        renderTagFilter();
        renderRows();
        renderChart();
      });
    }

    document.querySelectorAll("[data-range]").forEach(button => {
      button.addEventListener("click", () => {
        const range = button.dataset.range;
        setVisibleByBars(range === "all" ? "all" : Number(range), range);
        renderChart();
      });
    });

    document.getElementById("resetView").addEventListener("click", () => {
      setVisibleByBars(DEFAULT_BARS, "120");
      renderChart();
    });

    window.addEventListener("resize", () => {
      if (isChartOpen()) renderChart();
    });
    renderReportSections();
    loadHoldings();

    document.getElementById("closeChart").addEventListener("click", closeChart);
    document.getElementById("chartDialog").addEventListener("click", (event) => {
      if (event.target === event.currentTarget) closeChart();
    });
    document.getElementById("chartStockAi").addEventListener("click", () => {
      openStockAi(state.code);
    });
    document.getElementById("stockSearchButton").addEventListener("click", searchStocks);
    document.getElementById("stockSearchInput").addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        searchStocks();
      }
    });

    document.getElementById("cancelPosition").addEventListener("click", () => {
      document.getElementById("positionDialog").close();
    });
    document.getElementById("closeStockAi").addEventListener("click", () => {
      document.getElementById("stockAiDialog").close();
    });
    document.getElementById("positionForm").addEventListener("submit", (event) => {
      event.preventDefault();
      const code = document.getElementById("positionCode").value;
      savePosition({
        code,
        buy_price: document.getElementById("positionPrice").value,
        quantity: document.getElementById("positionQuantity").value,
        buy_date: document.getElementById("positionDate").value,
        note: document.getElementById("positionNote").value,
      }).then(() => {
        document.getElementById("positionDialog").close();
      });
    });
  </script>
</body>
</html>
"""
