from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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


HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>A股突破选股复核</title>
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
    <h1>A股突破选股复核</h1>
    <div class="stats">
      <div class="stat"><span>交易日</span><strong id="statDate"></strong></div>
      <div class="stat"><span>扫描股票</span><strong id="statScanned"></strong></div>
      <div class="stat"><span>全量候选</span><strong id="statAllCandidates"></strong></div>
      <div class="stat"><span>展示数量</span><strong id="statCandidates"></strong></div>
      <div class="stat"><span>A/B/C1/C2/D</span><strong id="statPools"></strong></div>
      <div class="stat"><span>数据失败</span><strong id="statFailed"></strong></div>
    </div>
  </header>
  <main>
    <section class="table-panel">
      <div class="section-head">
        <h2>候选结果</h2>
        <div class="filter-tools">
          <select id="tagFilter" aria-label="按题材标签筛选">
            <option value="">全部标签</option>
          </select>
          <span class="filter-count" id="filterCount"></span>
        </div>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>排名</th>
              <th>类型</th>
              <th class="name">股票</th>
              <th>收盘</th>
              <th>市值(亿)</th>
              <th>压力上沿</th>
              <th>突破</th>
              <th>成交倍数</th>
              <th>量趋势</th>
              <th>ATR%</th>
              <th>得分</th>
              <th>成长</th>
              <th class="name">标签</th>
            </tr>
          </thead>
          <tbody id="candidateRows"></tbody>
        </table>
      </div>
    </section>
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
      <div class="ai-panel" id="aiPanel">
        <h2>AI选股分析员</h2>
        <div class="ai-content" id="aiContent"></div>
      </div>
    </section>
  </main>
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
      lastDraw: null
    };

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
    document.getElementById("statFailed").textContent = DASHBOARD.meta.failedCount;

    function candidateByCode(code) {
      return DASHBOARD.candidates.find(item => item.code === code);
    }

    function historyFor(code) {
      return (DASHBOARD.history[code] || []).filter(row =>
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

      document.getElementById("chartName").textContent = `${item.code} ${item.name}`;
      document.getElementById("chartMeta").textContent =
        `${item.signalType || "-"}类 / 收盘 ${fmt(item.latestClose)} / 市值 ${item.circMv > 0 ? fmt(item.circMv, 1) + '亿' : '-'} / 压力上沿 ${fmt(item.zoneUpper || item.resistance)} / 突破 ${fmt(item.breakoutPct)}% / 技术 ${fmt(item.score, 1)} / 成长 ${item.growthScore > 0 ? fmt(item.growthScore, 1) : '-'} / ${item.hint}`;
      document.getElementById("signalType").textContent = `${item.signalType || "-"} ${item.signalReason || ""}`.trim();
      document.getElementById("tradeAction").textContent = item.tradeAction || "-";
      document.getElementById("pressureZone").textContent = `${fmt(item.zoneLow || item.resistance)} - ${fmt(item.zoneUpper || item.resistance)}`;
      document.getElementById("spanWeeks").textContent = item.spanWeeks ? `${item.spanWeeks} 周` : "-";
      document.getElementById("buyZone").textContent = `${fmt(item.buyLow)} - ${fmt(item.buyHigh)}`;
      document.getElementById("stopLoss").textContent = `${fmt(item.tradeStopLoss || item.stopLoss)} / 结构 ${fmt(item.structureStopLoss || item.stopLoss)}`;
      document.getElementById("activitySource").textContent = `${item.activitySource === "amount" ? "成交额" : "成交量"} / ${fmt(item.activityRatio || item.volumeRatio)}倍`;
      document.getElementById("touches").textContent = `${item.touches} 次`;
      document.getElementById("cluster").textContent = `${item.clusterSize || "-"} 根K线`;
      document.getElementById("atr").textContent = `${fmt(item.atrPct)}%`;
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
      highs.push(item.zoneUpper || item.resistance, item.zoneMid || item.resistance, item.buyHigh, item.stopLoss);
      lows.push(item.zoneLow || item.resistance, item.zoneMid || item.resistance, item.buyLow, item.stopLoss);
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

    function renderAIAnalysis() {
      const panel = document.getElementById("aiPanel");
      const content = document.getElementById("aiContent");
      const text = DASHBOARD.aiAnalysis || "";
      if (!text) {
        panel.style.display = "none";
        return;
      }
      panel.style.display = "block";
      content.textContent = "";
      let list = null;
      text.split("\\n").forEach(rawLine => {
        const line = rawLine.trim();
        if (!line) {
          list = null;
          return;
        }
        if (line.startsWith("## ")) {
          list = null;
          const heading = document.createElement("h3");
          heading.textContent = line.replace(/^##\\s+/, "");
          content.appendChild(heading);
          return;
        }
        if (line.startsWith("- ") || /^\\d+\\.\\s+/.test(line)) {
          if (!list) {
            list = document.createElement("ul");
            content.appendChild(list);
          }
          const item = document.createElement("li");
          item.textContent = line.replace(/^[-*]\\s+/, "").replace(/^\\d+\\.\\s+/, "");
          list.appendChild(item);
          return;
        }
        list = null;
        const paragraph = document.createElement("p");
        paragraph.textContent = line;
        content.appendChild(paragraph);
      });
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

    document.getElementById("tagFilter").addEventListener("change", (event) => {
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

    window.addEventListener("resize", renderChart);
    renderTagFilter();
    renderRows();
    renderAIAnalysis();
    setVisibleByBars(DEFAULT_BARS, "120");
    renderChart();
  </script>
</body>
</html>
"""
