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
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "breakout_dashboard.html"
    payload = {
        "meta": {
            "latestTradeDate": latest_trade_date,
            "scannedCount": scanned_count,
            "failedCount": failed_count,
            "candidateCount": len(candidates),
        },
        "candidates": [_candidate_payload(idx, item) for idx, item in enumerate(candidates, start=1)],
        "history": {
            code: _history_payload(history)
            for code, history in history_by_code.items()
        },
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
        "breakoutPct": round(item.breakout_pct * 100, 4),
        "volumeRatio": round(item.volume_ratio, 4),
        "volumeTrend": round(item.volume_trend, 4),
        "touches": item.resistance_touches,
        "clusterSize": item.resistance_cluster_size,
        "atrPct": round(item.atr_pct * 100, 4),
        "circMv": round(item.circ_mv, 2) if item.circ_mv > 0 else 0,
        "score": round(item.score, 4),
        "buyLow": round(item.buy_zone_low, 4),
        "buyHigh": round(item.buy_zone_high, 4),
        "stopLoss": round(item.stop_loss, 4),
        "tradeStopLoss": round(item.trade_stop_loss or item.stop_loss, 4),
        "structureStopLoss": round(item.structure_stop_loss or item.stop_loss, 4),
        "totalAssetPosition": item.total_asset_position,
        "strategyPosition": item.strategy_position,
        "maxLossAssetPct": item.max_loss_asset_pct,
        "hint": item.position_hint,
        "latestTradeDate": item.latest_trade_date.isoformat(),
    }


def _history_payload(history: pd.DataFrame) -> list[dict[str, Any]]:
    if history.empty:
        return []
    df = history.sort_values("date").tail(260).copy()
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
    .range-controls {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
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
    canvas {
      display: block;
      width: 100%;
      height: 510px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
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
      canvas { height: 420px; }
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
      <div class="stat"><span>入选数量</span><strong id="statCandidates"></strong></div>
      <div class="stat"><span>数据失败</span><strong id="statFailed"></strong></div>
    </div>
  </header>
  <main>
    <section class="table-panel">
      <div class="section-head">
        <h2>候选结果</h2>
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
              <th>阻力</th>
              <th>突破</th>
              <th>量比</th>
              <th>量趋势</th>
              <th>ATR%</th>
              <th>得分</th>
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
          <button type="button" data-range="60">60日</button>
          <button type="button" data-range="120" class="active">120日</button>
          <button type="button" data-range="250">250日</button>
        </div>
      </div>
      <div class="chart-box">
        <canvas id="klineCanvas" width="1200" height="560"></canvas>
      </div>
      <div class="detail-grid">
        <div class="detail"><span>买入区</span><strong id="buyZone"></strong></div>
        <div class="detail"><span>交易止损</span><strong id="tradeStopLoss"></strong></div>
        <div class="detail"><span>结构止损</span><strong id="structureStopLoss"></strong></div>
        <div class="detail"><span>阻力触达</span><strong id="touches"></strong></div>
        <div class="detail"><span>聚类大小</span><strong id="cluster"></strong></div>
        <div class="detail"><span>ATR%</span><strong id="atr"></strong></div>
        <div class="detail"><span>信号说明</span><strong id="signalReason"></strong></div>
        <div class="detail"><span>仓位口径</span><strong id="positionRule"></strong></div>
        <div class="detail"><span>选中K线</span><strong id="hoverInfo"></strong></div>
      </div>
    </section>
  </main>
  <script>
    const DASHBOARD = __DASHBOARD_DATA__;
    const state = {
      code: DASHBOARD.candidates[0]?.code || "",
      range: 120,
      hoverIndex: null
    };

    const fmt = (value, digits = 2) => {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
      return Number(value).toFixed(digits);
    };

    document.getElementById("statDate").textContent = DASHBOARD.meta.latestTradeDate;
    document.getElementById("statScanned").textContent = DASHBOARD.meta.scannedCount;
    document.getElementById("statCandidates").textContent = DASHBOARD.meta.candidateCount;
    document.getElementById("statFailed").textContent = DASHBOARD.meta.failedCount;

    function candidateByCode(code) {
      return DASHBOARD.candidates.find(item => item.code === code);
    }

    function renderRows() {
      const tbody = document.getElementById("candidateRows");
      tbody.textContent = "";
      if (!DASHBOARD.candidates.length) {
        const row = document.createElement("tr");
        const cell = document.createElement("td");
        cell.colSpan = 11;
        cell.className = "empty";
        cell.textContent = "今日没有符合条件的候选。";
        row.appendChild(cell);
        tbody.appendChild(row);
        return;
      }
      for (const item of DASHBOARD.candidates) {
        const row = document.createElement("tr");
        row.dataset.code = item.code;
        if (item.code === state.code) row.classList.add("selected");
        const cells = [
          item.rank,
          item.signalType,
          `${item.code} ${item.name}`,
          fmt(item.latestClose),
          item.circMv > 0 ? fmt(item.circMv, 1) : "-",
          fmt(item.resistance),
          `${fmt(item.breakoutPct)}%`,
          fmt(item.volumeRatio),
          fmt(item.volumeTrend),
          `${fmt(item.atrPct)}%`,
          fmt(item.score, 1)
        ];
        cells.forEach((value, idx) => {
          const cell = document.createElement("td");
          cell.textContent = value;
          if (idx === 2) cell.className = "name";
          if (idx === 6 && item.breakoutPct > 0) cell.classList.add("strong");
          if (idx === 10) cell.classList.add("score");
          row.appendChild(cell);
        });
        row.addEventListener("click", () => {
          state.code = item.code;
          state.hoverIndex = null;
          renderRows();
          renderChart();
        });
        tbody.appendChild(row);
      }
    }

    const canvas = document.getElementById("klineCanvas");
    const ctx = canvas.getContext("2d");

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

    function drawLine(y, color, label) {
      const width = canvas.clientWidth;
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(54, y);
      ctx.lineTo(width - 16, y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = color;
      ctx.font = "12px -apple-system, BlinkMacSystemFont, sans-serif";
      ctx.fillText(label, width - 92, y - 5);
      ctx.restore();
    }

    function renderChart() {
      setupCanvas();
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      ctx.clearRect(0, 0, width, height);

      const item = candidateByCode(state.code);
      if (!item) return;
      const raw = DASHBOARD.history[item.code] || [];
      const data = raw.slice(-state.range).filter(row =>
        row.open !== null && row.high !== null && row.low !== null && row.close !== null
      );

      document.getElementById("chartName").textContent = `${item.code} ${item.name}`;
      document.getElementById("chartMeta").textContent =
        `${item.signalType}类 / 收盘 ${fmt(item.latestClose)} / 市值 ${item.circMv > 0 ? fmt(item.circMv, 1) + '亿' : '-'} / 阻力 ${fmt(item.resistance)} / 突破 ${fmt(item.breakoutPct)}% / 得分 ${fmt(item.score, 1)} / ${item.hint}`;
      document.getElementById("buyZone").textContent = `${fmt(item.buyLow)} - ${fmt(item.buyHigh)}`;
      document.getElementById("tradeStopLoss").textContent = fmt(item.tradeStopLoss);
      document.getElementById("structureStopLoss").textContent = fmt(item.structureStopLoss);
      document.getElementById("touches").textContent = `${item.touches} 次`;
      document.getElementById("cluster").textContent = `${item.clusterSize || "-"} 根K线`;
      document.getElementById("atr").textContent = `${fmt(item.atrPct)}%`;
      document.getElementById("signalReason").textContent = item.signalReason || "-";
      document.getElementById("positionRule").textContent =
        `${item.totalAssetPosition} / 策略内 ${item.strategyPosition} / 最大亏损 ${item.maxLossAssetPct}`;

      if (!data.length) {
        ctx.fillStyle = "#667085";
        ctx.font = "15px -apple-system, BlinkMacSystemFont, sans-serif";
        ctx.fillText("没有可用K线数据", 32, 48);
        return;
      }

      const left = 54;
      const right = width - 16;
      const top = 22;
      const priceBottom = Math.max(240, height - 118);
      const volumeTop = priceBottom + 18;
      const volumeBottom = height - 26;
      const highs = data.map(row => row.high);
      const lows = data.map(row => row.low);
      highs.push(item.resistance, item.buyHigh, item.tradeStopLoss, item.structureStopLoss);
      lows.push(item.resistance, item.buyLow, item.tradeStopLoss, item.structureStopLoss);
      let maxPrice = Math.max(...highs);
      let minPrice = Math.min(...lows);
      const pad = Math.max((maxPrice - minPrice) * 0.08, maxPrice * 0.01, 0.5);
      maxPrice += pad;
      minPrice -= pad;
      const maxVolume = Math.max(...data.map(row => row.volume || 0), 1);
      const step = (right - left) / Math.max(data.length, 1);
      const candleWidth = Math.max(3, Math.min(12, step * 0.58));

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

      const buyY1 = yScale(item.buyHigh, minPrice, maxPrice, top, priceBottom);
      const buyY2 = yScale(item.buyLow, minPrice, maxPrice, top, priceBottom);
      ctx.fillStyle = "rgba(31, 111, 235, 0.08)";
      ctx.fillRect(left, Math.min(buyY1, buyY2), right - left, Math.abs(buyY2 - buyY1));
      drawLine(yScale(item.resistance, minPrice, maxPrice, top, priceBottom), "#1f6feb", "阻力");
      drawLine(yScale(item.tradeStopLoss, minPrice, maxPrice, top, priceBottom), "#b76e00", "交易止损");
      drawLine(yScale(item.structureStopLoss, minPrice, maxPrice, top, priceBottom), "#7654a6", "结构止损");

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
      ctx.fillText(last.date, Math.max(left, right - 82), height - 8);

      if (state.hoverIndex !== null && data[state.hoverIndex]) {
        const row = data[state.hoverIndex];
        const x = left + step * state.hoverIndex + step / 2;
        ctx.strokeStyle = "rgba(24, 32, 43, 0.38)";
        ctx.beginPath();
        ctx.moveTo(x, top);
        ctx.lineTo(x, volumeBottom);
        ctx.stroke();
        document.getElementById("hoverInfo").textContent =
          `${row.date} 开 ${fmt(row.open)} 高 ${fmt(row.high)} 低 ${fmt(row.low)} 收 ${fmt(row.close)}`;
      } else {
        document.getElementById("hoverInfo").textContent =
          `${last.date} 收 ${fmt(last.close)}`;
      }
    }

    canvas.addEventListener("mousemove", (event) => {
      const item = candidateByCode(state.code);
      const raw = item ? (DASHBOARD.history[item.code] || []) : [];
      const data = raw.slice(-state.range);
      if (!data.length) return;
      const rect = canvas.getBoundingClientRect();
      const left = 54;
      const right = canvas.clientWidth - 16;
      const step = (right - left) / Math.max(data.length, 1);
      const idx = Math.round((event.clientX - rect.left - left - step / 2) / step);
      state.hoverIndex = Math.max(0, Math.min(data.length - 1, idx));
      renderChart();
    });
    canvas.addEventListener("mouseleave", () => {
      state.hoverIndex = null;
      renderChart();
    });

    document.querySelectorAll("[data-range]").forEach(button => {
      button.addEventListener("click", () => {
        state.range = Number(button.dataset.range);
        document.querySelectorAll("[data-range]").forEach(item => item.classList.remove("active"));
        button.classList.add("active");
        state.hoverIndex = null;
        renderChart();
      });
    });

    window.addEventListener("resize", renderChart);
    renderRows();
    renderChart();
  </script>
</body>
</html>
"""
