from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import AppConfig
from .data import _normalize_history
from .models import Candidate
from .scoring import evaluate_stock


@dataclass(frozen=True)
class BacktestResult:
    output_dir: Path
    trades_path: Path
    summary_path: Path
    html_path: Path
    signal_days: int
    stock_count: int
    trade_count: int


def run_backtest(
    config: AppConfig,
    days: int = 252,
    top_ns: tuple[int, ...] = (10, 20, 30),
    holding_days: tuple[int, ...] = (5, 10, 20),
    symbols: set[str] | None = None,
) -> BacktestResult:
    histories = _load_histories(config.paths.cache_dir, symbols=symbols)
    if not histories:
        raise RuntimeError(f"没有可回测的历史缓存: {config.paths.cache_dir / 'hist'}")

    names = _load_names_from_latest_daily(config.paths.cache_dir)
    trading_dates = _common_trading_dates(histories)
    if len(trading_dates) < config.screener.min_history_rows + max(holding_days) + 2:
        raise RuntimeError("历史数据太少，无法回测")

    max_holding = max(holding_days)
    signal_dates = trading_dates[config.screener.min_history_rows : -max_holding - 1]
    signal_dates = signal_dates[-days:]
    max_top_n = max(top_ns)

    trades: list[dict[str, Any]] = []
    total = len(signal_dates)
    for idx, signal_ts in enumerate(signal_dates, start=1):
        daily_candidates: list[tuple[Candidate, pd.DataFrame, int]] = []
        for code, history in histories.items():
            pos = history["date"].searchsorted(signal_ts, side="right") - 1
            if pos < config.screener.min_history_rows:
                continue
            if pos + max_holding + 1 >= len(history):
                continue
            past = history.iloc[: pos + 1]
            latest = past.iloc[-1]
            if float(latest.get("amount", 0) or 0) < config.screener.min_amount:
                continue
            if float(latest.get("close", 0) or 0) < config.screener.min_price:
                continue
            candidate = evaluate_stock(
                code=code,
                name=names.get(code, code),
                history=past,
                params=config.screener,
            )
            if candidate:
                daily_candidates.append((candidate, history, pos))

        daily_candidates.sort(key=lambda item: item[0].score, reverse=True)
        for rank, (candidate, history, pos) in enumerate(daily_candidates[:max_top_n], start=1):
            entry_row = history.iloc[pos + 1]
            entry_price = float(entry_row["open"])
            if entry_price <= 0:
                continue
            for holding in holding_days:
                exit_row = history.iloc[pos + holding]
                exit_price = float(exit_row["close"])
                ret = exit_price / entry_price - 1
                trades.append(
                    {
                        "signal_date": signal_ts.date().isoformat(),
                        "entry_date": pd.Timestamp(entry_row["date"]).date().isoformat(),
                        "exit_date": pd.Timestamp(exit_row["date"]).date().isoformat(),
                        "holding_days": holding,
                        "rank": rank,
                        "code": candidate.code,
                        "name": candidate.name,
                        "score": round(candidate.score, 4),
                        "breakout_pct": round(candidate.breakout_pct * 100, 4),
                        "volume_ratio": round(candidate.volume_ratio, 4),
                        "entry_price": round(entry_price, 4),
                        "exit_price": round(exit_price, 4),
                        "return_pct": round(ret * 100, 4),
                    }
                )

        if idx == 1 or idx % 20 == 0 or idx == total:
            print(f"回测进度: {idx}/{total} 个信号日", flush=True)

    output_dir = config.paths.output_dir / "backtest" / date.today().isoformat()
    output_dir.mkdir(parents=True, exist_ok=True)
    trades_df = pd.DataFrame(trades)
    summary_df = _summarize(trades_df, top_ns=top_ns, holding_days=holding_days)

    trades_path = output_dir / "backtest_trades.csv"
    summary_path = output_dir / "backtest_summary.csv"
    html_path = output_dir / "backtest_report.html"
    trades_df.to_csv(trades_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    html_path.write_text(_render_html(summary_df, trades_df, total, len(histories)), encoding="utf-8")

    return BacktestResult(
        output_dir=output_dir,
        trades_path=trades_path,
        summary_path=summary_path,
        html_path=html_path,
        signal_days=total,
        stock_count=len(histories),
        trade_count=len(trades),
    )


def _load_histories(cache_dir: Path, symbols: set[str] | None = None) -> dict[str, pd.DataFrame]:
    hist_dir = cache_dir / "hist"
    normalized_symbols = {item.zfill(6) for item in symbols} if symbols else None
    histories: dict[str, pd.DataFrame] = {}
    for path in sorted(hist_dir.glob("*.csv")):
        code = path.stem.zfill(6)
        if normalized_symbols and code not in normalized_symbols:
            continue
        try:
            history = _normalize_history(pd.read_csv(path))
        except Exception:
            continue
        if history.empty:
            continue
        histories[code] = history.sort_values("date").reset_index(drop=True)
    return histories


def _load_names_from_latest_daily(cache_dir: Path) -> dict[str, str]:
    daily_dir = cache_dir / "tushare_daily"
    paths = sorted(daily_dir.glob("*.csv"))
    if not paths:
        return {}
    try:
        latest = pd.read_csv(paths[-1])
    except Exception:
        return {}
    if "code" not in latest.columns:
        if "ts_code" not in latest.columns:
            return {}
        latest["code"] = latest["ts_code"].astype(str).str.extract(r"(\d{6})", expand=False).str.zfill(6)
    # Tushare daily cache does not carry stock names. Keep hook for future enriched caches.
    return {str(row["code"]).zfill(6): str(row.get("name", row["code"])) for _, row in latest.iterrows()}


def _common_trading_dates(histories: dict[str, pd.DataFrame]) -> list[pd.Timestamp]:
    counts: dict[pd.Timestamp, int] = {}
    for history in histories.values():
        for ts in history["date"].drop_duplicates():
            key = pd.Timestamp(ts).normalize()
            counts[key] = counts.get(key, 0) + 1
    threshold = min(len(histories), max(1, int(len(histories) * 0.5)))
    return sorted(ts for ts, count in counts.items() if count >= threshold)


def _summarize(
    trades_df: pd.DataFrame,
    top_ns: tuple[int, ...],
    holding_days: tuple[int, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if trades_df.empty:
        return pd.DataFrame()
    for top_n in top_ns:
        for holding in holding_days:
            subset = trades_df[(trades_df["rank"] <= top_n) & (trades_df["holding_days"] == holding)].copy()
            if subset.empty:
                continue
            returns = subset["return_pct"].astype(float)
            grouped = subset.groupby("signal_date")["return_pct"].mean().sort_index() / 100
            equity = (1 + grouped).cumprod()
            max_drawdown = _max_drawdown(equity)
            rows.append(
                {
                    "top_n": top_n,
                    "holding_days": holding,
                    "trades": int(len(subset)),
                    "signal_days": int(subset["signal_date"].nunique()),
                    "win_rate_pct": round(float((returns > 0).mean() * 100), 2),
                    "avg_return_pct": round(float(returns.mean()), 3),
                    "median_return_pct": round(float(returns.median()), 3),
                    "best_return_pct": round(float(returns.max()), 3),
                    "worst_return_pct": round(float(returns.min()), 3),
                    "avg_daily_bucket_pct": round(float(grouped.mean() * 100), 3),
                    "compounded_bucket_return_pct": round(float((equity.iloc[-1] - 1) * 100), 3),
                    "max_drawdown_pct": round(float(max_drawdown * 100), 3),
                }
            )
    return pd.DataFrame(rows).sort_values(["holding_days", "top_n"]).reset_index(drop=True)


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    drawdown = equity / peak - 1
    return abs(float(drawdown.min()))


def _render_html(summary: pd.DataFrame, trades: pd.DataFrame, signal_days: int, stock_count: int) -> str:
    summary_rows = summary.to_dict(orient="records")
    top_trades = trades.sort_values("return_pct", ascending=False).head(20).to_dict(orient="records") if not trades.empty else []
    worst_trades = trades.sort_values("return_pct", ascending=True).head(20).to_dict(orient="records") if not trades.empty else []
    payload = {
        "summary": summary_rows,
        "topTrades": top_trades,
        "worstTrades": worst_trades,
        "meta": {
            "signalDays": signal_days,
            "stockCount": stock_count,
            "tradeCount": int(len(trades)),
        },
    }
    return HTML_TEMPLATE.replace("__BACKTEST_DATA__", json.dumps(payload, ensure_ascii=False))


HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>A股突破策略粗回测</title>
  <style>
    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f7f9; color: #18202b; }
    header { padding: 24px 28px 14px; background: #fff; border-bottom: 1px solid #d9dee8; }
    h1 { margin: 0 0 12px; font-size: 24px; letter-spacing: 0; }
    .stats { display: flex; flex-wrap: wrap; gap: 10px; }
    .stat { min-width: 132px; padding: 10px 12px; border: 1px solid #d9dee8; border-radius: 8px; background: #fbfcfe; }
    .stat span { display: block; color: #667085; font-size: 12px; }
    .stat strong { display: block; margin-top: 4px; font-size: 18px; }
    main { padding: 16px; display: grid; gap: 16px; }
    section { background: #fff; border: 1px solid #d9dee8; border-radius: 8px; box-shadow: 0 10px 24px rgba(24,32,43,.08); overflow: hidden; }
    h2 { margin: 0; padding: 14px 16px; font-size: 16px; border-bottom: 1px solid #d9dee8; letter-spacing: 0; }
    .note { padding: 12px 16px; color: #667085; font-size: 13px; line-height: 1.6; border-bottom: 1px solid #edf0f5; }
    .wrap { overflow: auto; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 9px 10px; border-bottom: 1px solid #edf0f5; text-align: right; white-space: nowrap; }
    th { background: #f9fafc; color: #667085; font-size: 12px; position: sticky; top: 0; }
    td.name, th.name { text-align: left; }
    .pos { color: #d33f49; font-weight: 700; }
    .neg { color: #16835f; font-weight: 700; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } header { padding: 18px 16px 12px; } main { padding: 12px; } }
  </style>
</head>
<body>
  <header>
    <h1>A股突破策略粗回测</h1>
    <div class="stats">
      <div class="stat"><span>信号日</span><strong id="signalDays"></strong></div>
      <div class="stat"><span>股票数</span><strong id="stockCount"></strong></div>
      <div class="stat"><span>模拟交易</span><strong id="tradeCount"></strong></div>
    </div>
  </header>
  <main>
    <section>
      <h2>参数组合表现</h2>
      <div class="note">口径：信号日收盘后选股，下一交易日开盘买入，持有 N 个交易日后收盘卖出；每个信号日内按 TopN 等权平均，仅用于粗筛策略有效性。</div>
      <div class="wrap"><table id="summaryTable"></table></div>
    </section>
    <div class="grid">
      <section>
        <h2>收益最高交易</h2>
        <div class="wrap"><table id="topTrades"></table></div>
      </section>
      <section>
        <h2>亏损最大交易</h2>
        <div class="wrap"><table id="worstTrades"></table></div>
      </section>
    </div>
  </main>
  <script>
    const DATA = __BACKTEST_DATA__;
    document.getElementById("signalDays").textContent = DATA.meta.signalDays;
    document.getElementById("stockCount").textContent = DATA.meta.stockCount;
    document.getElementById("tradeCount").textContent = DATA.meta.tradeCount;
    function fmt(value, digits = 2) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
      return Number(value).toFixed(digits);
    }
    function cell(value, cls = "") {
      const td = document.createElement("td");
      td.textContent = value;
      if (cls) td.className = cls;
      return td;
    }
    function renderSummary(rows) {
      const table = document.getElementById("summaryTable");
      table.innerHTML = "<thead><tr><th>TopN</th><th>持有</th><th>交易数</th><th>胜率</th><th>平均收益</th><th>中位数</th><th>最好</th><th>最差</th><th>按日复利</th><th>最大回撤</th></tr></thead>";
      const body = document.createElement("tbody");
      rows.forEach(row => {
        const tr = document.createElement("tr");
        tr.append(
          cell(row.top_n),
          cell(`${row.holding_days}日`),
          cell(row.trades),
          cell(`${fmt(row.win_rate_pct)}%`),
          signed(row.avg_return_pct),
          signed(row.median_return_pct),
          signed(row.best_return_pct),
          signed(row.worst_return_pct),
          signed(row.compounded_bucket_return_pct),
          cell(`${fmt(row.max_drawdown_pct)}%`, "neg")
        );
        body.appendChild(tr);
      });
      table.appendChild(body);
    }
    function signed(value) {
      return cell(`${fmt(value, 3)}%`, Number(value) >= 0 ? "pos" : "neg");
    }
    function renderTrades(id, rows) {
      const table = document.getElementById(id);
      table.innerHTML = "<thead><tr><th>信号日</th><th class='name'>股票</th><th>持有</th><th>排名</th><th>得分</th><th>买入</th><th>卖出</th><th>收益</th></tr></thead>";
      const body = document.createElement("tbody");
      rows.forEach(row => {
        const tr = document.createElement("tr");
        tr.append(
          cell(row.signal_date),
          cell(`${row.code} ${row.name}`, "name"),
          cell(`${row.holding_days}日`),
          cell(row.rank),
          cell(fmt(row.score, 1)),
          cell(fmt(row.entry_price)),
          cell(fmt(row.exit_price)),
          signed(row.return_pct)
        );
        body.appendChild(tr);
      });
      table.appendChild(body);
    }
    renderSummary(DATA.summary);
    renderTrades("topTrades", DATA.topTrades);
    renderTrades("worstTrades", DATA.worstTrades);
  </script>
</body>
</html>
"""
