from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .config import AppConfig
from .data import fetch_daily_basic, fetch_history, fetch_index_history, fetch_spot, filter_spot_universe, prepare_history_cache
from .html_report import write_html_dashboard
from .models import Candidate
from .scoring import evaluate_stock


@dataclass(frozen=True)
class ScanResult:
    candidates: list[Candidate]
    output_dir: Path
    csv_path: Path
    xlsx_path: Path
    markdown_path: Path
    html_path: Path
    scanned_count: int
    failed_count: int
    latest_trade_date: str


def run_scan(
    config: AppConfig,
    symbols: set[str] | None = None,
    limit: int | None = None,
    force_refresh: bool = False,
) -> ScanResult:
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    end_date = now.date()
    spot = fetch_spot()
    spot_trade_date = _latest_spot_trade_date(spot)
    if spot_trade_date:
        end_date = spot_trade_date
    start_date = end_date - timedelta(days=config.screener.history_days)
    output_dir = config.paths.output_dir / end_date.isoformat()
    output_dir.mkdir(parents=True, exist_ok=True)
    config.paths.cache_dir.mkdir(parents=True, exist_ok=True)

    # Check market regime before scanning.
    if config.screener.market_regime != "all" and not symbols:
        blocked, reason = _check_market_regime(
            market_index=config.screener.market_index,
            regime=config.screener.market_regime,
            start_date=start_date,
            end_date=end_date,
            cache_dir=config.paths.cache_dir,
        )
        if blocked:
            return _write_blocked_result(output_dir, reason, end_date.isoformat())

    universe = filter_spot_universe(
        spot=spot,
        allowed_prefixes=config.screener.allowed_prefixes,
        exclude_name_keywords=config.screener.exclude_name_keywords,
        min_amount=config.screener.min_amount,
        min_price=config.screener.min_price,
        symbols=symbols,
    )
    if limit:
        universe = universe.head(limit)

    prepare_history_cache(
        symbols={str(row["code"]) for _, row in universe.iterrows()},
        start_date=start_date,
        end_date=end_date,
        cache_dir=config.paths.cache_dir,
        force_refresh=force_refresh,
    )

    candidates: list[Candidate] = []
    history_by_code: dict[str, pd.DataFrame] = {}
    failures: list[tuple[str, str, str]] = []
    max_workers = max(1, config.screener.max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _check_one,
                row["code"],
                row["name"],
                start_date,
                end_date,
                config,
                force_refresh,
            ): (row["code"], row["name"])
            for _, row in universe.iterrows()
        }
        for future in as_completed(futures):
            code, name = futures[future]
            try:
                checked = future.result()
                if checked:
                    candidate, history = checked
                    candidates.append(candidate)
                    history_by_code[candidate.code] = history
            except Exception as exc:  # pragma: no cover - external data source failures vary
                failures.append((code, name, str(exc)))

    candidates = sorted(candidates, key=lambda item: item.score, reverse=True)[: config.screener.top_n]
    circ_mv_map: dict[str, float] = {}
    if candidates:
        trade_date_str = candidates[0].latest_trade_date.strftime("%Y%m%d")
        try:
            circ_mv_map = fetch_daily_basic(trade_date_str, config.paths.cache_dir)
        except Exception:  # pragma: no cover — external data source
            pass
    if circ_mv_map:
        candidates = [item.with_circ_mv(circ_mv_map.get(item.code, 0.0)) for item in candidates]
    history_by_code = {item.code: history_by_code[item.code] for item in candidates if item.code in history_by_code}
    latest_trade_date = max((item.latest_trade_date.isoformat() for item in candidates), default=end_date.isoformat())
    csv_path, xlsx_path, markdown_path, html_path = write_outputs(
        candidates=candidates,
        history_by_code=history_by_code,
        output_dir=output_dir,
        scanned_count=len(universe),
        failed_count=len(failures),
        latest_trade_date=latest_trade_date,
    )
    _write_failures(output_dir / "failures.csv", failures)
    return ScanResult(
        candidates=candidates,
        output_dir=output_dir,
        csv_path=csv_path,
        xlsx_path=xlsx_path,
        markdown_path=markdown_path,
        html_path=html_path,
        scanned_count=len(universe),
        failed_count=len(failures),
        latest_trade_date=latest_trade_date,
    )


def write_outputs(
    candidates: list[Candidate],
    history_by_code: dict[str, pd.DataFrame] | None,
    output_dir: Path,
    scanned_count: int,
    failed_count: int,
    latest_trade_date: str,
) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [candidate.to_chinese_dict() for candidate in candidates]
    df = pd.DataFrame(rows)
    csv_path = output_dir / "breakout_candidates.csv"
    xlsx_path = output_dir / "breakout_candidates.xlsx"
    markdown_path = output_dir / "breakout_report.md"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="突破候选")
    markdown_path.write_text(
        render_markdown_report(candidates, scanned_count, failed_count, latest_trade_date),
        encoding="utf-8",
    )
    html_path = write_html_dashboard(
        candidates=candidates,
        history_by_code=history_by_code or {},
        output_dir=output_dir,
        scanned_count=scanned_count,
        failed_count=failed_count,
        latest_trade_date=latest_trade_date,
    )
    return csv_path, xlsx_path, markdown_path, html_path


def render_markdown_report(
    candidates: list[Candidate],
    scanned_count: int,
    failed_count: int,
    latest_trade_date: str,
) -> str:
    lines = [
        f"# A股突破选股日报 {latest_trade_date}",
        "",
        f"- 扫描股票数: {scanned_count}",
        f"- 数据失败数: {failed_count}",
        f"- 入选数量: {len(candidates)}",
        "",
        "说明: 这是规则筛选和风险观察清单，不是投资建议。请结合大盘环境、行业事件和个人仓位做二次判断。",
        "",
    ]
    if not candidates:
        lines.append("今日没有符合突破条件的候选。")
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "|排名|类型|代码|名称|收盘|市值(亿)|阻力|突破%|量能比|量趋势|触达|聚类|ATR%|得分|买入区|交易止损|结构止损|总资产仓位|策略内仓位|最大亏损|提示|",
            "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|---|---|---|",
        ]
    )
    for idx, item in enumerate(candidates, start=1):
        lines.append(
            "|{rank}|{signal_type}|{code}|{name}|{close:.2f}|{mv}|{resistance:.2f}|{breakout:.2f}|"
            "{vr:.2f}|{vt:.2f}|{touches}|{cluster}|{atr:.1f}|{score:.1f}|"
            "{buy_low:.2f}-{buy_high:.2f}|{trade_stop:.2f}|{structure_stop:.2f}|"
            "{total_position}|{strategy_position}|{max_loss}|{hint}|".format(
                rank=idx,
                signal_type=item.signal_type,
                code=item.code,
                name=item.name,
                close=item.latest_close,
                mv=f"{item.circ_mv:.1f}" if item.circ_mv > 0 else "-",
                resistance=item.resistance,
                breakout=item.breakout_pct * 100,
                vr=item.volume_ratio,
                vt=item.volume_trend,
                touches=item.resistance_touches,
                cluster=item.resistance_cluster_size,
                atr=item.atr_pct * 100,
                score=item.score,
                buy_low=item.buy_zone_low,
                buy_high=item.buy_zone_high,
                trade_stop=item.trade_stop_loss or item.stop_loss,
                structure_stop=item.structure_stop_loss or item.stop_loss,
                total_position=item.total_asset_position,
                strategy_position=item.strategy_position,
                max_loss=item.max_loss_asset_pct,
                hint=item.position_hint,
            )
        )
    lines.append("")
    return "\n".join(lines)


def _check_one(
    code: str,
    name: str,
    start_date,
    end_date,
    config: AppConfig,
    force_refresh: bool,
) -> tuple[Candidate, pd.DataFrame] | None:
    history = fetch_history(
        symbol=code,
        start_date=start_date,
        end_date=end_date,
        cache_dir=config.paths.cache_dir,
        force_refresh=force_refresh,
    )
    candidate = evaluate_stock(code=code, name=name, history=history, params=config.screener)
    if not candidate:
        return None
    return candidate, history


def _write_failures(path: Path, failures: list[tuple[str, str, str]]) -> None:
    if not failures:
        path.write_text("代码,名称,错误\n", encoding="utf-8-sig")
        return
    df = pd.DataFrame(failures, columns=["代码", "名称", "错误"])
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _latest_spot_trade_date(spot: pd.DataFrame):
    if "trade_date" not in spot.columns or spot["trade_date"].dropna().empty:
        return None
    value = str(spot["trade_date"].dropna().max())
    parsed = pd.to_datetime(value, format="%Y%m%d", errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _check_market_regime(
    market_index: str,
    regime: str,
    start_date,
    end_date,
    cache_dir: Path,
) -> tuple[bool, str]:
    """Return (blocked, reason). blocked=True means stop scanning."""
    if regime == "all":
        return False, ""
    try:
        history = fetch_index_history(
            index_code=market_index,
            start_date=start_date,
            end_date=end_date,
            cache_dir=cache_dir,
        )
        daily = history.sort_values("date").reset_index(drop=True)
        if len(daily) < 120:
            return True, "大盘数据不足，暂停选股"
        daily["ma60"] = daily["close"].rolling(60).mean()
        latest = daily.iloc[-1]
        close = float(latest["close"])
        ma60 = float(latest["ma60"])
        if not np.isfinite(ma60) or ma60 <= 0:
            return True, "大盘 MA60 计算失败，暂停选股"
        index_name = {"000300": "沪深300", "000001": "上证指数", "399001": "深证成指"}.get(
            market_index.zfill(6), market_index
        )
        if regime == "bull_only" and close < ma60:
            return True, f"大盘弱势：{index_name} 收盘 {close:.2f} < MA60 {ma60:.2f}，暂停选股"
        return False, f"大盘OK：{index_name} 收盘 {close:.2f} >= MA60 {ma60:.2f}"
    except Exception as exc:
        return True, f"大盘数据获取失败({exc})，暂停选股"


def _write_blocked_result(output_dir: Path, reason: str, latest_trade_date: str) -> ScanResult:
    markdown_path = output_dir / "breakout_report.md"
    csv_path = output_dir / "breakout_candidates.csv"
    xlsx_path = output_dir / "breakout_candidates.xlsx"
    html_path = output_dir / "breakout_dashboard.html"

    markdown_path.write_text(
        f"# A股突破选股日报 {latest_trade_date}\n\n"
        f"**{reason}**\n\n"
        "说明: 大盘环境过滤已启用，当前不满足选股条件。\n",
        encoding="utf-8",
    )
    cols = ["代码", "名称", "最新收盘", "信号类型", "信号说明", "阻力位", "突破幅度%", "量能比", "量能趋势",
            "阻力触达次数", "阻力聚类大小", "月线跨度%", "ATR%", "MA10", "MA20", "得分", "首次阻力日期",
            "最近阻力日期", "最新交易日", "建议买入区", "交易止损", "结构止损", "流通市值(亿)",
            "总资产建议仓位", "策略内建议仓位", "最大允许亏损", "仓位提示"]
    pd.DataFrame(columns=cols).to_csv(csv_path, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame(columns=cols).to_excel(writer, index=False, sheet_name="突破候选")
    html_path.write_text(
        f"<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>选股日报</title></head>"
        f"<body><h1>A股突破选股日报 {latest_trade_date}</h1><p>{reason}</p></body></html>",
        encoding="utf-8",
    )
    return ScanResult(
        candidates=[],
        output_dir=output_dir,
        csv_path=csv_path,
        xlsx_path=xlsx_path,
        markdown_path=markdown_path,
        html_path=html_path,
        scanned_count=0,
        failed_count=0,
        latest_trade_date=latest_trade_date,
    )
