from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .config import AppConfig
from .ai_analysis import generate_ai_analysis
from .data import (
    fetch_daily_basic,
    fetch_financial_metrics,
    fetch_history,
    fetch_index_history,
    fetch_spot,
    fetch_stock_tags,
    fetch_trade_calendar,
    filter_spot_universe,
    is_last_trade_day_of_week,
    prepare_history_cache,
)
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
    ai_analysis_path: Path | None
    scanned_count: int
    failed_count: int
    latest_trade_date: str
    all_candidate_count: int = 0
    pool_counts: dict[str, int] = field(default_factory=dict)


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
    calendar = fetch_trade_calendar(
        start_date=start_date - timedelta(days=14),
        end_date=end_date + timedelta(days=14),
        cache_dir=config.paths.cache_dir,
    )

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
        max_workers=config.screener.history_cache_workers,
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
                calendar,
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

    all_candidates = _sort_candidates(candidates)
    pools = _split_candidate_pools(all_candidates)
    pool_counts = {key: len(items) for key, items in pools.items()}
    candidates = _select_display_candidates(pools, config.screener.top_n)
    circ_mv_map: dict[str, float] = {}
    if candidates:
        trade_date_str = candidates[0].latest_trade_date.strftime("%Y%m%d")
        try:
            circ_mv_map = fetch_daily_basic(trade_date_str, config.paths.cache_dir)
        except Exception:  # pragma: no cover — external data source
            pass
    if circ_mv_map:
        candidates = [item.with_circ_mv(circ_mv_map.get(item.code, 0.0)) for item in candidates]
    tag_map = _fetch_tags_for_candidates(candidates, config, force_refresh=force_refresh)
    if tag_map:
        candidates = [item.with_tags(tag_map.get(item.code, ())) for item in candidates]
    financial_map = _fetch_financials_for_candidates(candidates, config, force_refresh=force_refresh)
    if financial_map:
        candidates = [item.with_financial_metrics(financial_map.get(item.code, {})) for item in candidates]
    history_by_code = {item.code: history_by_code[item.code] for item in candidates if item.code in history_by_code}
    latest_trade_date = max((item.latest_trade_date.isoformat() for item in candidates), default=end_date.isoformat())
    ai_analysis = generate_ai_analysis(
        config=config.ai_analysis,
        candidates=candidates,
        scanned_count=len(universe),
        failed_count=len(failures),
        latest_trade_date=latest_trade_date,
    )
    csv_path, xlsx_path, markdown_path, html_path = write_outputs(
        candidates=candidates,
        history_by_code=history_by_code,
        output_dir=output_dir,
        scanned_count=len(universe),
        failed_count=len(failures),
        latest_trade_date=latest_trade_date,
        ai_analysis=ai_analysis,
        pools=pools,
        pool_counts=pool_counts,
    )
    ai_analysis_path = output_dir / "ai_analysis.md" if ai_analysis else None
    _write_failures(output_dir / "failures.csv", failures)
    return ScanResult(
        candidates=candidates,
        output_dir=output_dir,
        csv_path=csv_path,
        xlsx_path=xlsx_path,
        markdown_path=markdown_path,
        html_path=html_path,
        ai_analysis_path=ai_analysis_path,
        scanned_count=len(universe),
        failed_count=len(failures),
        latest_trade_date=latest_trade_date,
        all_candidate_count=len(all_candidates),
        pool_counts=pool_counts,
    )


def write_outputs(
    candidates: list[Candidate],
    history_by_code: dict[str, pd.DataFrame] | None,
    output_dir: Path,
    scanned_count: int,
    failed_count: int,
    latest_trade_date: str,
    ai_analysis: str = "",
    pools: dict[str, list[Candidate]] | None = None,
    pool_counts: dict[str, int] | None = None,
) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [candidate.to_chinese_dict() for candidate in candidates]
    df = pd.DataFrame(rows)
    csv_path = output_dir / "breakout_candidates.csv"
    xlsx_path = output_dir / "breakout_candidates.xlsx"
    markdown_path = output_dir / "breakout_report.md"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="展示候选")
        for pool_name, items in (pools or {}).items():
            pd.DataFrame([item.to_chinese_dict() for item in items]).to_excel(
                writer,
                index=False,
                sheet_name=f"{pool_name}类"[:31],
            )
    for pool_name, items in (pools or {}).items():
        pd.DataFrame([item.to_chinese_dict() for item in items]).to_csv(
            output_dir / f"breakout_{pool_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )
    markdown = render_markdown_report(candidates, scanned_count, failed_count, latest_trade_date, pool_counts=pool_counts)
    if ai_analysis:
        markdown += "\n## AI选股分析员\n\n" + ai_analysis.strip() + "\n"
        (output_dir / "ai_analysis.md").write_text(ai_analysis.strip() + "\n", encoding="utf-8")
    markdown_path.write_text(markdown, encoding="utf-8")
    html_path = write_html_dashboard(
        candidates=candidates,
        history_by_code=history_by_code or {},
        output_dir=output_dir,
        scanned_count=scanned_count,
        failed_count=failed_count,
        latest_trade_date=latest_trade_date,
        ai_analysis=ai_analysis,
        pool_counts=pool_counts,
    )
    return csv_path, xlsx_path, markdown_path, html_path


def render_markdown_report(
    candidates: list[Candidate],
    scanned_count: int,
    failed_count: int,
    latest_trade_date: str,
    pool_counts: dict[str, int] | None = None,
) -> str:
    pool_counts = pool_counts or {}
    all_candidate_count = sum(pool_counts.values()) if pool_counts else len(candidates)
    lines = [
        f"# A股突破选股日报 {latest_trade_date}",
        "",
        f"- 扫描股票数: {scanned_count}",
        f"- 数据失败数: {failed_count}",
        f"- 全量候选总数: {all_candidate_count}",
        f"- 展示候选数: {len(candidates)}",
        f"- A类周线确认: {pool_counts.get('A', 0)}",
        f"- B类日线预警: {pool_counts.get('B', 0)}",
        f"- C1强趋势观察: {pool_counts.get('C1', 0)}",
        f"- C2突破不追: {pool_counts.get('C2', 0)}",
        f"- D类排除/突破不足: {pool_counts.get('D', 0)}",
        f"- 今日可交易观察: {pool_counts.get('A', 0)}",
        "",
        "说明: 这是规则筛选和风险观察清单，不是投资建议。请结合大盘环境、行业事件和个人仓位做二次判断。",
        "",
    ]
    if not candidates:
        lines.append("今日没有符合突破条件的候选。")
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "|排名|类型|代码|名称|标签|收盘|市值(亿)|压力区|突破%|量能比|量趋势|触达|跨度周|ATR%|技术分|成长分|营收同比%|净利同比%|ROE%|买入区|交易止损|结论|",
            "|---:|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|",
        ]
    )
    for idx, item in enumerate(candidates, start=1):
        lines.append(
            "|{rank}|{signal}|{code}|{name}|{tags}|{close:.2f}|{mv}|{zone_low:.2f}-{zone_upper:.2f}|{breakout:.2f}|"
            "{vr:.2f}|{vt:.2f}|{touches}|{span}|{atr:.1f}|{score:.1f}|{growth}|{revenue}|{profit}|{roe}|"
            "{buy_low:.2f}-{buy_high:.2f}|{stop:.2f}|{action}|".format(
                rank=idx,
                signal=item.signal_type,
                code=item.code,
                name=item.name,
                tags="、".join(item.tags) if item.tags else "-",
                close=item.latest_close,
                mv=f"{item.circ_mv:.1f}" if item.circ_mv > 0 else "-",
                zone_low=item.zone_low or item.resistance,
                zone_upper=item.zone_upper or item.resistance,
                breakout=item.breakout_pct * 100,
                vr=item.volume_ratio,
                vt=item.volume_trend,
                touches=item.resistance_touches,
                span=item.span_weeks,
                atr=item.atr_pct * 100,
                score=item.score,
                growth=f"{item.growth_score:.1f}" if item.growth_score > 0 else "-",
                revenue=_fmt_optional(item.revenue_yoy),
                profit=_fmt_optional(item.profit_yoy),
                roe=_fmt_optional(item.roe),
                buy_low=item.buy_zone_low,
                buy_high=item.buy_zone_high,
                stop=item.trade_stop_loss or item.stop_loss,
                action=item.trade_action or item.position_hint,
            )
        )
    lines.append("")
    return "\n".join(lines)


def _fetch_tags_for_candidates(
    candidates: list[Candidate],
    config: AppConfig,
    force_refresh: bool,
) -> dict[str, tuple[str, ...]]:
    if not candidates:
        return {}
    max_workers = min(4, max(1, config.screener.max_workers))
    tag_map: dict[str, tuple[str, ...]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_stock_tags, item.code, config.paths.cache_dir, force_refresh): item.code
            for item in candidates
        }
        for future in as_completed(futures):
            code = futures[future]
            try:
                tag_map[code] = future.result()
            except Exception:  # pragma: no cover - tags are non-critical external metadata
                tag_map[code] = ()
    return tag_map


def _fetch_financials_for_candidates(
    candidates: list[Candidate],
    config: AppConfig,
    force_refresh: bool,
) -> dict[str, dict[str, object]]:
    if not candidates:
        return {}
    max_workers = min(4, max(1, config.screener.max_workers))
    financial_map: dict[str, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_financial_metrics, item.code, config.paths.cache_dir, force_refresh): item.code
            for item in candidates
        }
        for future in as_completed(futures):
            code = futures[future]
            try:
                financial_map[code] = future.result()
            except Exception:  # pragma: no cover - financial metadata is non-critical
                financial_map[code] = {}
    return financial_map


def _fmt_optional(value: float | None) -> str:
    return f"{value:.1f}" if value is not None else "-"


def _sort_candidates(candidates: list[Candidate]) -> list[Candidate]:
    signal_order = {"A": 0, "B": 1, "C1": 2, "C2": 3, "C": 3, "D": 4}
    return sorted(
        candidates,
        key=lambda item: (
            signal_order.get(item.signal_type, 9),
            -item.score,
            -item.volume_ratio,
            item.code,
        ),
    )


def _split_candidate_pools(candidates: list[Candidate]) -> dict[str, list[Candidate]]:
    pools: dict[str, list[Candidate]] = {"A": [], "B": [], "C1": [], "C2": [], "D": []}
    for item in candidates:
        key = item.signal_type if item.signal_type in pools else "D"
        if item.signal_type == "C":
            key = "C2"
        pools[key].append(item)
    return {key: _sort_candidates(items) for key, items in pools.items()}


def _select_display_candidates(pools: dict[str, list[Candidate]], top_n: int) -> list[Candidate]:
    selected: list[Candidate] = []
    selected.extend(pools.get("A", [])[:10])
    selected.extend(pools.get("B", [])[: max(10, top_n)])
    selected.extend(pools.get("C1", [])[:10])
    selected.extend(pools.get("C2", [])[:5])
    selected.extend(pools.get("D", [])[:5])
    return selected[: max(top_n, 30)]


def _check_one(
    code: str,
    name: str,
    start_date,
    end_date,
    config: AppConfig,
    force_refresh: bool,
    calendar: pd.DataFrame,
) -> tuple[Candidate, pd.DataFrame] | None:
    history = fetch_history(
        symbol=code,
        start_date=start_date,
        end_date=end_date,
        cache_dir=config.paths.cache_dir,
        force_refresh=force_refresh,
        allow_truncated_start=True,
    )
    if history.empty:
        return None
    latest_trade_date = pd.Timestamp(history.sort_values("date").iloc[-1]["date"]).date()
    week_confirmed = is_last_trade_day_of_week(latest_trade_date, calendar)
    candidate = evaluate_stock(
        code=code,
        name=name,
        history=history,
        params=config.screener,
        is_week_confirmed=week_confirmed,
    )
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
    cols = ["代码", "名称", "信号类型", "信号说明", "最新收盘", "压力区下沿", "压力区中枢", "压力区上沿",
            "阻力位", "突破幅度%", "量能比", "量能来源", "成交/量能倍数", "量能趋势", "阻力触达次数", "阻力聚类大小", "压力跨度周",
            "月线跨度%", "ATR%", "MA10", "MA20", "得分", "首次阻力日期", "最近阻力日期",
            "最新交易日", "建议买入区", "止损位", "交易止损", "结构止损", "流通市值(亿)",
            "总资产建议仓位", "策略内建议仓位", "最大允许亏损", "交易结论", "仓位提示", "题材标签",
            "财务期", "成长分", "营收同比%", "净利同比%", "ROE%", "毛利率%", "资产负债率%"]
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
        ai_analysis_path=None,
        scanned_count=0,
        failed_count=0,
        latest_trade_date=latest_trade_date,
    )
