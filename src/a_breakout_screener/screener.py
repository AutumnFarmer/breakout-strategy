from __future__ import annotations

from collections import Counter
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
    run_time: str = ""
    full_scan: bool = True


def run_scan(
    config: AppConfig,
    symbols: set[str] | None = None,
    limit: int | None = None,
    force_refresh: bool = False,
) -> ScanResult:
    now = datetime.now(ZoneInfo("Asia/Shanghai"))
    run_time = now.strftime("%Y-%m-%d %H:%M:%S")
    full_scan = symbols is None and limit is None
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
        run_time=run_time,
        full_scan=full_scan,
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
        run_time=run_time,
        full_scan=full_scan,
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
    run_time: str = "",
    full_scan: bool = True,
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
    markdown = render_markdown_report(
        candidates,
        scanned_count,
        failed_count,
        latest_trade_date,
        pool_counts=pool_counts,
        pools=pools,
        run_time=run_time,
        full_scan=full_scan,
        ai_analysis=ai_analysis,
    )
    if ai_analysis:
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
    pools: dict[str, list[Candidate]] | None = None,
    run_time: str = "",
    full_scan: bool = True,
    ai_analysis: str = "",
) -> str:
    pool_counts = pool_counts or {}
    all_candidate_count = sum(pool_counts.values()) if pool_counts else len(candidates)
    run_time = run_time or datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")
    final_action = _final_action(scanned_count, failed_count, pool_counts)
    data_verdict = _data_verdict(scanned_count, failed_count)
    a_count = pool_counts.get("A", 0)
    b_count = pool_counts.get("B", 0)
    c1_count = pool_counts.get("C1", 0)
    c2_count = pool_counts.get("C2", 0)
    d_count = pool_counts.get("D", 0)
    growth_items = _growth_watch_items(candidates, limit=10)
    lines = [
        "# A股突破选股日报",
        "",
        "## 1. 数据状态",
        "",
        f"Trade date: {latest_trade_date}",
        f"Run time: {run_time}",
        "Data source: Tushare + local cache",
        f"Full scan: {'true' if full_scan else 'false'}",
        f"Scanned stocks: {scanned_count}",
        f"Failed stocks: {failed_count}",
        f"All candidate signals: {all_candidate_count}",
        f"Displayed candidates: {len(candidates)}",
        f"Latest price date: {latest_trade_date}",
        "Calendar source: Tushare trade_cal",
        f"History cache status: {'complete' if failed_count == 0 else 'partial'}",
        "",
        f"Data verdict: {data_verdict}",
        "",
        "## 2. 今日最终结论",
        "",
        f"Final action: {final_action}",
        "",
        "原因：",
        f"- A类周线确认数量：{a_count}",
        f"- B类日线预警数量：{b_count}",
        f"- C1强趋势数量：{c1_count}",
        "- 市场环境：未接入四指数周线环境判断，按个股信号保守处理",
        "- 由于 Market regime = NOT_EVALUATED，本日报不输出 LOW_RISK_BUY_CONFIRMED，只输出 BUY_CHECK。",
        f"- 今日交易纪律：{_final_action_reason(final_action)}",
        "",
        "一句话结论：",
        _final_action_sentence(final_action),
        "",
        "## 3. 市场环境",
        "",
        "Market regime: NOT_EVALUATED",
        "",
        "指数状态：",
        "| 指数 | 收盘 | 20周线 | 是否站上 | 20周线方向 |",
        "|---|---:|---:|---|---|",
        "| 上证指数 | - | - | - | - |",
        "| 沪深300 | - | - | - | - |",
        "| 创业板指 | - | - | - | - |",
        "| 科创50 | - | - | - | - |",
        "",
        "成交状态：",
        "- 两市成交额：-",
        "- 相对20日均成交额：-",
        "- 涨停数量：-",
        "- 跌停数量：-",
        "- 主线集中度：" + _theme_concentration(candidates),
        "",
        "市场判断：",
        "当前邮件未接入四指数和全市场涨跌停统计；A 类只能进入 BUY_CHECK，B/C1 先按观察处理。",
        "",
        "## 4. 候选数量汇总",
        "",
        "| 类型 | 数量 | 交易含义 |",
        "|---|---:|---|",
        f"| A 周线确认 | {a_count} | 可交易观察 |",
        f"| B 日线预警 | {b_count} | 观察，等周线确认 |",
        f"| C1 强趋势右尾 | {c1_count} | 右尾观察，不是低风险买点 |",
        f"| C2 不追 | {c2_count} | 不追，等重新整理 |",
        f"| D 排除/突破不足 | {d_count} | 排除 |",
        f"| 成长观察 | {len(growth_items)} | 只跟踪，不买 |",
        "",
        "## 5. 题材/主线分布",
        "",
    ]
    lines.extend(_theme_distribution_table(candidates))

    if not candidates:
        lines.extend(
            [
                "",
                "今日没有符合突破条件的候选。",
                "",
                "## 6. A类：周线确认突破池",
                "",
                "今日 A 类数量：0",
                "",
                "结论：",
                "没有可直接进入低风险突破买入观察的标的。",
                "",
                "## 11. 今日交易计划",
                "",
                "低风险突破仓：",
                "- 今日新增：0",
                "- 原因：无 A 类周线确认",
                "",
                "明确不做：",
                "- 不买 D",
                "- 不买高开超过5%的突破票",
                "- 不买长上影回落票",
            ]
        )
        return "\n".join(lines) + "\n"

    a_items = _pool_display_items("A", candidates, pools)
    b_items = _pool_display_items("B", candidates, pools)[:10]
    c1_items = _pool_display_items("C1", candidates, pools)[:5]
    c2_d_items = (_pool_display_items("C2", candidates, pools) + _pool_display_items("D", candidates, pools))[:10]

    lines.extend(
        [
            "## 6. A类：周线确认突破池",
            "",
            f"今日 A 类数量：{a_count}",
            "",
        ]
    )
    if a_items:
        lines.extend(_core_candidate_table(a_items, limit=None, include_trade_constraints=True))
    else:
        lines.extend(
            [
                "结论：",
                "没有可直接进入低风险突破买入观察的标的。",
            ]
        )
    lines.extend(
        [
            "",
            "## 7. B类：日线预警池",
            "",
        ]
    )
    if b_items:
        lines.extend(_core_candidate_table(b_items, limit=10))
    else:
        lines.append("今日 B 类数量：0")
    lines.extend(
        [
            "",
            "B类复核结论：",
            "- 只观察，不直接重仓。",
            "- 若次日高开超过3%，不追。",
            "- 若回踩压力区上沿不破，继续观察。",
            "- 若跌回压力区内，信号取消。",
            "",
            "## 8. C1类：强趋势右尾观察池",
            "",
        ]
    )
    if c1_items:
        lines.extend(_c1_candidate_table(c1_items))
    else:
        lines.append("今日 C1 类数量：0")
    lines.extend(
        [
            "",
            "C1复核结论：",
            "- C1 不等于低风险买点。",
            "- 只能作为右尾长持观察池。",
            "- 若没有题材共振，降级为 C2。",
            "",
            "## 9. C2 / D 排除摘要",
            "",
            "主要排除原因统计：",
            "",
        ]
    )
    lines.extend(_exclusion_reason_table(c2_d_items))
    lines.extend(
        [
            "",
            "重点不追标的：",
        ]
    )
    if c2_d_items:
        lines.extend(_no_chase_table(c2_d_items[:5]))
    else:
        lines.append("无重点不追标的。")
    lines.extend(
        [
            "",
            "## 10. AI 复核分析",
            "",
            ai_analysis.strip() if ai_analysis.strip() else "未启用或未生成 AI 复核分析。",
            "",
            "## 11. 成长观察池",
            "",
            "说明：当前尚未启用独立成长观察池；下表为本次突破候选中成长分靠前的观察对象，不是买入清单。",
            "",
        ]
    )
    if growth_items:
        lines.extend(_growth_table(growth_items))
    else:
        lines.append("暂无成长分可用的观察对象。")
    lines.extend(
        [
            "",
            "成长观察结论：",
            "这些不是买入清单，只是后续重点跟踪池。",
            "",
            "## 12. 今日交易计划",
            "",
            "低风险突破仓：",
            f"- 今日新增：{a_count}",
            f"- 原因：{'存在 A 类周线确认，但市场环境未评估，只进入 BUY_CHECK' if a_count else '无 A 类周线确认'}",
            "",
            "右尾长持仓：",
            f"- 今日新增：{'0 或 1' if c1_count else '0'}",
            "- 仅当 C1 所属题材共振强，且次日不高开超过3%",
            "",
            "观察列表：",
        ]
    )
    for theme in _top_theme_names(candidates, limit=3):
        lines.append(f"- {theme}：继续观察")
    if not _top_theme_names(candidates, limit=3):
        lines.append("- 暂无集中题材：弱观察")
    lines.extend(
        [
            "",
            "明确不做：",
            "- 不追 C2",
            "- 不买 D",
            "- 不买高开超过5%的突破票",
            "- 不买长上影回落票",
            "",
            "## 13. 输出说明",
            "",
            "正文用于第一轮复核；关键 CSV、HTML 仪表盘和 AI 分析会作为附件随邮件发送，并继续发布到网站端。",
        ]
    )
    return "\n".join(lines) + "\n"


def build_daily_email_subject(result: ScanResult) -> str:
    verdict = _data_verdict(result.scanned_count, result.failed_count)
    action = _final_action(result.scanned_count, result.failed_count, result.pool_counts)
    return (
        f"[Breakout][Daily] {result.latest_trade_date} {verdict} "
        f"scan={result.scanned_count} fail={result.failed_count} "
        f"A={result.pool_counts.get('A', 0)} B={result.pool_counts.get('B', 0)} "
        f"C1={result.pool_counts.get('C1', 0)} action={action}"
    )


def _data_verdict(scanned_count: int, failed_count: int) -> str:
    if scanned_count <= 0:
        return "DATA_FAILED"
    if failed_count >= scanned_count or failed_count > max(50, int(scanned_count * 0.2)):
        return "DATA_FAILED"
    if failed_count > 0:
        return "WARN"
    return "OK"


def _final_action(
    scanned_count: int,
    failed_count: int,
    pool_counts: dict[str, int],
    market_regime: str = "NOT_EVALUATED",
) -> str:
    if _data_verdict(scanned_count, failed_count) == "DATA_FAILED":
        return "DATA_FAILED"
    if pool_counts.get("A", 0) > 0:
        if market_regime == "STRONG_ATTACK":
            return "LOW_RISK_BUY_CONFIRMED"
        if market_regime == "DEFENSIVE":
            return "WATCH_ONLY"
        return "BUY_CHECK"
    if pool_counts.get("B", 0) > 0:
        return "WATCH_ONLY"
    if pool_counts.get("C1", 0) > 0:
        return "RIGHT_TAIL_WATCH"
    return "NO_BUY"


def _final_action_reason(final_action: str) -> str:
    if final_action == "LOW_RISK_BUY_CONFIRMED":
        return "A 类周线确认且市场环境支持，仍需复核题材、财务和次日开盘。"
    if final_action == "BUY_CHECK":
        return "存在 A 类周线确认，但市场环境未确认或偏谨慎，只进入人工买入复核。"
    if final_action == "RIGHT_TAIL_WATCH":
        return "只有 C1 右尾强趋势信号，不属于低风险买点。"
    if final_action == "WATCH_ONLY":
        return "无 A 类低风险买点，B/C1 仅观察。"
    if final_action == "DATA_FAILED":
        return "数据不完整，不输出交易判断。"
    return "无有效低风险突破信号。"


def _final_action_sentence(final_action: str) -> str:
    if final_action == "LOW_RISK_BUY_CONFIRMED":
        return "今天有 A 类标的且市场环境支持，可进入低风险突破买入复核，但不自动交易。"
    if final_action == "BUY_CHECK":
        return "今天有 A 类标的，进入人工复核；市场环境未确认前不自动视为低风险买入确认。"
    if final_action == "RIGHT_TAIL_WATCH":
        return "今天只有右尾强趋势观察信号，不新增低风险突破仓。"
    if final_action == "WATCH_ONLY":
        return "今天只观察，不新增突破仓。"
    if final_action == "DATA_FAILED":
        return "今日扫描数据不完整，不做交易判断。"
    return "今天没有交易价值，不新增突破仓。"


def _pool_display_items(
    signal_type: str,
    candidates: list[Candidate],
    pools: dict[str, list[Candidate]] | None,
) -> list[Candidate]:
    enriched = [item for item in candidates if item.signal_type == signal_type]
    if enriched:
        return enriched
    return list((pools or {}).get(signal_type, []))


def _primary_tag(item: Candidate) -> str:
    return item.tags[0] if item.tags else "未标记"


def _top_theme_names(candidates: list[Candidate], limit: int = 3) -> list[str]:
    counts = Counter(_primary_tag(item) for item in candidates if item.signal_type != "D")
    return [theme for theme, _ in counts.most_common(limit) if theme != "未标记"]


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


def _theme_distribution_table(candidates: list[Candidate]) -> list[str]:
    lines = [
        "| 题材 | A | B | C1 | C2 | 合计 | 判断 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    if not candidates:
        lines.append("| - | 0 | 0 | 0 | 0 | 0 | 无候选 |")
        return lines + ["", "今日主线判断：无候选，无法判断题材共振。"]

    grouped: dict[str, Counter[str]] = {}
    for item in candidates:
        if item.signal_type == "D":
            continue
        theme = _primary_tag(item)
        grouped.setdefault(theme, Counter())[item.signal_type] += 1

    rows: list[tuple[str, Counter[str], int]] = []
    for theme, counts in grouped.items():
        total = sum(counts.get(key, 0) for key in ("A", "B", "C1", "C2"))
        rows.append((theme, counts, total))
    rows.sort(key=lambda row: (-row[2], -row[1].get("A", 0), -row[1].get("B", 0), row[0]))

    for theme, counts, total in rows[:8]:
        lines.append(
            "| {theme} | {a} | {b} | {c1} | {c2} | {total} | {judgement} |".format(
                theme=_md(theme),
                a=counts.get("A", 0),
                b=counts.get("B", 0),
                c1=counts.get("C1", 0),
                c2=counts.get("C2", 0),
                total=total,
                judgement=_theme_judgement(total, counts.get("A", 0), counts.get("C1", 0)),
            )
        )

    top_themes = _top_theme_names(candidates, limit=3)
    if top_themes:
        lines.extend(["", f"今日主线判断：{', '.join(top_themes)} 更值得观察；孤立突破不追。"])
    else:
        lines.extend(["", "今日主线判断：候选较分散，孤立突破不追。"])
    return lines


def _theme_judgement(total: int, a_count: int, c1_count: int) -> str:
    if a_count > 0 and total >= 2:
        return "有A类共振，重点复核"
    if total >= 4:
        return "有共振，重点观察"
    if c1_count > 0:
        return "强趋势但波动偏高"
    if total >= 2:
        return "扩散线"
    return "孤立信号"


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


def _core_candidate_table(
    items: list[Candidate],
    limit: int | None,
    include_trade_constraints: bool = False,
) -> list[str]:
    shown = items if limit is None else items[:limit]
    header = (
        "| 排名 | 股票 | 代码 | 题材 | 收盘 | 压力区上沿 | 突破% | 成交额倍数 | 量能来源 | "
        "触碰次数 | 跨度周 | 技术分 | 成长分 | 买入区 | 交易止损 |"
    )
    align = "|---:|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---:|"
    if include_trade_constraints:
        header += " 是否在买入区 | 次日高开限制 | 建议动作 |"
        align += "---|---|---|"
    header += " 结论 |"
    align += "---|"
    lines = [header, align]
    for idx, item in enumerate(shown, start=1):
        cells = [
            str(idx),
            _md(item.name),
            item.code,
            _md(_primary_tag(item)),
            f"{item.latest_close:.2f}",
            f"{(item.zone_upper or item.resistance):.2f}",
            f"{item.breakout_pct * 100:.2f}",
            f"{(item.activity_ratio or item.volume_ratio):.2f}",
            item.activity_source or "-",
            str(item.resistance_touches),
            str(item.span_weeks),
            f"{item.score:.1f}",
            _fmt_score(item.growth_score),
            f"{item.buy_zone_low:.2f}-{item.buy_zone_high:.2f}",
            f"{(item.trade_stop_loss or item.stop_loss):.2f}",
        ]
        if include_trade_constraints:
            cells.extend(
                [
                    _buy_zone_status(item),
                    "高开>3%不追，>5%放弃；跌回压力区上沿取消",
                    _next_day_trade_action(item),
                ]
            )
        cells.append(_md(item.trade_action or item.position_hint or item.signal_reason))
        lines.append(
            "|{cells}|".format(
                cells="|".join(_md(cell) for cell in cells),
            )
        )
    return lines


def _c1_candidate_table(items: list[Candidate]) -> list[str]:
    lines = [
        "| 排名 | 股票 | 代码 | 题材 | 收盘 | 压力区上沿 | 突破% | 成交额倍数 | 近5日涨幅% | 近10日涨幅% | 是否连续涨停 | 是否长上影 | 技术分 | 风险 | 结论 |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---|---|",
    ]
    for idx, item in enumerate(items, start=1):
        lines.append(
            "|{rank}|{name}|{code}|{theme}|{close:.2f}|{zone_upper:.2f}|{breakout:.2f}|{activity:.2f}|"
            "{ret5:.2f}|{ret10:.2f}|{limit_up}|{long_shadow}|{score:.1f}|{risk}|{action}|".format(
                rank=idx,
                name=_md(item.name),
                code=item.code,
                theme=_md(_primary_tag(item)),
                close=item.latest_close,
                zone_upper=item.zone_upper or item.resistance,
                breakout=item.breakout_pct * 100,
                activity=item.activity_ratio or item.volume_ratio,
                ret5=item.recent_5d_pct * 100,
                ret10=item.recent_10d_pct * 100,
                limit_up="是" if item.consecutive_limit_up_days >= 2 else "否",
                long_shadow="是" if item.long_upper_shadow else "否",
                score=item.score,
                risk=_md("已远离低风险买点" if item.breakout_pct >= 0.08 else "强趋势但追高风险"),
                action=_md(item.trade_action or "右尾观察，不低吸"),
            )
        )
    return lines


def _exclusion_reason_table(items: list[Candidate]) -> list[str]:
    counts = Counter(_exclusion_reason(item) for item in items)
    if not counts:
        counts["无展示样本"] = 0
    lines = ["| 原因 | 数量 |", "|---|---:|"]
    for reason, count in counts.most_common():
        lines.append(f"| {_md(reason)} | {count} |")
    return lines


def _no_chase_table(items: list[Candidate]) -> list[str]:
    lines = ["| 股票 | 代码 | 原因 |", "|---|---|---|"]
    for item in items:
        lines.append(f"| {_md(item.name)} | {item.code} | {_md(_exclusion_reason(item))} |")
    return lines


def _growth_watch_items(candidates: list[Candidate], limit: int = 10) -> list[Candidate]:
    return sorted(
        [item for item in candidates if item.growth_score > 0],
        key=lambda item: (-item.growth_score, item.signal_type, item.code),
    )[:limit]


def _growth_table(items: list[Candidate]) -> list[str]:
    lines = [
        "| 排名 | 股票 | 代码 | 题材 | 成长分 | 营收同比% | 净利同比% | ROE% | 距压力区% | 观察触发条件 |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for idx, item in enumerate(items, start=1):
        lines.append(
            "|{rank}|{name}|{code}|{theme}|{growth}|{revenue}|{profit}|{roe}|{distance:.2f}|{trigger}|".format(
                rank=idx,
                name=_md(item.name),
                code=item.code,
                theme=_md(_primary_tag(item)),
                growth=_fmt_score(item.growth_score),
                revenue=_fmt_optional(item.revenue_yoy),
                profit=_fmt_optional(item.profit_yoy),
                roe=_fmt_optional(item.roe),
                distance=item.breakout_pct * 100,
                trigger=_md("周线确认 + 成交额倍数>1.8" if item.signal_type != "A" else "回踩不破压力区上沿 + 缩量企稳"),
            )
        )
    return lines


def _exclusion_reason(item: Candidate) -> str:
    text = " ".join(part for part in (item.signal_reason, item.trade_action, item.position_hint) if part)
    if item.breakout_pct < 0.02 or "突破不足" in text:
        return "突破不足2%"
    if item.breakout_pct > 0.12 or "过远" in text or "远离" in text or "不追" in text:
        return "距离压力区过远"
    if (item.activity_ratio or item.volume_ratio) < 1.8 or "量能" in text or "成交" in text:
        return "成交额倍数不足1.8"
    if "基本面" in text or "ROE" in text or "净利" in text or "营收" in text:
        return "基本面风险"
    if not item.tags:
        return "题材孤立"
    return "交易纪律不满足"


def _fmt_score(value: float) -> str:
    return f"{value:.1f}" if value and value > 0 else "-"


def _md(value: object) -> str:
    return str(value if value is not None else "-").replace("|", "/").replace("\n", " ").strip() or "-"


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
