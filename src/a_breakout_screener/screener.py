from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from .config import AppConfig
from .data import fetch_history, fetch_spot, filter_spot_universe, prepare_history_cache
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
            "|排名|代码|名称|收盘|阻力|突破%|量能比|触达|得分|买入区|止损|提示|",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---|---:|---|",
        ]
    )
    for idx, item in enumerate(candidates, start=1):
        lines.append(
            "|{rank}|{code}|{name}|{close:.2f}|{resistance:.2f}|{breakout:.2f}|"
            "{vr:.2f}|{touches}|{score:.1f}|{buy_low:.2f}-{buy_high:.2f}|{stop:.2f}|{hint}|".format(
                rank=idx,
                code=item.code,
                name=item.name,
                close=item.latest_close,
                resistance=item.resistance,
                breakout=item.breakout_pct * 100,
                vr=item.volume_ratio,
                touches=item.resistance_touches,
                score=item.score,
                buy_low=item.buy_zone_low,
                buy_high=item.buy_zone_high,
                stop=item.stop_loss,
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
