from __future__ import annotations

import argparse
from dataclasses import replace
import os
import sys
from pathlib import Path

from .config import AppConfig, load_config
from .emailer import send_report, validate_email_transport
from .backtest import run_backtest, run_first_signal_backtest, run_first_signal_executable_backtest
from .screener import ScanResult, build_daily_email_subject, run_scan


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "env_file", None):
        load_env_file(Path(args.env_file))
    config = load_config(Path(args.config) if args.config else None)
    apply_network_env(config)

    if args.command == "doctor":
        return _doctor(config, skip_network=args.skip_network)
    if args.command == "run":
        return _run(args, config)
    if args.command == "backtest":
        return _backtest(args, config)
    if args.command == "backtest-first-signal":
        return _backtest_first_signal(args, config)
    if args.command == "backtest-first-signal-executable":
        return _backtest_first_signal_executable(args, config)

    parser.print_help()
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="a-breakout", description="A股突破选股自动扫描器")
    parser.add_argument("--config", default="config.toml", help="TOML config path, default: config.toml")
    parser.add_argument("--env-file", default="", help="Optional env file for SMTP secrets")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Run daily stock scan")
    run.add_argument("--symbols", default="", help="Comma separated stock codes for a focused smoke run")
    run.add_argument("--limit", type=int, default=0, help="Limit the universe size, useful for smoke tests")
    run.add_argument("--force-refresh", action="store_true", help="Ignore cached history and fetch fresh data")
    email_group = run.add_mutually_exclusive_group()
    email_group.add_argument("--send-email", action="store_true", help="Force sending the email report")
    email_group.add_argument("--no-email", action="store_true", help="Disable email sending for this run")

    doctor = sub.add_parser("doctor", help="Check dependencies and data-source connectivity")
    doctor.add_argument("--skip-network", action="store_true", help="Do not call AkShare data endpoints")

    backtest = sub.add_parser("backtest", help="Run a coarse historical backtest from cached daily bars")
    backtest.add_argument("--days", type=int, default=120, help="Number of recent signal days to backtest")
    backtest.add_argument("--top-n", default="10,20,30", help="Comma separated TopN groups, default: 10,20,30")
    backtest.add_argument("--holding-days", default="5,10,20", help="Comma separated holding days, default: 5,10,20")
    backtest.add_argument("--symbols", default="", help="Comma separated stock codes for a focused backtest")
    backtest.add_argument("--workers", type=int, default=0, help="Override backtest worker threads for this run")
    backtest.add_argument("--long-hold-days", type=int, default=365, help="Long-hold lookback calendar days, default: 365")
    backtest.add_argument("--long-hold-top-n", type=int, default=10, help="Daily TopN for long-hold backtest, default: 10")
    backtest.add_argument("--capital-per-trade", type=float, default=1000.0, help="Capital per selected stock in long-hold backtest")
    backtest.add_argument("--stop-loss-pct", type=float, default=50.0, help="Stop loss percent for long-hold backtest")

    first_signal = sub.add_parser("backtest-first-signal", help="Run fixed-amount first-signal research backtest")
    first_signal.add_argument("--symbols", default="", help="Comma separated stock codes for a focused backtest")
    first_signal.add_argument("--workers", type=int, default=0, help="Override worker threads for this run")
    first_signal.add_argument("--lookback-days", type=int, default=365, help="Lookback calendar days, default: 365")
    first_signal.add_argument("--capital-per-trade", type=float, default=1000.0, help="Capital per first signal, default: 1000")
    first_signal.add_argument("--stop-loss-pct", type=float, default=30.0, help="Stop loss percent, default: 30")

    executable = sub.add_parser("backtest-first-signal-executable", help="Run executable first-signal buy-once backtest")
    executable.add_argument("--symbols", default="", help="Comma separated stock codes for a focused backtest")
    executable.add_argument("--workers", type=int, default=0, help="Override worker threads for this run")
    executable.add_argument("--lookback-days", type=int, default=365, help="Lookback calendar days, default: 365")
    executable.add_argument("--lot-size", type=int, default=100, help="Round-lot size, default: 100")
    executable.add_argument("--max-capital-per-trade", type=float, default=5000.0, help="Maximum cash per stock, default: 5000")
    executable.add_argument("--max-total-capital", type=float, default=0.0, help="Maximum total reserved cash, default: 0 means unlimited")
    executable.add_argument("--min-capital-per-trade", type=float, default=0.0, help="Minimum invested cash per stock, default: 0")
    executable.add_argument("--max-buys-per-day", type=int, default=3, help="Maximum new buys per signal day, default: 3")
    executable.add_argument("--max-theme-buys-per-day", type=int, default=2, help="Maximum buys per primary tag per day, default: 2")
    executable.add_argument("--slippage-bps", type=float, default=10.0, help="Entry/exit slippage in basis points, default: 10")
    executable.add_argument("--fee-bps", type=float, default=3.0, help="Buy/sell fee rate in basis points, default: 3")
    executable.add_argument("--min-fee", type=float, default=5.0, help="Minimum commission per side, default: 5")
    executable.add_argument("--sell-tax-bps", type=float, default=5.0, help="Sell-side stamp tax in basis points, default: 5")
    executable.add_argument(
        "--buy-signal-types",
        type=_parse_signal_types,
        default=("A",),
        help="Comma separated signal types eligible for executable buys, default: A",
    )
    executable.add_argument("--capital-per-trade", type=float, default=None, help=argparse.SUPPRESS)
    executable.add_argument("--stop-loss-pct", type=float, default=30.0, help="Stop loss percent, default: 30")
    return parser


def _run(args: argparse.Namespace, config: AppConfig) -> int:
    symbols = {item.strip() for item in args.symbols.split(",") if item.strip()} or None
    result = run_scan(
        config=config,
        symbols=symbols,
        limit=args.limit or None,
        force_refresh=args.force_refresh,
    )
    print(f"扫描完成: {result.scanned_count} 只，入选 {len(result.candidates)} 只，失败 {result.failed_count} 只")
    print(f"CSV: {result.csv_path}")
    print(f"Excel: {result.xlsx_path}")
    print(f"日报: {result.markdown_path}")
    print(f"可视化: {result.html_path}")

    should_send = config.email.enabled
    if args.send_email:
        should_send = True
    if args.no_email:
        should_send = False
    if should_send:
        body = result.markdown_path.read_text(encoding="utf-8")
        subject = build_daily_email_subject(result)
        send_report(
            email_config=config.email,
            subject=subject,
            body=body,
            attachments=[],
        )
        print("邮件发送完成，正文邮件，无附件")
    else:
        print("邮件发送已跳过")
    return 0


def _daily_scan_email_attachments(result: ScanResult) -> list[Path]:
    wanted = [
        result.csv_path,
        result.output_dir / "breakout_A.csv",
        result.output_dir / "breakout_B.csv",
        result.output_dir / "breakout_C1.csv",
        result.output_dir / "breakout_C2.csv",
        result.output_dir / "breakout_D.csv",
        result.output_dir / "growth_watchlist.csv",
        result.html_path,
        result.ai_analysis_path,
    ]
    attachments: list[Path] = []
    for path in wanted:
        if path is None:
            continue
        path = Path(path)
        if path.exists() and path.is_file():
            attachments.append(path)
    return attachments


def _backtest(args: argparse.Namespace, config: AppConfig) -> int:
    if args.workers and args.workers > 0:
        config = replace(config, screener=replace(config.screener, max_workers=args.workers))
    result = run_backtest(
        config=config,
        days=max(1, args.days),
        top_ns=_parse_int_tuple(args.top_n),
        holding_days=_parse_int_tuple(args.holding_days),
        symbols={item.strip() for item in args.symbols.split(",") if item.strip()} or None,
        long_hold_days=max(1, args.long_hold_days),
        long_hold_top_n=max(1, args.long_hold_top_n),
        capital_per_trade=max(0.01, args.capital_per_trade),
        stop_loss_pct=max(0.0, args.stop_loss_pct),
    )
    print(f"回测完成: {result.stock_count} 只股票，{result.signal_days} 个信号日，{result.trade_count} 笔模拟交易")
    print(f"交易明细: {result.trades_path}")
    print(f"汇总结果: {result.summary_path}")
    print(f"长持交易: {result.long_hold_trades_path}")
    print(f"长持汇总: {result.long_hold_summary_path}")
    print(f"首次信号交易: {result.first_signal_trades_path}")
    print(f"首次信号汇总: {result.first_signal_summary_path}")
    print(f"可视化: {result.html_path}")
    return 0


def _backtest_first_signal(args: argparse.Namespace, config: AppConfig) -> int:
    if args.workers and args.workers > 0:
        config = replace(config, screener=replace(config.screener, max_workers=args.workers))
    result = run_first_signal_backtest(
        config=config,
        lookback_days=max(1, args.lookback_days),
        capital_per_trade=max(0.01, args.capital_per_trade),
        stop_loss_pct=max(0.0, args.stop_loss_pct),
        symbols={item.strip() for item in args.symbols.split(",") if item.strip()} or None,
    )
    print(
        f"首次信号研究回测完成: {result.stock_count} 只股票，"
        f"{result.signal_days} 个信号日，{result.trade_count} 笔买入"
    )
    print(f"首次信号交易: {result.trades_path}")
    print(f"首次信号汇总: {result.summary_path}")
    print(f"首次信号过滤: {result.filters_path}")
    print(f"可视化: {result.html_path}")
    return 0


def _backtest_first_signal_executable(args: argparse.Namespace, config: AppConfig) -> int:
    if args.workers and args.workers > 0:
        config = replace(config, screener=replace(config.screener, max_workers=args.workers))
    max_capital_per_trade = args.max_capital_per_trade
    if args.capital_per_trade is not None:
        max_capital_per_trade = args.capital_per_trade
    result = run_first_signal_executable_backtest(
        config=config,
        lookback_days=max(1, args.lookback_days),
        lot_size=max(1, args.lot_size),
        max_capital_per_trade=max(0.01, max_capital_per_trade),
        max_total_capital=max(0.0, args.max_total_capital),
        min_capital_per_trade=max(0.0, args.min_capital_per_trade),
        max_buys_per_day=max(0, args.max_buys_per_day),
        max_theme_buys_per_day=max(0, args.max_theme_buys_per_day),
        slippage_bps=max(0.0, args.slippage_bps),
        fee_bps=max(0.0, args.fee_bps),
        min_fee=max(0.0, args.min_fee),
        sell_tax_bps=max(0.0, args.sell_tax_bps),
        stop_loss_pct=max(0.0, args.stop_loss_pct),
        buy_signal_types=args.buy_signal_types,
        symbols={item.strip() for item in args.symbols.split(",") if item.strip()} or None,
    )
    print(
        f"首次信号实盘化回测完成: {result.stock_count} 只股票，"
        f"{result.signal_days} 个信号日，{result.trade_count} 笔买入"
    )
    print(f"首次信号交易: {result.trades_path}")
    print(f"首次信号汇总: {result.summary_path}")
    print(f"首次信号过滤: {result.filters_path}")
    print(f"可视化: {result.html_path}")
    return 0


def _doctor(config: AppConfig, skip_network: bool) -> int:
    print(f"输出目录: {config.paths.output_dir}")
    print(f"缓存目录: {config.paths.cache_dir}")
    try:
        import akshare  # noqa: F401
        import numpy  # noqa: F401
        import openpyxl  # noqa: F401
        import pandas  # noqa: F401
    except ImportError as exc:
        print(f"依赖检查失败: {exc}", file=sys.stderr)
        return 1
    print("依赖检查: OK")

    if not skip_network:
        from .data import fetch_spot

        spot = fetch_spot()
        print(f"A股行情列表: {len(spot)} 行")
    if config.email.enabled:
        problems = validate_email_transport(config.email)
        if problems:
            print(f"邮件配置不完整: {', '.join(problems)}", file=sys.stderr)
            return 1
        print("邮件配置: OK")
    else:
        print("邮件配置: 未启用")
    return 0


def load_env_file(path: Path) -> None:
    path = path.expanduser()
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def apply_network_env(config: AppConfig) -> None:
    if not config.network.disable_system_proxy:
        return
    os.environ.setdefault("NO_PROXY", "*")
    os.environ.setdefault("no_proxy", "*")


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    items = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not items:
        raise ValueError("expected at least one integer")
    return items


def _parse_signal_types(value: str) -> tuple[str, ...]:
    allowed = {"A", "B"}
    items = tuple(dict.fromkeys(item.strip().upper() for item in value.split(",") if item.strip()))
    invalid = [item for item in items if item not in allowed]
    if invalid:
        raise argparse.ArgumentTypeError(f"invalid signal type(s): {', '.join(invalid)}")
    return items or ("A",)


if __name__ == "__main__":
    raise SystemExit(main())
