#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$APP_DIR"

SITE_DIR="${SITE_DIR:-$APP_DIR/site}"
mkdir -p "$SITE_DIR"

LATEST_DASHBOARD="$(find "$APP_DIR/outputs" -mindepth 2 -maxdepth 2 -name breakout_dashboard.html 2>/dev/null | sort | tail -1 || true)"
LATEST_DIR="$(dirname "$LATEST_DASHBOARD")"
if [ -z "$LATEST_DASHBOARD" ] || [ ! -f "$LATEST_DASHBOARD" ]; then
  echo "没有找到可发布的 breakout_dashboard.html" >&2
  exit 1
fi

cp "$LATEST_DIR/breakout_dashboard.html" "$SITE_DIR/index.html"
cp "$LATEST_DIR/breakout_report.md" "$SITE_DIR/breakout_report.md" 2>/dev/null || true
cp "$LATEST_DIR/ai_analysis.md" "$SITE_DIR/ai_analysis.md" 2>/dev/null || true
cp "$LATEST_DIR/breakout_candidates.csv" "$SITE_DIR/breakout_candidates.csv" 2>/dev/null || true
cp "$LATEST_DIR/breakout_candidates.xlsx" "$SITE_DIR/breakout_candidates.xlsx" 2>/dev/null || true
for pool in A B C1 C2 D; do
  cp "$LATEST_DIR/breakout_${pool}.csv" "$SITE_DIR/breakout_${pool}.csv" 2>/dev/null || true
done
basename "$LATEST_DIR" > "$SITE_DIR/latest_trade_date.txt"

LATEST_BACKTEST="$(find "$APP_DIR/outputs/backtest" -mindepth 2 -maxdepth 2 -name backtest_report.html 2>/dev/null | sort | tail -1 || true)"
if [ -n "$LATEST_BACKTEST" ] && [ -f "$LATEST_BACKTEST" ]; then
  BACKTEST_DIR="$(dirname "$LATEST_BACKTEST")"
  cp "$BACKTEST_DIR/backtest_report.html" "$SITE_DIR/backtest.html"
  cp "$BACKTEST_DIR/backtest_summary.csv" "$SITE_DIR/backtest_summary.csv" 2>/dev/null || true
  cp "$BACKTEST_DIR/backtest_trades.csv" "$SITE_DIR/backtest_trades.csv" 2>/dev/null || true
  cp "$BACKTEST_DIR/backtest_filters.csv" "$SITE_DIR/backtest_filters.csv" 2>/dev/null || true
  cp "$BACKTEST_DIR/long_hold_summary.csv" "$SITE_DIR/long_hold_summary.csv" 2>/dev/null || true
  cp "$BACKTEST_DIR/long_hold_trades.csv" "$SITE_DIR/long_hold_trades.csv" 2>/dev/null || true
  cp "$BACKTEST_DIR/long_hold_filters.csv" "$SITE_DIR/long_hold_filters.csv" 2>/dev/null || true
  cp "$BACKTEST_DIR/first_signal_summary.csv" "$SITE_DIR/first_signal_summary.csv" 2>/dev/null || true
  cp "$BACKTEST_DIR/first_signal_trades.csv" "$SITE_DIR/first_signal_trades.csv" 2>/dev/null || true
  cp "$BACKTEST_DIR/first_signal_filters.csv" "$SITE_DIR/first_signal_filters.csv" 2>/dev/null || true
fi

echo "Published $LATEST_DIR to $SITE_DIR"
