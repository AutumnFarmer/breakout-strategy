# A股突破选股模型

本系统把“突破长期压力区”的 V2 思路固化成每日可执行的规则扫描器。核心目标是严格识别可交易观察的突破标的；宁可候选为空，也不放松突破交易池。

## 数据源

- 全市场列表: `ak.stock_zh_a_spot_em()`
- 个股历史日线: `ak.stock_zh_a_hist(symbol, period="daily", adjust="qfq")`
- 周线由本地从日线按 `W-FRI` 聚合生成
- Tushare 可用时用于行情、题材 tag、流通市值和候选股财务成长指标

## 默认过滤

- 保留 `00`、`30`、`60`、`68` 开头的 A 股代码
- 排除名称包含 `ST`、`*ST`、`退` 的股票
- 排除成交额低于 1 亿元、价格低于 3 元的股票
- 历史数据不足 520 个交易日不参与评分

## 入选条件

- 周线压力区使用最近 156 周历史构建，并排除最近 4 周
- 压力区由 pivot high 聚类得到，而不是单一历史高点
- 有效触碰不少于 3 次，触碰间隔不少于 4 周
- 首次触碰到最近触碰跨度不少于 20 周
- 突破幅度全部基于压力区上沿计算
- 2%-6% 为主要可交易观察区，6%-8% 谨慎观察，8%-12% 不追，12% 以上排除
- 收盘价、MA10、MA20 的排列越强得分越高
- 最新成交量相对近 20 日均量放大但不过热时得分更高
- 压力区触达次数、跨度和聚类质量越好，压力区有效性得分越高

## 信号分类

- A 类: 周线确认且成交额放大，可交易观察
- B 类: 日线预警，周线或量能仍需确认
- C 类: 突破距离压力区过远，不追
- D 类: 突破不足 2%，只观察压力区附近反应

买入区、突破幅度和交易止损均基于压力区上沿:

- 买入区: `zone_upper * 1.02` 到 `zone_upper * 1.06`
- 交易止损: `zone_upper * 0.97`
- 结构止损: 结合 MA20、近 20 日低点和 ATR 辅助给出

## 成长性复核

全市场筛选仍以突破技术指标为核心。最终候选出来后，系统再补充非阻断的成长性数据:

- 题材/行业 tag: 用于观察市场热点和板块聚类
- 财务期、营收同比、净利同比、ROE、毛利率、资产负债率
- 成长分: 将收入增长、利润增长、ROE、毛利率和负债水平折算为 0-100 的辅助分

成长分不直接替代技术突破得分。它用于区分“只有短线量价突破”和“量价突破同时具备基本面成长证据”的候选，供网页端和 AI 分析员复核。

## 输出

每日输出三个文件:

- `breakout_candidates.csv`
- `breakout_candidates.xlsx`
- `breakout_report.md`
- `breakout_dashboard.html`
- `ai_analysis.md`

邮件正文使用 Markdown 报告，附件包含 CSV、Excel 和 HTML 仪表盘。

## 实盘化 first-signal 回测

单独命令 `backtest-first-signal-executable` 使用本地历史缓存做首次信号的实盘化近似回测，不会联网拉取行情或交易日历；交易日历只读本地 `trade_cal` 缓存，缺失时按保守口径处理，只有周五可视为周线确认。旧命令 `backtest-first-signal` 保留固定金额研究口径，用于兼容原有输出。

```bash
uv run a-breakout --config config.toml backtest-first-signal-executable \
  --lot-size 100 \
  --max-capital-per-trade 5000 \
  --max-total-capital 200000 \
  --min-capital-per-trade 0 \
  --max-buys-per-day 3 \
  --max-theme-buys-per-day 2 \
  --slippage-bps 10 \
  --fee-bps 3 \
  --min-fee 5 \
  --sell-tax-bps 5 \
  --buy-signal-types A
```

该模式按交易日回放首次信号，默认只买入 A 类周线确认；B/C1/C2 只进入观察统计，不会消耗后续 A 类买点。需要研究 B 类小仓试错时可显式传 `--buy-signal-types A,B`，避免默认把日线预警和 A 类买点混成同一口径。下一交易日开盘按 `lot_size` 整手买入，如果一手成本加最低佣金超过 `max_capital_per_trade` 则跳过。设置 `max_total_capital` 后会按持仓占用约束总资金，默认 0 表示只统计峰值占用、不限制总资金。同一天最多新增 `max_buys_per_day` 只，传 0 可做纯观察 dry-run；同一主标签最多新增 `max_theme_buys_per_day` 只。对 `--buy-signal-types` 内的买入信号，首次出现就是一次执行机会；若因日内限流、题材限额、资金约束或一手过贵跳过，后续重复买入信号不再追买，并在过滤统计中记录为机会损失。主标签来自本地 `stock_tags` 缓存，缺失时会进入 `UNKNOWN` 主题桶参与限额，汇总里会标明题材标签缓存覆盖率和候选覆盖率。

输出目录为 `outputs/backtest/<date>/first_signal_executable/`。`first_signal_executable_trades.csv` 会同时保留原始开盘/卖出价、滑点后的有效成交价、量能来源、整手数、实际投入、买卖费用、资金占用和 PnL；`first_signal_executable_summary.csv` 汇总总投入、含买入费的现金投入、总收益、胜率、止损率、最大持仓数、峰值资金占用、总资金约束跳过数量和一手过贵跳过数量。费用模型包含 `fee_bps` 比例佣金、`min_fee` 单边最低佣金和 `sell_tax_bps` 卖出侧印花税。

## 风险边界

这是观察清单，不是自动交易系统。邮件里的买入区、止损位和仓位提示只用于二次判断，不应直接作为下单指令。
实盘化回测仍然只是基于历史缓存的近似模拟，不是收益承诺；真实成交还会受到涨跌停、盘口深度、停牌、税费细则和人工执行延迟影响。
