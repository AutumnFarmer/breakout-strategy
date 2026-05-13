# A股突破选股自动化系统

每天拉取 A 股行情，筛选“突破 156 周压力区但未明显追高”的股票，生成 CSV/Excel/Markdown/HTML 可视化复核页，并通过邮件提醒。

## 本地运行

```bash
cp config.example.toml config.toml
uv sync
uv run a-breakout --config config.toml doctor
uv run a-breakout --config config.toml run --no-email --symbols 000001,600519,300750
```

运行后可打开输出目录里的 `breakout_dashboard.html`。网页支持候选股列表、A/B/C/D 信号类型、题材 tag 筛选、交互式 K 线缩放/平移、压力区、买入区、止损线和 AI 选股分析员复核。

配置邮件后运行:

```bash
cp config.example.toml config.toml
cat > .env <<'EOF'
A_BREAKOUT_EMAIL_ENABLED=true
SMTP_HOST=smtp.example.com
SMTP_PORT=465
SMTP_USER=your-account@example.com
SMTP_PASSWORD=your-smtp-auth-code
SMTP_SSL=true
MAIL_FROM=your-account@example.com
MAIL_TO=your-target@example.com
EOF

./run_daily.sh
```

不要把 `.env` 提交到 Git；这里面是邮箱授权码。

服务器上推荐使用 Codex Gmail 插件模式:

```bash
cat > .env <<'EOF'
A_BREAKOUT_EMAIL_ENABLED=true
A_BREAKOUT_EMAIL_METHOD=codex_gmail
MAIL_TO=zhangmc895@gmail.com
EOF
```

该模式要求服务器上的 Codex 已登录，并且 Gmail 插件连接器可用；发送时会调用 `codex exec --sandbox read-only`，并通过 Gmail 插件附加日报文件。

Tushare 接口偶发超时时，可以在 `.env` 中调高超时和重试参数:

```bash
TUSHARE_TIMEOUT=30
TUSHARE_RETRIES=3
TUSHARE_RETRY_DELAY=2
TUSHARE_RETRY_MAX_DELAY=15
```

题材 tag 和财务成长指标只会在最终 TopN 候选出来后补充，不参与全市场预筛选，避免为了辅助数据拖慢全量扫描。题材优先使用 Tushare `stock_basic`/`concept_detail`，失败时退回 AkShare 个股行业信息，并写入 `data/cache/stock_tags/` 缓存。成长指标使用 Tushare `fina_indicator`，输出成长分、营收同比、净利同比、ROE、毛利率和资产负债率，并写入 `data/cache/financial_metrics/` 缓存。

辅助数据缓存天数可通过环境变量调整:

```bash
STOCK_TAG_CACHE_DAYS=30
STOCK_FINANCIAL_CACHE_DAYS=30
```

AI 分析默认关闭；服务器可通过本机 Sub2API 网关调用 `gpt-5.5`:

```toml
[ai_analysis]
enabled = true
provider = "sub2api"
base_url = "http://127.0.0.1:8080"
model = "gpt-5.5"
timeout = 240
max_candidates = 20
allow_local_key_discovery = true
```

如果不允许本机自动发现 Sub2API key，可关闭 `allow_local_key_discovery`，并在 `.env` 里设置 `A_BREAKOUT_AI_API_KEY` 或 `SUB2API_API_KEY`。

## 服务器部署

默认部署到本机 SSH 配置里的 `aiwork-server`，目录是 `/opt/a-breakout-screener`。

```bash
cp config.example.toml config.toml
bash deploy/deploy.sh
```

部署脚本会在服务器上自动生成一个 `codex_gmail` 模式的 `.env`。如需改成 SMTP，再到服务器补充真实 `.env`:

```bash
ssh aiwork-server
cd /opt/a-breakout-screener
vi .env
```

再检查:

```bash
bash deploy/check_server.sh
```

默认 cron 是每个交易日 08:20 运行:

```text
20 8 * * 1-5
```

如果想改时间:

```bash
CRON_TIME="30 8 * * 1-5" bash deploy/deploy.sh
```

## 重要说明

- AKShare 文档说明 `stock_zh_a_hist` 的当日收盘价应在收盘后获取；早上 08:20 运行时拿到的是最近一个已收盘交易日。
- 邮件结果是规则扫描结果，不是投资建议，也不会自动下单。
- 完整规则见 [docs/model.md](docs/model.md)。
