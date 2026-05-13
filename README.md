# A股突破选股自动化系统

每天拉取 A 股行情，筛选“突破周线阻力位但未明显追高”的股票，生成 CSV/Excel/Markdown/HTML 可视化复核页，并通过邮件提醒。

## 本地运行

```bash
cp config.example.toml config.toml
uv sync
uv run a-breakout --config config.toml doctor
uv run a-breakout --config config.toml run --no-email --symbols 000001,600519,300750
```

运行后可打开输出目录里的 `breakout_dashboard.html`，点击候选股票查看最近 60/120/250 日 K 线、阻力位、买入区和止损线。

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
MAIL_TO=your-email@example.com
EOF
```

该模式要求服务器上的 Codex 已登录，并且 Gmail 插件连接器可用；发送时会调用 `codex exec --dangerously-bypass-approvals-and-sandbox`。

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
- 完整规则见 [docs/model.md](/Users/zmc/codex/stocks/docs/model.md)。
