# Codex Issue Runner

This repository can use GitHub Issues as the handoff surface for server-side Codex work:

1. Create a GitHub Issue with the `codex-task` label.
2. The server runner creates a branch from `optimize/breakout-strategy`.
3. The runner feeds the issue body to `codex exec`.
4. The runner runs `uv run pytest`.
5. If tests pass, it commits, pushes, opens a PR, comments on the issue, and labels the PR `needs-review`.

## Requirements

- `git`
- `gh`
- `codex`
- `uv`
- `flock`
- GitHub CLI authenticated for `AutumnFarmer/breakout-strategy`

Check authentication:

```bash
gh auth status
```

Authenticate with an interactive flow:

```bash
gh auth login --hostname github.com --git-protocol ssh --web
```

Or with a token that has access to issues, pull requests, and repository contents:

```bash
printf '%s' "$GITHUB_TOKEN" | gh auth login --with-token
```

## Manual Run

```bash
cd /opt/a-breakout-screener
bash scripts/codex_issue_runner.sh
```

The script refuses to run if the worktree is dirty. It writes logs under:

```text
logs/codex_issue_runner/
```

## Cron

Only enable cron after `gh auth status` succeeds and a manual runner invocation
can read GitHub issues.

```cron
*/10 * * * * cd /opt/a-breakout-screener && bash scripts/codex_issue_runner.sh >> logs/codex_issue_runner.log 2>&1
```

## Environment Overrides

```bash
REPO_DIR=/opt/a-breakout-screener
REPO=AutumnFarmer/breakout-strategy
BASE_BRANCH=optimize/breakout-strategy
TEST_CMD="uv run pytest"
```

The runner intentionally stages only repository changes and excludes `.env`, local config, caches, generated reports, site output, virtualenvs, and logs.
