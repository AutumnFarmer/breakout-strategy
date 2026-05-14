# Codex Automation Runners

This repository can use GitHub Issues as the handoff surface for server-side Codex work:

1. Create a GitHub Issue with the `codex-task` label.
2. The server runner creates a branch from `optimize/breakout-strategy`.
3. The runner feeds the issue body to `codex exec`.
4. The runner runs `uv run pytest`.
5. If tests pass, it commits, pushes, opens a PR, comments on the issue, and labels the PR `needs-review`.
6. A second runner can review `needs-review` PRs and comment with a Codex first-pass review.

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

Run the PR reviewer manually:

```bash
cd /opt/a-breakout-screener
bash scripts/codex_pr_reviewer.sh
```

The PR reviewer scans open non-draft PRs with the `needs-review` label, asks
Codex to review the GitHub diff in read-only mode, comments on the PR, and moves
the label to `codex-reviewed`. It does not approve, merge, or close PRs.

Reviewer logs are written under:

```text
logs/codex_pr_reviewer/
```

## Cron

Only enable cron after `gh auth status` succeeds and a manual runner invocation
can read GitHub issues.

```cron
*/10 * * * * cd /opt/a-breakout-screener && bash scripts/codex_issue_runner.sh >> logs/codex_issue_runner.log 2>&1
*/10 * * * * cd /opt/a-breakout-screener && bash scripts/codex_pr_reviewer.sh >> logs/codex_pr_reviewer.log 2>&1
```

## Environment Overrides

```bash
REPO_DIR=/opt/a-breakout-screener
REPO=AutumnFarmer/breakout-strategy
BASE_BRANCH=optimize/breakout-strategy
TEST_CMD="uv run pytest"
MAX_DIFF_BYTES=180000
MAX_REVIEW_COMMENT_BYTES=55000
```

The runner intentionally stages only repository changes and excludes `.env`, local config, caches, generated reports, site output, virtualenvs, and logs.

## Labels

Issue runner labels:

- `codex-task`
- `codex-running`
- `codex-done`
- `codex-failed`
- `codex-nochange`

PR reviewer labels:

- `needs-review`
- `codex-reviewing`
- `codex-reviewed`
- `codex-review-failed`
- `human-reviewed`
