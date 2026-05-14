#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-/opt/a-breakout-screener}"
REPO="${REPO:-AutumnFarmer/breakout-strategy}"
BASE_BRANCH="${BASE_BRANCH:-optimize/breakout-strategy}"

PENDING_LABEL="${PENDING_LABEL:-codex-task}"
RUNNING_LABEL="${RUNNING_LABEL:-codex-running}"
DONE_LABEL="${DONE_LABEL:-codex-done}"
FAILED_LABEL="${FAILED_LABEL:-codex-failed}"
NOCHANGE_LABEL="${NOCHANGE_LABEL:-codex-nochange}"
NEEDS_REVIEW_LABEL="${NEEDS_REVIEW_LABEL:-needs-review}"

TEST_CMD="${TEST_CMD:-uv run pytest}"
CODEX_BYPASS_FLAG="${CODEX_BYPASS_FLAG:---dangerously-bypass-approvals-and-sandbox}"
LOG_DIR="${LOG_DIR:-$REPO_DIR/logs/codex_issue_runner}"
LOCK_FILE="${LOCK_FILE:-/tmp/a_breakout_screener_codex_issue_runner.lock}"
COMMENT_LOG_TAIL_LINES="${COMMENT_LOG_TAIL_LINES:-80}"

ISSUE_NUMBER=""
ISSUE_TITLE=""
BRANCH=""
LOG_FILE=""
RUN_ID="$(date -u +%Y%m%d%H%M%S)"
HANDLED_EXIT=0

require_cmds() {
  local missing=()
  local cmd

  for cmd in git gh codex uv flock; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
      missing+=("$cmd")
    fi
  done

  if ((${#missing[@]} > 0)); then
    echo "Missing required command(s): ${missing[*]}" >&2
    exit 127
  fi
}

ensure_label() {
  local name="$1"
  local color="$2"
  local description="$3"

  if gh label view "$name" --repo "$REPO" >/dev/null 2>&1; then
    return 0
  fi

  gh label create "$name" \
    --repo "$REPO" \
    --color "$color" \
    --description "$description" >/dev/null 2>&1 || true
}

ensure_labels() {
  ensure_label "$PENDING_LABEL" "0e8a16" "Tasks waiting for the server Codex runner"
  ensure_label "$RUNNING_LABEL" "1d76db" "Task is being processed by the server Codex runner"
  ensure_label "$DONE_LABEL" "5319e7" "Task completed by the server Codex runner"
  ensure_label "$FAILED_LABEL" "d73a4a" "Task failed in the server Codex runner"
  ensure_label "$NOCHANGE_LABEL" "cfd3d7" "Task produced no code changes"
  ensure_label "$NEEDS_REVIEW_LABEL" "fbca04" "Pull request needs human review"
}

git_is_clean() {
  [[ -z "$(git status --porcelain --untracked-files=normal)" ]]
}

stage_allowed_changes() {
  git add -A -- . \
    ":(exclude).env" \
    ":(exclude)config.toml" \
    ":(exclude)data/cache/**" \
    ":(exclude)outputs/**" \
    ":(exclude)site/**" \
    ":(exclude)dist/**" \
    ":(exclude)logs/**" \
    ":(exclude).venv/**" \
    ":(exclude).pytest_cache/**" \
    ":(exclude).idea/**" \
    ":(exclude)*.log"
}

sanitize_for_comment() {
  sed 's/```/` ` `/g'
}

log_tail_for_comment() {
  if [[ -n "${LOG_FILE:-}" && -f "$LOG_FILE" ]]; then
    tail -n "$COMMENT_LOG_TAIL_LINES" "$LOG_FILE" | sanitize_for_comment
  fi
}

comment_issue() {
  local body="$1"
  gh issue comment "$ISSUE_NUMBER" --repo "$REPO" --body "$body" >/dev/null || true
}

set_issue_label_state() {
  local add_label="$1"
  gh issue edit "$ISSUE_NUMBER" \
    --repo "$REPO" \
    --remove-label "$RUNNING_LABEL" \
    --add-label "$add_label" >/dev/null || true
}

save_failure_branch_if_needed() {
  if git_is_clean; then
    return 0
  fi

  stage_allowed_changes

  if git diff --cached --quiet; then
    return 0
  fi

  git commit -m "WIP issue #${ISSUE_NUMBER}: runner failure" >/dev/null
  git push origin "$BRANCH" >/dev/null
}

mark_failed() {
  local reason="$1"
  local detail=""
  local log_tail=""

  save_failure_branch_if_needed || true
  log_tail="$(log_tail_for_comment || true)"

  if [[ -n "$log_tail" ]]; then
    detail="

最近日志：

\`\`\`text
${log_tail}
\`\`\`"
  fi

  comment_issue "Codex runner 处理 Issue #${ISSUE_NUMBER} 失败。

原因：${reason}
分支：\`${BRANCH:-未创建}\`
服务器日志：\`${LOG_FILE:-未创建}\`${detail}"

  set_issue_label_state "$FAILED_LABEL"
}

on_exit() {
  local status=$?
  if [[ "$status" -ne 0 && "$HANDLED_EXIT" -eq 0 && -n "${ISSUE_NUMBER:-}" ]]; then
    mark_failed "runner 异常退出，exit code ${status}"
  fi
}
trap on_exit EXIT

slugify() {
  tr '[:upper:]' '[:lower:]' \
    | tr -cs 'a-z0-9' '-' \
    | sed 's/^-//;s/-$//' \
    | cut -c1-40
}

build_prompt() {
  local prompt_file="$1"
  local issue_body="$2"

  cat >"$prompt_file" <<EOF
你正在修改 GitHub 仓库 ${REPO}。

基础分支：${BASE_BRANCH}
任务 Issue：#${ISSUE_NUMBER}
标题：${ISSUE_TITLE}

请严格按下面任务说明修改代码。

要求：
1. 只修改与任务相关的文件。
2. 不要提交 .env、密钥、缓存、输出报告、历史行情数据。
3. 修改后运行测试；如果测试失败，继续修复，直到测试通过或明确说明失败原因。
4. 提交信息要简洁，说明本次修改点。
5. 最终回复必须包含修改摘要、测试结果、风险或遗留问题。

任务说明：
${issue_body}
EOF
}

main() {
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "Another codex issue runner is already running."
    exit 0
  fi

  require_cmds

  cd "$REPO_DIR"
  mkdir -p "$LOG_DIR"

  gh auth status --hostname github.com >/dev/null
  ensure_labels

  if ! git_is_clean; then
    echo "Worktree is dirty. Refusing to run automated issue task." >&2
    git status --short >&2
    exit 1
  fi

  ISSUE_NUMBER="$(gh issue list \
    --repo "$REPO" \
    --label "$PENDING_LABEL" \
    --state open \
    --limit 1 \
    --json number \
    --jq '.[0].number // empty')"

  if [[ -z "$ISSUE_NUMBER" ]]; then
    echo "No ${PENDING_LABEL} issue."
    HANDLED_EXIT=1
    exit 0
  fi

  ISSUE_TITLE="$(gh issue view "$ISSUE_NUMBER" --repo "$REPO" --json title --jq '.title')"
  local issue_body
  issue_body="$(gh issue view "$ISSUE_NUMBER" --repo "$REPO" --json body --jq '.body // ""')"

  BRANCH="codex/issue-${ISSUE_NUMBER}-${RUN_ID}-$(printf '%s' "$ISSUE_TITLE" | slugify)"
  LOG_FILE="$LOG_DIR/issue-${ISSUE_NUMBER}-${RUN_ID}.log"

  echo "Processing issue #${ISSUE_NUMBER}: ${ISSUE_TITLE}"
  echo "Log file: ${LOG_FILE}"

  gh issue edit "$ISSUE_NUMBER" \
    --repo "$REPO" \
    --remove-label "$PENDING_LABEL" \
    --add-label "$RUNNING_LABEL" >/dev/null

  git fetch origin
  git checkout "$BASE_BRANCH"
  git pull --ff-only origin "$BASE_BRANCH"
  git checkout -b "$BRANCH"

  local prompt_file
  prompt_file="$(mktemp)"
  build_prompt "$prompt_file" "$issue_body"

  set +e
  codex exec "$CODEX_BYPASS_FLAG" --cd "$REPO_DIR" <"$prompt_file" 2>&1 | tee "$LOG_FILE"
  local codex_status=${PIPESTATUS[0]}
  set -e
  rm -f "$prompt_file"

  if [[ "$codex_status" -ne 0 ]]; then
    mark_failed "codex exec 失败，exit code ${codex_status}"
    HANDLED_EXIT=1
    exit "$codex_status"
  fi

  if git_is_clean; then
    comment_issue "Codex runner 已处理 Issue #${ISSUE_NUMBER}，但未产生代码变更。"
    set_issue_label_state "$NOCHANGE_LABEL"
    HANDLED_EXIT=1
    exit 0
  fi

  set +e
  bash -lc "$TEST_CMD" 2>&1 | tee -a "$LOG_FILE"
  local test_status=${PIPESTATUS[0]}
  set -e

  if [[ "$test_status" -ne 0 ]]; then
    mark_failed "测试失败：\`${TEST_CMD}\` exit code ${test_status}"
    HANDLED_EXIT=1
    exit "$test_status"
  fi

  stage_allowed_changes

  if git diff --cached --quiet; then
    comment_issue "Codex runner 已处理 Issue #${ISSUE_NUMBER}，但没有可提交的代码变更。"
    set_issue_label_state "$NOCHANGE_LABEL"
    HANDLED_EXIT=1
    exit 0
  fi

  git commit -m "Implement issue #${ISSUE_NUMBER}: ${ISSUE_TITLE}"
  git push origin "$BRANCH"

  local pr_body
  pr_body="$(cat <<EOF
Closes #${ISSUE_NUMBER}

## Source
Generated from issue #${ISSUE_NUMBER}.

## Test
- \`${TEST_CMD}\`: passed

## Review focus
Please review strategy correctness, signal classification, backtest assumptions, and data-source edge cases.
EOF
)"

  local pr_url
  pr_url="$(gh pr create \
    --repo "$REPO" \
    --base "$BASE_BRANCH" \
    --head "$BRANCH" \
    --title "Codex: ${ISSUE_TITLE}" \
    --body "$pr_body")"

  gh pr edit "$pr_url" --repo "$REPO" --add-label "$NEEDS_REVIEW_LABEL" >/dev/null || true

  comment_issue "Codex runner 已完成 Issue #${ISSUE_NUMBER} 并创建 PR：
${pr_url}

测试结果：
- \`${TEST_CMD}\`: passed

分支：\`${BRANCH}\`
服务器日志：\`${LOG_FILE}\`"

  set_issue_label_state "$DONE_LABEL"

  git checkout "$BASE_BRANCH" >/dev/null
  echo "Done: ${pr_url}"
  HANDLED_EXIT=1
}

main "$@"
