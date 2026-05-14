#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-/opt/a-breakout-screener}"
REPO="${REPO:-AutumnFarmer/breakout-strategy}"
BASE_BRANCH="${BASE_BRANCH:-optimize/breakout-strategy}"

NEEDS_REVIEW_LABEL="${NEEDS_REVIEW_LABEL:-needs-review}"
REVIEWING_LABEL="${REVIEWING_LABEL:-codex-reviewing}"
REVIEWED_LABEL="${REVIEWED_LABEL:-codex-reviewed}"
REVIEW_FAILED_LABEL="${REVIEW_FAILED_LABEL:-codex-review-failed}"
HUMAN_REVIEWED_LABEL="${HUMAN_REVIEWED_LABEL:-human-reviewed}"

LOG_DIR="${LOG_DIR:-$REPO_DIR/logs/codex_pr_reviewer}"
LOCK_FILE="${LOCK_FILE:-/tmp/a_breakout_screener_codex_pr_reviewer.lock}"
COMMENT_LOG_TAIL_LINES="${COMMENT_LOG_TAIL_LINES:-80}"
MAX_DIFF_BYTES="${MAX_DIFF_BYTES:-180000}"
MAX_REVIEW_COMMENT_BYTES="${MAX_REVIEW_COMMENT_BYTES:-55000}"
RUN_ID="$(date -u +%Y%m%d%H%M%S)"

PR_NUMBER=""
LOG_FILE=""
REVIEW_FILE=""
HANDLED_EXIT=0

require_cmds() {
  local missing=()
  local cmd

  for cmd in git gh codex flock; do
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
  ensure_label "$NEEDS_REVIEW_LABEL" "fbca04" "Pull request needs review"
  ensure_label "$REVIEWING_LABEL" "1d76db" "Pull request is being reviewed by Codex"
  ensure_label "$REVIEWED_LABEL" "0e8a16" "Pull request reviewed by Codex"
  ensure_label "$REVIEW_FAILED_LABEL" "d73a4a" "Codex PR review failed"
  ensure_label "$HUMAN_REVIEWED_LABEL" "5319e7" "Pull request reviewed by a human"
}

sanitize_for_comment() {
  sed 's/```/` ` `/g'
}

log_tail_for_comment() {
  if [[ -n "${LOG_FILE:-}" && -f "$LOG_FILE" ]]; then
    tail -n "$COMMENT_LOG_TAIL_LINES" "$LOG_FILE" | sanitize_for_comment
  fi
}

comment_pr_from_file() {
  local body_file="$1"
  gh pr comment "$PR_NUMBER" --repo "$REPO" --body-file "$body_file" >/dev/null || true
}

set_pr_label_state() {
  local add_label="$1"
  gh pr edit "$PR_NUMBER" \
    --repo "$REPO" \
    --remove-label "$REVIEWING_LABEL" \
    --add-label "$add_label" >/dev/null || true
}

mark_review_failed() {
  local reason="$1"
  local comment_file
  local log_tail=""

  comment_file="$(mktemp)"
  log_tail="$(log_tail_for_comment || true)"

  {
    echo "## Codex 自动 Review 失败"
    echo
    echo "原因：${reason}"
    echo
    echo "日志：\`${LOG_FILE:-未创建}\`"
    if [[ -n "$log_tail" ]]; then
      echo
      echo "最近日志："
      echo
      echo '```text'
      echo "$log_tail"
      echo '```'
    fi
  } >"$comment_file"

  comment_pr_from_file "$comment_file"
  rm -f "$comment_file"
  set_pr_label_state "$REVIEW_FAILED_LABEL"
}

on_exit() {
  local status=$?
  if [[ "$status" -ne 0 && "$HANDLED_EXIT" -eq 0 && -n "${PR_NUMBER:-}" ]]; then
    mark_review_failed "review runner 异常退出，exit code ${status}"
  fi
}
trap on_exit EXIT

write_prompt() {
  local prompt_file="$1"
  local pr_info="$2"
  local changed_files="$3"
  local diff_file="$4"
  local diff_bytes="$5"

  {
    cat <<EOF
你是 A 股突破策略仓库的代码审查员。请审查 GitHub PR #${PR_NUMBER}。

仓库：${REPO}
基础分支：${BASE_BRANCH}

PR 信息：
${pr_info}

变更文件：
${changed_files}

请基于下面 diff 做严格 review。重点不是泛泛代码风格，而是策略正确性、实盘风险、自动化可靠性和测试覆盖。

Review 重点：
1. 是否会误把未完成周线当成 A 类周线确认。
2. 成交额/成交量口径是否正确。
3. 压力区计算是否仍然过宽或过窄。
4. A/B/C1/C2/D 分类是否符合策略定义。
5. 是否引入回测假设错误，例如小数股、无滑点、无限买入。
6. 是否破坏报告输出、CSV/Excel/HTML、邮件正文。
7. 是否缺少关键测试。
8. 是否有数据源失败、缓存过期、Tushare/AkShare fallback 的边界问题。
9. 是否改动过大、触及无关模块。
10. 是否有安全问题，例如提交密钥、读取本地敏感文件、暴露 token。

输出 Markdown，结构固定：

## Review 结论

必须从以下三种选一种：
- APPROVE
- REQUEST_CHANGES
- COMMENT

## 主要问题

按严重程度列出：
- Critical
- Major
- Minor

## 策略逻辑复核

逐条说明是否符合突破策略。

## 测试建议

列出应该补充的测试。

## 建议修改

给出具体文件和修改方向。

EOF

    if ((diff_bytes > MAX_DIFF_BYTES)); then
      echo "注意：PR diff 原始大小为 ${diff_bytes} bytes，下面只包含前 ${MAX_DIFF_BYTES} bytes。请在 Review 结论中明确说明 diff 被截断，自动 review 只能作为初筛。"
      echo
    fi

    echo "下面是 PR diff："
    echo
    echo '```diff'
    if ((diff_bytes > MAX_DIFF_BYTES)); then
      head -c "$MAX_DIFF_BYTES" "$diff_file"
      echo
      echo "... diff truncated ..."
    else
      cat "$diff_file"
    fi
    echo
    echo '```'
  } >"$prompt_file"
}

write_success_comment() {
  local comment_file
  local review_bytes

  comment_file="$(mktemp)"
  review_bytes="$(wc -c <"$REVIEW_FILE" | tr -d ' ')"

  {
    echo "## Codex 自动 Review"
    echo
    if ((review_bytes > MAX_REVIEW_COMMENT_BYTES)); then
      head -c "$MAX_REVIEW_COMMENT_BYTES" "$REVIEW_FILE"
      echo
      echo
      echo "_Review 内容过长，已截断。完整日志：\`${LOG_FILE}\`_"
    else
      cat "$REVIEW_FILE"
      echo
      echo
      echo "---"
      echo "日志：\`${LOG_FILE}\`"
    fi
  } >"$comment_file"

  comment_pr_from_file "$comment_file"
  rm -f "$comment_file"
}

main() {
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "Another codex PR reviewer is already running."
    exit 0
  fi

  require_cmds

  cd "$REPO_DIR"
  mkdir -p "$LOG_DIR"

  gh auth status --hostname github.com >/dev/null
  ensure_labels

  PR_NUMBER="$(gh pr list \
    --repo "$REPO" \
    --label "$NEEDS_REVIEW_LABEL" \
    --state open \
    --limit 10 \
    --json number,isDraft \
    --jq '[.[] | select(.isDraft == false)][0].number // empty')"

  if [[ -z "$PR_NUMBER" ]]; then
    echo "No open non-draft PR with label ${NEEDS_REVIEW_LABEL}."
    HANDLED_EXIT=1
    exit 0
  fi

  LOG_FILE="$LOG_DIR/pr-${PR_NUMBER}-${RUN_ID}.log"
  REVIEW_FILE="$LOG_DIR/pr-${PR_NUMBER}-${RUN_ID}.review.md"

  echo "Reviewing PR #${PR_NUMBER}"
  echo "Log file: ${LOG_FILE}"

  gh pr edit "$PR_NUMBER" \
    --repo "$REPO" \
    --remove-label "$NEEDS_REVIEW_LABEL" \
    --add-label "$REVIEWING_LABEL" >/dev/null || true

  git fetch origin "$BASE_BRANCH" >/dev/null 2>&1 || true

  local pr_info
  local changed_files
  local prompt_file
  local diff_file
  local diff_bytes

  pr_info="$(gh pr view "$PR_NUMBER" \
    --repo "$REPO" \
    --json title,body,author,baseRefName,headRefName,url,mergeable,reviewDecision \
    --jq '.')"

  changed_files="$(gh pr view "$PR_NUMBER" \
    --repo "$REPO" \
    --json files \
    --jq '.files[].path')"

  prompt_file="$(mktemp)"
  diff_file="$(mktemp)"

  gh pr diff "$PR_NUMBER" --repo "$REPO" >"$diff_file"
  diff_bytes="$(wc -c <"$diff_file" | tr -d ' ')"
  write_prompt "$prompt_file" "$pr_info" "$changed_files" "$diff_file" "$diff_bytes"

  set +e
  codex exec \
    --color never \
    --sandbox read-only \
    --cd "$REPO_DIR" \
    --output-last-message "$REVIEW_FILE" \
    <"$prompt_file" 2>&1 | tee "$LOG_FILE"
  local review_status=${PIPESTATUS[0]}
  set -e

  rm -f "$prompt_file" "$diff_file"

  if [[ "$review_status" -ne 0 ]]; then
    mark_review_failed "codex exec review 失败，exit code ${review_status}"
    HANDLED_EXIT=1
    exit "$review_status"
  fi

  if [[ ! -s "$REVIEW_FILE" ]]; then
    tail -n "$COMMENT_LOG_TAIL_LINES" "$LOG_FILE" >"$REVIEW_FILE"
  fi

  write_success_comment

  gh pr edit "$PR_NUMBER" \
    --repo "$REPO" \
    --remove-label "$REVIEWING_LABEL" \
    --add-label "$REVIEWED_LABEL" >/dev/null || true

  echo "Reviewed PR #${PR_NUMBER}"
  HANDLED_EXIT=1
}

main "$@"
