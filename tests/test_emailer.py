from __future__ import annotations

import subprocess

from a_breakout_screener.config import EmailConfig
from a_breakout_screener.emailer import send_report, validate_email_transport


def test_codex_gmail_method_invokes_codex(monkeypatch) -> None:
    calls = {}

    def fake_which(name: str) -> str | None:
        return "/usr/local/bin/codex" if name == "codex" else None

    def fake_run(cmd, **kwargs):
        calls["cmd"] = cmd
        calls["kwargs"] = kwargs
        return subprocess.CompletedProcess(cmd, 0, stdout="mcp: codex_apps/gmail_send_email (completed)\n", stderr="")

    monkeypatch.setattr("a_breakout_screener.emailer.shutil.which", fake_which)
    monkeypatch.setattr("a_breakout_screener.emailer.subprocess.run", fake_run)

    send_report(
        EmailConfig(enabled=True, method="codex_gmail", mail_to=("user@example.com",)),
        "测试主题",
        "测试正文",
    )

    cmd = calls["cmd"]
    assert cmd[:2] == ["/usr/local/bin/codex", "exec"]
    assert "--dangerously-bypass-approvals-and-sandbox" in cmd
    assert "user@example.com" in cmd[-1]
    assert "测试主题" in cmd[-1]


def test_validate_codex_gmail_transport_requires_codex(monkeypatch) -> None:
    monkeypatch.setattr("a_breakout_screener.emailer.shutil.which", lambda name: None)

    problems = validate_email_transport(
        EmailConfig(enabled=True, method="codex_gmail", mail_to=("user@example.com",)),
    )

    assert problems == ["codex"]
