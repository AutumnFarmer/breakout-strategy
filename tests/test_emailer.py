from __future__ import annotations

import subprocess

import pytest

from a_breakout_screener.config import EmailConfig
from a_breakout_screener.emailer import send_report


def test_codex_gmail_method_invokes_codex_with_attachments(monkeypatch, tmp_path) -> None:
    calls = {}
    attachment = tmp_path / "report.csv"
    attachment.write_text("code,name\n000001,平安银行\n", encoding="utf-8")

    def fake_which(name: str) -> str | None:
        return "/usr/local/bin/codex" if name == "codex" else None

    def fake_run(cmd, **kwargs):
        calls["cmd"] = cmd
        calls["kwargs"] = kwargs
        return subprocess.CompletedProcess(cmd, 0, stdout="mcp: codex_apps/gmail_send_email (completed)\n", stderr="")

    monkeypatch.setattr("a_breakout_screener.emailer.shutil.which", fake_which)
    monkeypatch.setattr("a_breakout_screener.emailer.subprocess.run", fake_run)

    send_report(
        EmailConfig(enabled=True, method="codex_gmail", mail_to=("zhangmc895@gmail.com",)),
        "测试主题",
        "测试正文",
        [attachment],
    )

    cmd = calls["cmd"]
    assert cmd[:2] == ["/usr/local/bin/codex", "exec"]
    assert "--dangerously-bypass-approvals-and-sandbox" not in cmd
    assert cmd[-1] == "-"
    assert calls["kwargs"]["input"]
    assert "attachment_files" in calls["kwargs"]["input"]
    assert str(attachment.resolve()) in calls["kwargs"]["input"]
    assert "zhangmc895@gmail.com" in calls["kwargs"]["input"]
    assert "测试主题" in calls["kwargs"]["input"]


def test_send_report_rejects_missing_attachment(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="email attachment does not exist"):
        send_report(
            EmailConfig(
                enabled=True,
                method="smtp",
                smtp_host="smtp.example.com",
                smtp_user="user",
                smtp_password="password",
                mail_to=("to@example.com",),
                mail_from="from@example.com",
            ),
            "subject",
            "body",
            [tmp_path / "missing.csv"],
        )
