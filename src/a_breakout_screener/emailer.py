from __future__ import annotations

import mimetypes
import shutil
import subprocess
import smtplib
from email.message import EmailMessage
from pathlib import Path
import textwrap

from .config import EmailConfig


def send_report(
    email_config: EmailConfig,
    subject: str,
    body: str,
    attachments: list[Path] | None = None,
) -> None:
    missing = email_config.missing_fields()
    if missing:
        raise RuntimeError(f"email config is incomplete: {', '.join(missing)}")

    if email_config.method == "mail":
        _send_with_local_mail(email_config, subject, body, attachments or [])
        return
    if email_config.method in {"codex_gmail", "gmail"}:
        _send_with_codex_gmail(email_config, subject, body, attachments or [])
        return
    if email_config.method != "smtp":
        raise RuntimeError(f"unsupported email method: {email_config.method}")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = email_config.mail_from
    msg["To"] = ", ".join(email_config.mail_to)
    msg.set_content(body)

    for attachment in attachments or []:
        _attach_file(msg, attachment)

    if email_config.smtp_ssl:
        with smtplib.SMTP_SSL(email_config.smtp_host, email_config.smtp_port, timeout=30) as smtp:
            _login_and_send(smtp, email_config, msg)
    else:
        with smtplib.SMTP(email_config.smtp_host, email_config.smtp_port, timeout=30) as smtp:
            if email_config.smtp_starttls:
                smtp.starttls()
            _login_and_send(smtp, email_config, msg)


def _login_and_send(smtp: smtplib.SMTP, email_config: EmailConfig, msg: EmailMessage) -> None:
    smtp.login(email_config.smtp_user, email_config.smtp_password)
    smtp.send_message(msg)


def _attach_file(msg: EmailMessage, path: Path) -> None:
    if not path.exists():
        return
    mime_type, _ = mimetypes.guess_type(path.name)
    maintype, subtype = (mime_type or "application/octet-stream").split("/", 1)
    msg.add_attachment(path.read_bytes(), maintype=maintype, subtype=subtype, filename=path.name)


def _send_with_local_mail(email_config: EmailConfig, subject: str, body: str, attachments: list[Path]) -> None:
    sendmail = shutil.which("sendmail") or "/usr/sbin/sendmail"
    if attachments and Path(sendmail).exists():
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = email_config.mail_from or "a-breakout-screener@localhost"
        msg["To"] = ", ".join(email_config.mail_to)
        msg.set_content(body)
        for attachment in attachments:
            _attach_file(msg, attachment)
        subprocess.run([sendmail, "-t"], input=msg.as_bytes(), check=True)
        return

    subprocess.run(
        ["mail", "-s", subject, *email_config.mail_to],
        input=_body_with_attachment_paths(body, attachments),
        text=True,
        check=True,
    )


def _send_with_codex_gmail(email_config: EmailConfig, subject: str, body: str, attachments: list[Path]) -> None:
    codex = shutil.which("codex")
    if not codex:
        raise RuntimeError("codex CLI is not installed or not on PATH")

    prompt = _codex_gmail_prompt(email_config, subject, body, attachments)
    result = subprocess.run(
        [
            codex,
            "exec",
            "--skip-git-repo-check",
            "-C",
            str(Path.cwd()),
            "--dangerously-bypass-approvals-and-sandbox",
            prompt,
        ],
        text=True,
        capture_output=True,
        timeout=240,
    )
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    if result.returncode != 0:
        raise RuntimeError(f"codex gmail send failed with exit code {result.returncode}: {_tail(output)}")
    if "gmail_send_email (completed)" not in output or "gmail_send_email (failed)" in output or "发送失败" in output:
        raise RuntimeError(f"codex gmail send did not report success: {_tail(output)}")
    print(_tail(output, lines=12))


def _codex_gmail_prompt(email_config: EmailConfig, subject: str, body: str, attachments: list[Path]) -> str:
    recipient_text = ", ".join(email_config.mail_to)
    attachment_note = _body_with_attachment_paths("", attachments).strip()
    full_body = body.rstrip()
    if attachment_note:
        full_body = f"{full_body}\n\n{attachment_note}"
    return textwrap.dedent(
        f"""
        请使用 Gmail 插件发送一封邮件。
        收件人: {recipient_text}
        主题: {subject}

        正文如下，请保持正文内容，不要改写事实，不要修改任何文件，只发送这一封邮件：
        ---
        {full_body}
        ---

        发送后请报告 Gmail message id；如果 Gmail 插件或连接器不可用，请明确失败原因。
        """
    ).strip()


def _tail(text: str, lines: int = 20) -> str:
    return "\n".join(text.strip().splitlines()[-lines:])


def _body_with_attachment_paths(body: str, attachments: list[Path]) -> str:
    if not attachments:
        return body
    lines = [body.rstrip(), "", "附件文件路径:"]
    lines.extend(str(path) for path in attachments)
    return "\n".join(lines) + "\n"
