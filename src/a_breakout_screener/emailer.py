from __future__ import annotations

import mimetypes
import shutil
import subprocess
import smtplib
from email.message import EmailMessage
from pathlib import Path

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


def _body_with_attachment_paths(body: str, attachments: list[Path]) -> str:
    if not attachments:
        return body
    lines = [body.rstrip(), "", "附件文件路径:"]
    lines.extend(str(path) for path in attachments)
    return "\n".join(lines) + "\n"
