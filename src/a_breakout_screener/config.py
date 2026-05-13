from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ScreenerConfig:
    top_n: int = 30
    max_workers: int = 6
    history_days: int = 900
    resistance_lookback_weeks: int = 52
    resistance_exclude_recent_weeks: int = 4
    min_history_rows: int = 160
    breakout_buffer: float = 0.0
    effective_breakout_pct: float = 0.02
    max_extension: float = 0.12
    strong_volume_ratio: float = 1.8
    min_amount: float = 80_000_000
    min_price: float = 3.0
    allowed_prefixes: tuple[str, ...] = ("00", "30", "60", "68")
    exclude_name_keywords: tuple[str, ...] = ("ST", "*ST", "退")
    market_regime: str = "all"
    market_index: str = "000300"
    confirmation_bars: int = 0
    ma_trend_period: int = 0
    max_open_gap_pct: float = 0.0
    consolidation_weeks: int = 0
    consolidation_max_span: float = 0.0
    consolidation_penalty_weight: float = 0.0


@dataclass(frozen=True)
class PathsConfig:
    cache_dir: Path = Path("data/cache")
    output_dir: Path = Path("outputs")


@dataclass(frozen=True)
class NetworkConfig:
    disable_system_proxy: bool = True


@dataclass(frozen=True)
class EmailConfig:
    enabled: bool = False
    method: str = "smtp"
    smtp_host: str = ""
    smtp_port: int = 465
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_ssl: bool = True
    smtp_starttls: bool = False
    mail_from: str = ""
    mail_to: tuple[str, ...] = field(default_factory=tuple)
    subject_prefix: str = "A股突破选股"

    def missing_fields(self) -> list[str]:
        if self.method in {"mail", "codex_gmail", "gmail"}:
            return [] if self.mail_to else ["mail_to"]
        missing: list[str] = []
        for field_name in ("smtp_host", "smtp_user", "smtp_password", "mail_from"):
            if not getattr(self, field_name):
                missing.append(field_name)
        if not self.mail_to:
            missing.append("mail_to")
        return missing


@dataclass(frozen=True)
class AppConfig:
    screener: ScreenerConfig = field(default_factory=ScreenerConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    email: EmailConfig = field(default_factory=EmailConfig)


def load_config(config_path: Path | None = None) -> AppConfig:
    raw: dict[str, Any] = {}
    base_dir = Path.cwd()
    if config_path:
        config_path = config_path.expanduser().resolve()
        base_dir = config_path.parent
        if config_path.exists():
            with config_path.open("rb") as fp:
                raw = tomllib.load(fp)

    paths_raw = raw.get("paths", {})
    paths = PathsConfig(
        cache_dir=_resolve_path(paths_raw.get("cache_dir", "data/cache"), base_dir),
        output_dir=_resolve_path(paths_raw.get("output_dir", "outputs"), base_dir),
    )

    screener_raw = raw.get("screener", {})
    screener = ScreenerConfig(
        top_n=int(screener_raw.get("top_n", 30)),
        max_workers=int(screener_raw.get("max_workers", 6)),
        history_days=int(screener_raw.get("history_days", 900)),
        resistance_lookback_weeks=int(screener_raw.get("resistance_lookback_weeks", 52)),
        resistance_exclude_recent_weeks=int(screener_raw.get("resistance_exclude_recent_weeks", 4)),
        min_history_rows=int(screener_raw.get("min_history_rows", 160)),
        breakout_buffer=float(screener_raw.get("breakout_buffer", 0.0)),
        effective_breakout_pct=float(screener_raw.get("effective_breakout_pct", 0.02)),
        max_extension=float(screener_raw.get("max_extension", 0.12)),
        strong_volume_ratio=float(screener_raw.get("strong_volume_ratio", 1.8)),
        min_amount=float(screener_raw.get("min_amount", 80_000_000)),
        min_price=float(screener_raw.get("min_price", 3.0)),
        allowed_prefixes=tuple(screener_raw.get("allowed_prefixes", ("00", "30", "60", "68"))),
        exclude_name_keywords=tuple(screener_raw.get("exclude_name_keywords", ("ST", "*ST", "退"))),
        market_regime=str(screener_raw.get("market_regime", "all")),
        market_index=str(screener_raw.get("market_index", "000300")),
        confirmation_bars=int(screener_raw.get("confirmation_bars", 0)),
        ma_trend_period=int(screener_raw.get("ma_trend_period", 0)),
        max_open_gap_pct=float(screener_raw.get("max_open_gap_pct", 0.0)),
        consolidation_weeks=int(screener_raw.get("consolidation_weeks", 0)),
        consolidation_max_span=float(screener_raw.get("consolidation_max_span", 0.0)),
        consolidation_penalty_weight=float(screener_raw.get("consolidation_penalty_weight", 0.0)),
    )

    network_raw = raw.get("network", {})
    network = NetworkConfig(
        disable_system_proxy=_env_bool(
            "A_BREAKOUT_DISABLE_SYSTEM_PROXY",
            bool(network_raw.get("disable_system_proxy", True)),
        )
    )

    email_raw = raw.get("email", {})
    email = EmailConfig(
        enabled=_env_bool("A_BREAKOUT_EMAIL_ENABLED", bool(email_raw.get("enabled", False))),
        method=_env("A_BREAKOUT_EMAIL_METHOD", str(email_raw.get("method", "smtp"))).lower(),
        smtp_host=_env("SMTP_HOST", str(email_raw.get("smtp_host", ""))),
        smtp_port=int(_env("SMTP_PORT", str(email_raw.get("smtp_port", 465))) or 465),
        smtp_user=_env("SMTP_USER", str(email_raw.get("smtp_user", ""))),
        smtp_password=_env("SMTP_PASSWORD", str(email_raw.get("smtp_password", ""))),
        smtp_ssl=_env_bool("SMTP_SSL", bool(email_raw.get("smtp_ssl", True))),
        smtp_starttls=_env_bool("SMTP_STARTTLS", bool(email_raw.get("smtp_starttls", False))),
        mail_from=_env("MAIL_FROM", str(email_raw.get("mail_from", ""))),
        mail_to=_as_tuple(_env("MAIL_TO", email_raw.get("mail_to", []))),
        subject_prefix=str(email_raw.get("subject_prefix", "A股突破选股")),
    )

    return AppConfig(screener=screener, paths=paths, network=network, email=email)


def _resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _env(name: str, default: Any) -> Any:
    value = os.getenv(name)
    return default if value is None else value


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()
