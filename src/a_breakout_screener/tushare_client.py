from __future__ import annotations

import os
from functools import lru_cache
from typing import Any


DEFAULT_TUSHARE_HTTP_URL = "http://118.89.66.41:8010/"


def create_tushare_pro(token: str | None = None, http_url: str | None = None) -> Any:
    """Create the configured Tushare pro client used by this project."""
    ts = _tushare()
    resolved_token = token or os.getenv("TUSHARE_TOKEN") or os.getenv("TUSHARE_PRO_TOKEN")
    if not resolved_token:
        raise RuntimeError("TUSHARE_TOKEN is not configured. Put it in .env or export it before running.")

    timeout = float(os.getenv("TUSHARE_TIMEOUT", "10"))
    pro = ts.pro_api(resolved_token, timeout=timeout)
    # Required for the configured token route. Without this URL Tushare may report an invalid token.
    pro._DataApi__http_url = http_url or os.getenv("TUSHARE_HTTP_URL", DEFAULT_TUSHARE_HTTP_URL)
    return pro


@lru_cache(maxsize=1)
def get_tushare_pro() -> Any:
    return create_tushare_pro()


def pro_bar(*, api: Any | None = None, **kwargs: Any) -> Any:
    ts = _tushare()
    return ts.pro_bar(api=api or create_tushare_pro(), **kwargs)


def _tushare() -> Any:
    try:
        import tushare as ts
    except ImportError as exc:  # pragma: no cover - exercised by environment setup
        raise RuntimeError("tushare is not installed. Run `uv sync` before using Tushare data.") from exc
    return ts
