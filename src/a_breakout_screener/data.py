from __future__ import annotations

import warnings
from datetime import date, timedelta
import json
import os
from pathlib import Path
import time

import pandas as pd


SPOT_COLUMN_MAP = {
    "代码": "code",
    "名称": "name",
    "最新价": "latest",
    "成交额": "amount",
    "成交量": "volume",
    "换手率": "turnover_rate",
    "量比": "volume_ratio",
    "涨跌幅": "pct_change",
}

LEGACY_SPOT_COLUMN_MAP = {
    "代码": "code",
    "名称": "name",
    "最新价": "latest",
    "成交额": "amount",
    "成交量": "volume",
    "涨跌幅": "pct_change",
}

HIST_COLUMN_MAP = {
    "日期": "date",
    "股票代码": "code",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "换手率": "turnover_rate",
    "涨跌幅": "pct_change",
}

MAX_CACHE_STALENESS_DAYS = 7
DEFAULT_TUSHARE_RETRIES = 3
DEFAULT_TUSHARE_RETRY_DELAY = 2.0
DEFAULT_TUSHARE_RETRY_MAX_DELAY = 15.0
DEFAULT_TAG_CACHE_DAYS = 30
DEFAULT_FINANCIAL_CACHE_DAYS = 30
MAX_STOCK_TAGS = 8
GENERIC_CONCEPT_TAGS = {
    "AB股",
    "AH股",
    "B股",
    "HS300_",
    "MSCI概念",
    "MSCI中国",
    "QFII重仓",
    "上证180_",
    "上证380",
    "中证500",
    "互联互通",
    "机构重仓",
    "标准普尔",
    "标准普尔概念",
    "沪股通",
    "深股通",
    "融资标的股",
    "融资融券",
    "融券标的股",
    "证金持股",
    "转融券标的",
}


def fetch_spot() -> pd.DataFrame:
    if _tushare_enabled():
        try:
            return _fetch_spot_from_tushare()
        except Exception as exc:  # pragma: no cover - external data source fallback path
            warnings.warn(f"Tushare spot fetch failed; falling back to AkShare: {exc}", RuntimeWarning)
    return _fetch_spot_from_akshare()


def _fetch_spot_from_tushare() -> pd.DataFrame:
    from .tushare_client import get_tushare_pro

    pro = get_tushare_pro()
    trade_date, daily = _latest_tushare_daily(pro, date.today())
    basic = _call_tushare(
        "stock_basic",
        lambda: pro.stock_basic(
            exchange="",
            list_status="L",
            fields="ts_code,symbol,name",
        ),
    )
    if daily is None or daily.empty:
        raise RuntimeError("Tushare daily returned no rows")
    if basic is None or basic.empty:
        raise RuntimeError("Tushare stock_basic returned no rows")

    df = daily.merge(basic, on="ts_code", how="left")
    df["code"] = df["ts_code"].astype(str).str.extract(r"(\d{6})", expand=False).str.zfill(6)
    df["name"] = df["name"].fillna(df["code"]).astype(str)
    df["latest"] = pd.to_numeric(df["close"], errors="coerce")
    # Tushare daily amount is reported in thousand yuan; normalize to yuan for local filters.
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce") * 1000
    df["volume"] = pd.to_numeric(df["vol"], errors="coerce")
    df["pct_change"] = pd.to_numeric(df.get("pct_chg"), errors="coerce")
    df["trade_date"] = trade_date
    return df[["code", "name", "latest", "amount", "volume", "pct_change", "trade_date"]].dropna(subset=["code"])


def _fetch_spot_from_akshare() -> pd.DataFrame:
    ak = _akshare()
    try:
        raw = ak.stock_zh_a_spot_em()
        df = raw.rename(columns=SPOT_COLUMN_MAP).copy()
    except Exception:
        raw = ak.stock_zh_a_spot()
        df = raw.rename(columns=LEGACY_SPOT_COLUMN_MAP).copy()
    required = {"code", "name", "latest", "amount"}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"AkShare spot data missing columns: {sorted(missing)}")
    df["code"] = df["code"].astype(str).str.extract(r"(\d{6})$", expand=False).str.zfill(6)
    df = df.dropna(subset=["code"])
    df["name"] = df["name"].astype(str)
    for col in ("latest", "amount", "volume", "turnover_rate", "volume_ratio", "pct_change"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def filter_spot_universe(
    spot: pd.DataFrame,
    allowed_prefixes: tuple[str, ...],
    exclude_name_keywords: tuple[str, ...],
    min_amount: float,
    min_price: float,
    symbols: set[str] | None = None,
) -> pd.DataFrame:
    df = spot.copy()
    if symbols:
        normalized = {symbol.zfill(6) for symbol in symbols}
        df = df[df["code"].isin(normalized)]
    else:
        df = df[df["code"].str.startswith(allowed_prefixes)]
        for keyword in exclude_name_keywords:
            df = df[~df["name"].str.contains(keyword, case=False, regex=False, na=False)]
        df = df[df["amount"].fillna(0) >= min_amount]
        df = df[df["latest"].fillna(0) >= min_price]
    return df.reset_index(drop=True)


def fetch_history(
    symbol: str,
    start_date: date,
    end_date: date,
    cache_dir: Path,
    force_refresh: bool = False,
) -> pd.DataFrame:
    symbol = symbol.zfill(6)
    cache_path = cache_dir / "hist" / f"{symbol}.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cached = _read_history_cache(cache_path)
    fetch_start = start_date
    if not force_refresh and not cached.empty:
        latest_cached = cached["date"].max().date()
        if latest_cached >= end_date:
            return _slice_history(cached, start_date, end_date)
        fetch_start = max(start_date, latest_cached - timedelta(days=14))

    try:
        fresh = _fetch_history_from_source(symbol, fetch_start, end_date)
    except Exception as exc:  # pragma: no cover - network/data-source fallback path
        if not cached.empty and _cache_is_fresh(cached, end_date):
            warnings.warn(f"{symbol} history fetch failed; using cached data: {exc}", RuntimeWarning)
            return _slice_history(cached, start_date, end_date)
        if not cached.empty:
            latest_cached = cached["date"].max().date()
            raise RuntimeError(
                f"{symbol} history fetch failed and cache is stale "
                f"(latest {latest_cached}, requested {end_date}): {exc}"
            ) from exc
        raise

    if cached.empty:
        merged = fresh
    else:
        merged = pd.concat([cached, fresh], ignore_index=True)
        merged = merged.drop_duplicates(subset=["date"], keep="last")
        merged = merged.sort_values("date").reset_index(drop=True)

    merged.to_csv(cache_path, index=False)
    return _slice_history(merged, start_date, end_date)


def prepare_history_cache(
    symbols: set[str],
    start_date: date,
    end_date: date,
    cache_dir: Path,
    force_refresh: bool = False,
) -> tuple[int, int]:
    if not _tushare_enabled() or not symbols:
        return 0, 0

    normalized_symbols = {symbol.zfill(6) for symbol in symbols}
    needs: dict[str, date] = {}
    for symbol in normalized_symbols:
        cache_path = cache_dir / "hist" / f"{symbol}.csv"
        cached = pd.DataFrame() if force_refresh else _read_history_cache(cache_path)
        if not cached.empty and cached["date"].max().date() >= end_date:
            continue
        if cached.empty:
            needs[symbol] = start_date
        else:
            needs[symbol] = max(start_date, cached["date"].max().date() - timedelta(days=14))

    if not needs:
        return 0, 0

    daily = _fetch_tushare_daily_range(min(needs.values()), end_date, cache_dir)
    if daily.empty:
        return len(needs), 0

    cache_root = cache_dir / "hist"
    cache_root.mkdir(parents=True, exist_ok=True)
    daily = daily[daily["code"].isin(needs)]
    written = 0
    rows = 0
    for symbol, fresh in daily.groupby("code"):
        fetch_start = pd.Timestamp(needs[symbol])
        fresh = fresh[fresh["date"] >= fetch_start]
        if fresh.empty:
            continue

        cache_path = cache_root / f"{symbol}.csv"
        cached = pd.DataFrame() if force_refresh else _read_history_cache(cache_path)
        if cached.empty:
            merged = fresh
        else:
            merged = pd.concat([cached, fresh], ignore_index=True)
            merged = merged.drop_duplicates(subset=["date"], keep="last")
            merged = merged.sort_values("date").reset_index(drop=True)
        merged.to_csv(cache_path, index=False)
        written += 1
        rows += len(fresh)

    return written, rows


def _fetch_history_from_source(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    if _tushare_enabled():
        try:
            return _fetch_history_from_tushare(symbol, start_date, end_date)
        except Exception as tushare_exc:  # pragma: no cover - external data source fallback path
            try:
                return _fetch_history_from_akshare(symbol, start_date, end_date)
            except Exception as akshare_exc:
                raise RuntimeError(
                    f"Tushare history failed: {tushare_exc}; AkShare history failed: {akshare_exc}"
                ) from akshare_exc
    return _fetch_history_from_akshare(symbol, start_date, end_date)


def _fetch_history_from_tushare(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    from .tushare_client import get_tushare_pro

    pro = get_tushare_pro()
    raw = _call_tushare(
        f"daily {symbol}",
        lambda: pro.daily(
            ts_code=_tushare_ts_code(symbol),
            start_date=start_date.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d"),
        ),
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"Tushare daily returned no rows for {symbol}")
    return _normalize_tushare_daily(raw)


def _fetch_history_from_akshare(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    ak = _akshare()
    start = start_date.strftime("%Y%m%d")
    end = end_date.strftime("%Y%m%d")
    try:
        raw = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start,
            end_date=end,
            adjust="qfq",
        )
        df = raw.rename(columns=HIST_COLUMN_MAP).copy()
    except Exception:
        raw = ak.stock_zh_a_hist_tx(
            symbol=_tx_symbol(symbol),
            start_date=start,
            end_date=end,
            adjust="qfq",
            timeout=20,
        )
        df = raw.copy()
        if "volume" not in df.columns and "amount" in df.columns:
            # Tencent names the trade-volume-like field as amount in this endpoint.
            df["volume"] = df["amount"]
        if "amount" not in df.columns:
            df["amount"] = 0
    required = {"date", "open", "close", "high", "low", "volume", "amount"}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"AkShare history data for {symbol} missing columns: {sorted(missing)}")
    return _normalize_history(df)


def _tx_symbol(symbol: str) -> str:
    symbol = symbol.zfill(6)
    if symbol.startswith(("60", "68", "90")):
        return f"sh{symbol}"
    return f"sz{symbol}"


def _tushare_ts_code(symbol: str) -> str:
    symbol = symbol.zfill(6)
    if symbol.startswith(("60", "68", "90")):
        return f"{symbol}.SH"
    return f"{symbol}.SZ"


def _latest_tushare_daily(pro, end_date: date) -> tuple[str, pd.DataFrame]:
    start_date = end_date - timedelta(days=15)
    calendar = _call_tushare(
        "trade_cal",
        lambda: pro.trade_cal(
            exchange="SSE",
            start_date=start_date.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d"),
            is_open="1",
        ),
    )
    if calendar is None or calendar.empty:
        raise RuntimeError("Tushare trade_cal returned no open trading days")
    trade_dates = sorted(str(item) for item in calendar["cal_date"].dropna().tolist())
    last_error: Exception | None = None
    for trade_date in reversed(trade_dates):
        try:
            daily = _call_tushare(f"daily {trade_date}", lambda: pro.daily(trade_date=trade_date))
        except Exception as exc:  # pragma: no cover - external data source variance
            last_error = exc
            warnings.warn(f"Tushare daily {trade_date} failed after retries: {exc}", RuntimeWarning)
            continue
        if daily is not None and not daily.empty:
            return trade_date, daily
    if last_error:
        raise RuntimeError(f"Tushare daily returned no rows for recent open trading days; last error: {last_error}")
    raise RuntimeError("Tushare daily returned no rows for recent open trading days")


def _fetch_tushare_daily_range(start_date: date, end_date: date, cache_dir: Path) -> pd.DataFrame:
    from .tushare_client import get_tushare_pro

    pro = get_tushare_pro()
    calendar = _call_tushare(
        "trade_cal",
        lambda: pro.trade_cal(
            exchange="SSE",
            start_date=start_date.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d"),
            is_open="1",
        ),
    )
    if calendar is None or calendar.empty:
        raise RuntimeError("Tushare trade_cal returned no open trading days")

    date_cache_dir = cache_dir / "tushare_daily"
    date_cache_dir.mkdir(parents=True, exist_ok=True)
    trade_dates = sorted(str(item) for item in calendar["cal_date"].dropna().tolist())
    frames: list[pd.DataFrame] = []
    total = len(trade_dates)
    for idx, trade_date in enumerate(trade_dates, start=1):
        cache_path = date_cache_dir / f"{trade_date}.csv"
        if cache_path.exists():
            daily = _read_history_cache(cache_path)
        else:
            daily = _fetch_tushare_daily_with_retry(pro, trade_date)
            if not daily.empty:
                daily.to_csv(cache_path, index=False)
            time.sleep(0.15)
        if daily is not None and not daily.empty:
            frames.append(daily)
        if idx == 1 or idx % 25 == 0 or idx == total:
            print(f"Tushare历史缓存: {idx}/{total} 个交易日", flush=True)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _fetch_tushare_daily_with_retry(pro, trade_date: str) -> pd.DataFrame:
    try:
        daily = _call_tushare(f"daily {trade_date}", lambda: pro.daily(trade_date=trade_date))
    except Exception as exc:  # pragma: no cover - external network variance
        warnings.warn(f"Tushare daily failed for {trade_date}; skipped: {exc}", RuntimeWarning)
        return pd.DataFrame()
    if daily is None or daily.empty:
        return pd.DataFrame()
    return _normalize_tushare_daily(daily)


def _normalize_tushare_daily(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns={"trade_date": "date", "vol": "volume"}).copy()
    df["code"] = df["ts_code"].astype(str).str.extract(r"(\d{6})", expand=False).str.zfill(6)
    if "amount" in df.columns:
        # Tushare daily amount is reported in thousand yuan; normalize to yuan.
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce") * 1000
    else:
        df["amount"] = 0
    required = {"date", "open", "close", "high", "low", "volume", "amount"}
    missing = required.difference(df.columns)
    if missing:
        raise RuntimeError(f"Tushare daily data missing columns: {sorted(missing)}")
    return _normalize_history(df)


def _normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized = normalized.dropna(subset=["date"])
    for col in ("open", "close", "high", "low", "volume", "amount", "turnover_rate", "pct_change"):
        if col in normalized.columns:
            normalized[col] = pd.to_numeric(normalized[col], errors="coerce")
    normalized = normalized.dropna(subset=["open", "close", "high", "low"])
    normalized = normalized.sort_values("date").reset_index(drop=True)
    return normalized


def _read_history_cache(cache_path: Path) -> pd.DataFrame:
    if not cache_path.exists():
        return pd.DataFrame()
    try:
        cached = pd.read_csv(cache_path)
        return _normalize_history(cached)
    except Exception as exc:  # pragma: no cover - corrupt local cache path
        warnings.warn(f"discarding invalid cache {cache_path}: {exc}", RuntimeWarning)
        return pd.DataFrame()


def _slice_history(df: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    return df[(df["date"] >= start_ts) & (df["date"] <= end_ts)].copy().reset_index(drop=True)


def _cache_is_fresh(cached: pd.DataFrame, end_date: date, max_age_days: int = MAX_CACHE_STALENESS_DAYS) -> bool:
    if cached.empty:
        return False
    latest_cached = cached["date"].max().date()
    return latest_cached >= end_date - timedelta(days=max_age_days)


def _akshare():
    try:
        import akshare as ak
    except ImportError as exc:  # pragma: no cover - tested through CLI doctor manually
        raise RuntimeError("akshare is not installed. Run `uv sync` or install project dependencies.") from exc
    return ak


def fetch_index_history(
    index_code: str,
    start_date: date,
    end_date: date,
    cache_dir: Path,
) -> pd.DataFrame:
    """Fetch daily bars for a market index (e.g. 000300 for CSI 300)."""
    index_code = index_code.zfill(6)
    cache_path = cache_dir / "hist" / f"index_{index_code}.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cached = _read_history_cache(cache_path)
    if not cached.empty and cached["date"].max().date() >= end_date:
        return _slice_history(cached, start_date, end_date)

    try:
        fresh = _fetch_index_from_source(index_code, start_date, end_date)
    except Exception as exc:  # pragma: no cover - external data-source fallback path
        if not cached.empty and _cache_is_fresh(cached, end_date):
            warnings.warn(f"{index_code} index fetch failed; using cached data: {exc}", RuntimeWarning)
            return _slice_history(cached, start_date, end_date)
        if not cached.empty:
            latest_cached = cached["date"].max().date()
            raise RuntimeError(
                f"{index_code} index fetch failed and cache is stale "
                f"(latest {latest_cached}, requested {end_date}): {exc}"
            ) from exc
        raise
    if cached.empty:
        merged = fresh
    else:
        merged = pd.concat([cached, fresh], ignore_index=True)
        merged = merged.drop_duplicates(subset=["date"], keep="last")
        merged = merged.sort_values("date").reset_index(drop=True)
    merged.to_csv(cache_path, index=False)
    return _slice_history(merged, start_date, end_date)


def _fetch_index_from_source(index_code: str, start_date: date, end_date: date) -> pd.DataFrame:
    if _tushare_enabled():
        try:
            return _fetch_index_from_tushare(index_code, start_date, end_date)
        except Exception as exc:  # pragma: no cover - fallback path
            warnings.warn(f"Tushare index fetch failed; falling back to AkShare: {exc}", RuntimeWarning)
    return _fetch_index_from_akshare(index_code, start_date, end_date)


def _fetch_index_from_tushare(index_code: str, start_date: date, end_date: date) -> pd.DataFrame:
    from .tushare_client import get_tushare_pro

    if index_code.startswith(("60", "68", "90", "00")):
        ts_code = f"{index_code}.SH"
    else:
        ts_code = f"{index_code}.SZ"

    pro = get_tushare_pro()
    raw = _call_tushare(
        f"index_daily {ts_code}",
        lambda: pro.index_daily(
            ts_code=ts_code,
            start_date=start_date.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d"),
        ),
    )
    if raw is None or raw.empty:
        raise RuntimeError(f"Tushare index_daily returned no rows for {ts_code}")
    df = raw.rename(columns={"trade_date": "date", "vol": "volume"}).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ("open", "close", "high", "low", "volume", "amount"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "amount" not in df.columns:
        df["amount"] = 0
    return df.dropna(subset=["date", "open", "close", "high", "low"]).sort_values("date").reset_index(drop=True)


def _fetch_index_from_akshare(index_code: str, start_date: date, end_date: date) -> pd.DataFrame:
    ak = _akshare()
    if index_code.startswith(("60", "68", "90", "00")):
        symbol = f"sh{index_code}"
    else:
        symbol = f"sz{index_code}"

    raw = ak.stock_zh_index_daily(symbol=symbol)
    if raw is None or raw.empty:
        raise RuntimeError(f"AkShare index_daily returned no rows for {symbol}")
    df = raw.rename(columns={"date": "date"}).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ("open", "close", "high", "low", "volume", "amount"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "amount" not in df.columns:
        df["amount"] = 0
    df = df.dropna(subset=["date", "open", "close", "high", "low"]).sort_values("date").reset_index(drop=True)
    return df[(df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))]


def fetch_daily_basic(trade_date: str, cache_dir: Path) -> dict[str, float]:
    """Fetch daily basic info (market cap, PE, PB) for all stocks on a trade date.

    Returns a dict mapping 6-digit code → circ_mv in 亿元.
    """
    cache_path = cache_dir / "tushare_daily_basic" / f"{trade_date}.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        df = pd.read_csv(cache_path, dtype={"code": str})
        df["code"] = df["code"].str.zfill(6)
        return dict(zip(df["code"], df["circ_mv"]))

    if not _tushare_enabled():
        return {}

    from .tushare_client import get_tushare_pro

    pro = get_tushare_pro()
    raw = _call_tushare(f"daily_basic {trade_date}", lambda: pro.daily_basic(trade_date=trade_date))
    if raw is None or raw.empty:
        return {}

    df = raw.copy()
    df["code"] = df["ts_code"].astype(str).str.extract(r"(\d{6})", expand=False).str.zfill(6)
    # Tushare reports market cap in 万元; convert to 亿元 for display.
    df["circ_mv"] = pd.to_numeric(df.get("circ_mv", 0), errors="coerce").fillna(0) / 10000
    df = df[df["circ_mv"] > 0]

    result: dict[str, float] = dict(zip(df["code"], df["circ_mv"]))
    pd.DataFrame({"code": list(result.keys()), "circ_mv": list(result.values())}).to_csv(
        cache_path, index=False
    )
    return result


def fetch_stock_tags(symbol: str, cache_dir: Path, force_refresh: bool = False) -> tuple[str, ...]:
    """Fetch industry/concept tags for a selected stock.

    This is intentionally called only after the screener has selected TopN candidates.
    Tag endpoints are slower and less critical than price data, so stale tag cache is
    acceptable if the upstream source is temporarily unavailable.
    """
    symbol = symbol.zfill(6)
    cache_path = cache_dir / "stock_tags" / f"{symbol}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not force_refresh:
        cached = _read_stock_tags_cache(cache_path)
        if cached and _stock_tag_cache_is_fresh(cache_path):
            return cached

    cached = _read_stock_tags_cache(cache_path)
    tags: tuple[str, ...] = ()
    errors: list[str] = []
    if _tushare_enabled():
        try:
            tags = _fetch_stock_tags_from_tushare(symbol)
        except Exception as exc:  # pragma: no cover - external source variance
            errors.append(f"Tushare tags failed: {exc}")
    if not tags:
        try:
            tags = _fetch_stock_tags_from_akshare(symbol)
        except Exception as exc:  # pragma: no cover - external source variance
            errors.append(f"AkShare tags failed: {exc}")

    if tags:
        _write_stock_tags_cache(cache_path, tags)
        return tags
    if cached:
        warnings.warn(f"{symbol} tag fetch failed; using cached tags: {'; '.join(errors)}", RuntimeWarning)
        return cached
    if errors:
        warnings.warn(f"{symbol} tag fetch failed: {'; '.join(errors)}", RuntimeWarning)
    return ()


def _fetch_stock_tags_from_tushare(symbol: str) -> tuple[str, ...]:
    from .tushare_client import get_tushare_pro

    pro = get_tushare_pro()
    ts_code = _tushare_ts_code(symbol)
    tags: list[str] = []
    basic = _call_tushare(
        f"stock_basic tags {symbol}",
        lambda: pro.stock_basic(ts_code=ts_code, fields="ts_code,symbol,name,industry"),
    )
    if basic is not None and not basic.empty and "industry" in basic.columns:
        tags.extend(str(item) for item in basic["industry"].dropna().tolist())

    concepts = _call_tushare(
        f"concept_detail {symbol}",
        lambda: pro.concept_detail(ts_code=ts_code, fields="id,concept_name,ts_code,name"),
    )
    if concepts is not None and not concepts.empty and "concept_name" in concepts.columns:
        tags.extend(str(item) for item in concepts["concept_name"].dropna().tolist())
    return _clean_stock_tags(tags)


def _fetch_stock_tags_from_akshare(symbol: str) -> tuple[str, ...]:
    ak = _akshare()
    raw = ak.stock_individual_info_em(symbol=symbol, timeout=10)
    if raw is None or raw.empty or "item" not in raw.columns or "value" not in raw.columns:
        return ()
    rows = raw.set_index("item")["value"].to_dict()
    return _clean_stock_tags([str(rows.get("行业", ""))])


def _clean_stock_tags(values: list[str]) -> tuple[str, ...]:
    tags: list[str] = []
    seen: set[str] = set()
    for raw in values:
        tag = str(raw).strip()
        if not tag or tag in {"-", "None", "nan"} or tag in GENERIC_CONCEPT_TAGS:
            continue
        if tag in seen:
            continue
        seen.add(tag)
        tags.append(tag)
        if len(tags) >= MAX_STOCK_TAGS:
            break
    return tuple(tags)


def _read_stock_tags_cache(cache_path: Path) -> tuple[str, ...]:
    if not cache_path.exists():
        return ()
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        tags = payload.get("tags", [])
        if isinstance(tags, list):
            return _clean_stock_tags([str(item) for item in tags])
    except Exception as exc:  # pragma: no cover - corrupt local cache path
        warnings.warn(f"discarding invalid tag cache {cache_path}: {exc}", RuntimeWarning)
    return ()


def _write_stock_tags_cache(cache_path: Path, tags: tuple[str, ...]) -> None:
    cache_path.write_text(
        json.dumps({"updated_at": date.today().isoformat(), "tags": list(tags)}, ensure_ascii=False),
        encoding="utf-8",
    )


def _stock_tag_cache_is_fresh(cache_path: Path) -> bool:
    max_age_days = _env_int("STOCK_TAG_CACHE_DAYS", DEFAULT_TAG_CACHE_DAYS)
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        updated_at = pd.to_datetime(payload.get("updated_at"), errors="coerce")
        if pd.isna(updated_at):
            return False
        return updated_at.date() >= date.today() - timedelta(days=max_age_days)
    except Exception:
        return False


def fetch_financial_metrics(symbol: str, cache_dir: Path, force_refresh: bool = False) -> dict[str, object]:
    """Fetch latest growth/quality metrics for a selected stock.

    This runs only for final candidates. It is a secondary growth-quality lens,
    not part of the full-market technical breakout scan.
    """
    symbol = symbol.zfill(6)
    cache_path = cache_dir / "financial_metrics" / f"{symbol}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not force_refresh:
        cached = _read_financial_metrics_cache(cache_path)
        if cached and _financial_cache_is_fresh(cache_path):
            return cached

    cached = _read_financial_metrics_cache(cache_path)
    if not _tushare_enabled():
        return cached

    try:
        metrics = _fetch_financial_metrics_from_tushare(symbol)
    except Exception as exc:  # pragma: no cover - external source variance
        if cached:
            warnings.warn(f"{symbol} financial metrics fetch failed; using cached metrics: {exc}", RuntimeWarning)
            return cached
        warnings.warn(f"{symbol} financial metrics fetch failed: {exc}", RuntimeWarning)
        return {}
    if metrics:
        _write_financial_metrics_cache(cache_path, metrics)
    return metrics


def _fetch_financial_metrics_from_tushare(symbol: str) -> dict[str, object]:
    from .tushare_client import get_tushare_pro

    pro = get_tushare_pro()
    raw = _call_tushare(
        f"fina_indicator {symbol}",
        lambda: pro.fina_indicator(
            ts_code=_tushare_ts_code(symbol),
            fields=(
                "ts_code,end_date,ann_date,roe,roe_dt,roa,"
                "netprofit_yoy,or_yoy,grossprofit_margin,debt_to_assets"
            ),
        ),
    )
    if raw is None or raw.empty:
        return {}

    df = raw.copy()
    df["end_date"] = pd.to_datetime(df["end_date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["end_date"]).sort_values("end_date", ascending=False)
    if df.empty:
        return {}
    row = df.iloc[0]
    metrics: dict[str, object] = {
        "financial_end_date": row["end_date"].date().isoformat(),
        "financial_ann_date": _date_text(row.get("ann_date")),
        "revenue_yoy": _optional_number(row.get("or_yoy")),
        "profit_yoy": _optional_number(row.get("netprofit_yoy")),
        "roe": _optional_number(row.get("roe_dt", row.get("roe"))),
        "gross_margin": _optional_number(row.get("grossprofit_margin")),
        "debt_to_assets": _optional_number(row.get("debt_to_assets")),
    }
    metrics["growth_score"] = _calc_growth_score(metrics)
    return metrics


def _calc_growth_score(metrics: dict[str, object]) -> float:
    revenue_yoy = _optional_number(metrics.get("revenue_yoy"))
    profit_yoy = _optional_number(metrics.get("profit_yoy"))
    roe = _optional_number(metrics.get("roe"))
    gross_margin = _optional_number(metrics.get("gross_margin"))
    debt_to_assets = _optional_number(metrics.get("debt_to_assets"))

    score = 0.0
    available = 0
    if revenue_yoy is not None:
        score += _linear_score(revenue_yoy, -10, 40, 0, 25)
        available += 25
    if profit_yoy is not None:
        score += _linear_score(profit_yoy, -20, 60, 0, 25)
        available += 25
    if roe is not None:
        score += _linear_score(roe, 0, 18, 0, 20)
        available += 20
    if gross_margin is not None:
        score += _linear_score(gross_margin, 10, 45, 0, 15)
        available += 15
    if debt_to_assets is not None:
        score += _linear_score(80 - debt_to_assets, 0, 50, 0, 15)
        available += 15
    if not available:
        return 0.0
    return round(score / available * 100, 2)


def _linear_score(value: float, low: float, high: float, min_score: float, max_score: float) -> float:
    if value <= low:
        return min_score
    if value >= high:
        return max_score
    return min_score + (value - low) / (high - low) * (max_score - min_score)


def _read_financial_metrics_cache(cache_path: Path) -> dict[str, object]:
    if not cache_path.exists():
        return {}
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        metrics = payload.get("metrics", {})
        return metrics if isinstance(metrics, dict) else {}
    except Exception as exc:  # pragma: no cover - corrupt local cache path
        warnings.warn(f"discarding invalid financial cache {cache_path}: {exc}", RuntimeWarning)
        return {}


def _write_financial_metrics_cache(cache_path: Path, metrics: dict[str, object]) -> None:
    cache_path.write_text(
        json.dumps({"updated_at": date.today().isoformat(), "metrics": metrics}, ensure_ascii=False),
        encoding="utf-8",
    )


def _financial_cache_is_fresh(cache_path: Path) -> bool:
    max_age_days = _env_int("STOCK_FINANCIAL_CACHE_DAYS", DEFAULT_FINANCIAL_CACHE_DAYS)
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        updated_at = pd.to_datetime(payload.get("updated_at"), errors="coerce")
        if pd.isna(updated_at):
            return False
        return updated_at.date() >= date.today() - timedelta(days=max_age_days)
    except Exception:
        return False


def _optional_number(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def _date_text(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    if pd.isna(parsed):
        return text
    return parsed.date().isoformat()


def _tushare_enabled() -> bool:
    return bool(os.getenv("TUSHARE_TOKEN") or os.getenv("TUSHARE_PRO_TOKEN"))


def _call_tushare(label: str, callback):
    attempts = _env_int("TUSHARE_RETRIES", DEFAULT_TUSHARE_RETRIES)
    base_delay = _env_float("TUSHARE_RETRY_DELAY", DEFAULT_TUSHARE_RETRY_DELAY)
    max_delay = _env_float("TUSHARE_RETRY_MAX_DELAY", DEFAULT_TUSHARE_RETRY_MAX_DELAY)
    for attempt in range(1, attempts + 1):
        try:
            return callback()
        except Exception as exc:
            if attempt >= attempts:
                raise
            wait = min(max(0.0, base_delay) * attempt, max(0.0, max_delay))
            warnings.warn(
                f"Tushare {label} failed ({attempt}/{attempts}); retrying in {wait:.1f}s: {exc}",
                RuntimeWarning,
            )
            if wait > 0:
                time.sleep(wait)
    raise RuntimeError(f"Tushare {label} failed")


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default
