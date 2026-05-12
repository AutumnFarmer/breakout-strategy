from __future__ import annotations

import warnings
from datetime import date, timedelta
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
    basic = pro.stock_basic(
        exchange="",
        list_status="L",
        fields="ts_code,symbol,name",
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
        if not cached.empty:
            warnings.warn(f"{symbol} history fetch failed; using cached data: {exc}", RuntimeWarning)
            return _slice_history(cached, start_date, end_date)
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

    raw = get_tushare_pro().daily(
        ts_code=_tushare_ts_code(symbol),
        start_date=start_date.strftime("%Y%m%d"),
        end_date=end_date.strftime("%Y%m%d"),
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
    calendar = pro.trade_cal(
        exchange="SSE",
        start_date=start_date.strftime("%Y%m%d"),
        end_date=end_date.strftime("%Y%m%d"),
        is_open="1",
    )
    if calendar is None or calendar.empty:
        raise RuntimeError("Tushare trade_cal returned no open trading days")
    trade_dates = sorted(str(item) for item in calendar["cal_date"].dropna().tolist())
    for trade_date in reversed(trade_dates):
        daily = pro.daily(trade_date=trade_date)
        if daily is not None and not daily.empty:
            return trade_date, daily
    raise RuntimeError("Tushare daily returned no rows for recent open trading days")


def _fetch_tushare_daily_range(start_date: date, end_date: date, cache_dir: Path) -> pd.DataFrame:
    from .tushare_client import get_tushare_pro

    pro = get_tushare_pro()
    calendar = pro.trade_cal(
        exchange="SSE",
        start_date=start_date.strftime("%Y%m%d"),
        end_date=end_date.strftime("%Y%m%d"),
        is_open="1",
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


def _fetch_tushare_daily_with_retry(pro, trade_date: str, attempts: int = 3) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            daily = pro.daily(trade_date=trade_date)
            if daily is None or daily.empty:
                return pd.DataFrame()
            return _normalize_tushare_daily(daily)
        except Exception as exc:  # pragma: no cover - external network variance
            last_error = exc
            wait = min(3.0 * attempt, 10.0)
            print(f"Tushare历史缓存: {trade_date} 请求失败，{wait:.0f}s 后重试: {exc}", flush=True)
            time.sleep(wait)
    warnings.warn(f"Tushare daily failed for {trade_date}; skipped: {last_error}", RuntimeWarning)
    return pd.DataFrame()


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
        if not cached.empty:
            warnings.warn(f"{index_code} index fetch failed; using cached data: {exc}", RuntimeWarning)
            return _slice_history(cached, start_date, end_date)
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

    raw = get_tushare_pro().index_daily(
        ts_code=ts_code,
        start_date=start_date.strftime("%Y%m%d"),
        end_date=end_date.strftime("%Y%m%d"),
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
    raw = pro.daily_basic(trade_date=trade_date)
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


def _tushare_enabled() -> bool:
    return bool(os.getenv("TUSHARE_TOKEN") or os.getenv("TUSHARE_PRO_TOKEN"))
