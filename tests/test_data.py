from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from a_breakout_screener import data


def test_fetch_index_history_falls_back_to_cache(monkeypatch, tmp_path) -> None:
    cache_path = tmp_path / "hist" / "index_000300.csv"
    cache_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "date": "2026-01-02",
                "open": 1,
                "high": 2,
                "low": 1,
                "close": 2,
                "volume": 100,
                "amount": 1000,
            }
        ]
    ).to_csv(cache_path, index=False)

    def fail_fetch(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(data, "_fetch_index_from_source", fail_fetch)

    with pytest.warns(RuntimeWarning, match="using cached data"):
        history = data.fetch_index_history("000300", date(2026, 1, 1), date(2026, 1, 5), tmp_path)

    assert len(history) == 1
    assert float(history.iloc[0]["close"]) == 2


def test_fetch_history_rejects_stale_cache_on_source_failure(monkeypatch, tmp_path) -> None:
    cache_path = tmp_path / "hist" / "000001.csv"
    cache_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "date": "2026-01-02",
                "open": 1,
                "high": 2,
                "low": 1,
                "close": 2,
                "volume": 100,
                "amount": 1000,
            }
        ]
    ).to_csv(cache_path, index=False)

    def fail_fetch(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(data, "_fetch_history_from_source", fail_fetch)

    with pytest.raises(RuntimeError, match="cache is stale"):
        data.fetch_history("000001", date(2026, 1, 1), date(2026, 2, 1), tmp_path)


def test_fetch_history_backfills_older_cache_when_lookback_expands(monkeypatch, tmp_path) -> None:
    cache_path = tmp_path / "hist" / "000001.csv"
    cache_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"date": "2026-01-10", "open": 10, "high": 11, "low": 9, "close": 10, "volume": 100, "amount": 1000},
            {"date": "2026-01-12", "open": 12, "high": 13, "low": 11, "close": 12, "volume": 100, "amount": 1000},
        ]
    ).to_csv(cache_path, index=False)
    calls = []

    def fake_fetch(symbol, start_date, end_date):
        calls.append((symbol, start_date, end_date))
        return pd.DataFrame(
            [
                {"date": pd.Timestamp("2026-01-02"), "open": 2, "high": 3, "low": 1, "close": 2, "volume": 100, "amount": 1000},
                {"date": pd.Timestamp("2026-01-05"), "open": 5, "high": 6, "low": 4, "close": 5, "volume": 100, "amount": 1000},
            ]
        )

    monkeypatch.setattr(data, "_fetch_history_from_source", fake_fetch)

    history = data.fetch_history("000001", date(2026, 1, 1), date(2026, 1, 12), tmp_path)

    assert calls == [("000001", date(2026, 1, 1), date(2026, 1, 12))]
    assert history["date"].min().date() == date(2026, 1, 2)
    assert history["date"].max().date() == date(2026, 1, 12)
    cached = pd.read_csv(cache_path)
    assert "2026-01-02" in set(cached["date"].astype(str))


def test_call_tushare_retries_transient_failure(monkeypatch) -> None:
    calls = {"count": 0}

    monkeypatch.setenv("TUSHARE_RETRIES", "2")
    monkeypatch.setenv("TUSHARE_RETRY_DELAY", "0")
    monkeypatch.setattr(data.time, "sleep", lambda _: None)

    def flaky_call():
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("temporary timeout")
        return "ok"

    with pytest.warns(RuntimeWarning, match="retrying"):
        result = data._call_tushare("daily 20260512", flaky_call)

    assert result == "ok"
    assert calls["count"] == 2


def test_fetch_stock_tags_filters_generic_tags_and_uses_cache(monkeypatch, tmp_path) -> None:
    class FakePro:
        def __init__(self) -> None:
            self.calls = 0

        def stock_basic(self, **kwargs):
            self.calls += 1
            return pd.DataFrame([{"ts_code": "300750.SZ", "symbol": "300750", "name": "宁德时代", "industry": "电气设备"}])

        def concept_detail(self, **kwargs):
            return pd.DataFrame(
                [
                    {"concept_name": "融资融券"},
                    {"concept_name": "新能源车"},
                    {"concept_name": "锂电池"},
                ]
            )

    fake_pro = FakePro()
    monkeypatch.setenv("TUSHARE_TOKEN", "token")
    monkeypatch.setattr("a_breakout_screener.tushare_client.get_tushare_pro", lambda: fake_pro)

    tags = data.fetch_stock_tags("300750", tmp_path)

    assert tags == ("电气设备", "新能源车", "锂电池")
    assert (tmp_path / "stock_tags" / "300750.json").exists()

    monkeypatch.setattr("a_breakout_screener.tushare_client.get_tushare_pro", lambda: (_ for _ in ()).throw(RuntimeError("down")))

    assert data.fetch_stock_tags("300750", tmp_path) == tags


def test_fetch_financial_metrics_scores_and_uses_cache(monkeypatch, tmp_path) -> None:
    class FakePro:
        def __init__(self) -> None:
            self.calls = 0

        def fina_indicator(self, **kwargs):
            self.calls += 1
            return pd.DataFrame(
                [
                    {
                        "ts_code": "300750.SZ",
                        "end_date": "20260331",
                        "ann_date": "20260416",
                        "roe_dt": 5.2,
                        "netprofit_yoy": 48.5,
                        "or_yoy": 52.4,
                        "grossprofit_margin": 24.8,
                        "debt_to_assets": 62.3,
                    }
                ]
            )

    fake_pro = FakePro()
    monkeypatch.setenv("TUSHARE_TOKEN", "token")
    monkeypatch.setattr("a_breakout_screener.tushare_client.get_tushare_pro", lambda: fake_pro)

    metrics = data.fetch_financial_metrics("300750", tmp_path)

    assert metrics["financial_end_date"] == "2026-03-31"
    assert metrics["revenue_yoy"] == 52.4
    assert metrics["profit_yoy"] == 48.5
    assert metrics["growth_score"] > 0
    assert (tmp_path / "financial_metrics" / "300750.json").exists()

    monkeypatch.setattr("a_breakout_screener.tushare_client.get_tushare_pro", lambda: (_ for _ in ()).throw(RuntimeError("down")))

    assert data.fetch_financial_metrics("300750", tmp_path) == metrics
