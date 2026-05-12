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
