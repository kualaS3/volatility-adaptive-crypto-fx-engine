"""End-to-end test: Stage 0A runs on fake 5-minute bars.

The bars are simulated, so the numbers mean nothing about any real
market. The test checks the pipeline holds together: the daily frame
builds, the GARCH fit converges, the forecasts cover the test set,
and every reported metric is finite with the Diebold-Mariano p-value
in [0, 1].
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from stage0a import analyze_asset, available_assets


def _fake_bars(n_days: int = 100, seed: int = 5) -> pd.DataFrame:
    """100 days of 5-minute bars from a seeded random walk.

    Volatility switches between a calm and a choppy regime halfway
    through, so the series has clustering for the GARCH fit to find.
    """
    rng = np.random.default_rng(seed)
    bars_per_day = 288
    n = n_days * bars_per_day
    # 5-minute log-return vol of 0.15% calm and 0.4% choppy, which is
    # roughly 2.5% and 7% a day: realistic crypto magnitudes, so the
    # GARCH fit sees a well-scaled series.
    vol = np.where(
        np.arange(n) < n // 2, 0.0015, 0.004
    )
    shocks = rng.standard_normal(n) * vol
    log_close = np.cumsum(shocks) + np.log(60000.0)
    closes = np.exp(log_close)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    times = [start + timedelta(minutes=5 * i) for i in range(n)]
    opens = np.concatenate([[closes[0]], closes[:-1]])
    return pd.DataFrame(
        {
            "open_time": times,
            "open": opens,
            "high": np.maximum(opens, closes) * 1.0005,
            "low": np.minimum(opens, closes) * 0.9995,
            "close": closes,
            "volume": rng.uniform(1.0, 10.0, size=n),
        }
    )


def test_stage0a_runs_end_to_end_on_synthetic_bars(tmp_path):
    bars = _fake_bars()
    path = tmp_path / "fake_5m.parquet"
    bars.to_parquet(path, index=False)
    result = analyze_asset("FAKE", path, annual_days=365)
    # 100 days of bars give 99 daily rows: the first day has no
    # previous close to return from, so it is dropped.
    assert result["n_days"] == 99
    assert result["n_train"] == 79
    assert result["n_test"] == 20
    assert result["bars_per_day_median"] == 288
    summary = result["summary"]
    assert np.isfinite(summary.persistence)
    assert summary.persistence > 0.0
    assert summary.omega > 0.0
    assert summary.alpha >= 0.0
    assert summary.beta >= 0.0
    # The synthetic series switches volatility regime halfway, which
    # is exactly the situation that pushes measured persistence to the
    # boundary of 1. When that happens the half-life is infinite and
    # the unconditional vol is undefined, which summarize_fit reports
    # as inf and NaN rather than crashing.
    if summary.persistence < 1.0:
        assert summary.annualized_vol_pct > 0.0
    else:
        assert math.isinf(summary.half_life_days)
        assert math.isnan(summary.annualized_vol_pct)
    assert 0.0 <= result["lb_pvalue"] <= 1.0
    assert 0.0 <= result["arch_pvalue"] <= 1.0
    for metrics in result["metrics"].values():
        assert np.isfinite(metrics["qlike"])
        assert np.isfinite(metrics["mse"])
        assert 0.0 <= metrics["mz"]["r_squared"] <= 1.0
    assert np.isfinite(result["dm_stat"])
    assert 0.0 <= result["dm_pvalue"] <= 1.0


def test_stage0a_runs_end_to_end_on_a_second_synthetic_asset(tmp_path):
    # The pipeline is per-asset, so a second asset file with a
    # different seed and length must run the same way as the first:
    # daily frame, fit, forecasts over the test set, finite metrics.
    bars = _fake_bars(n_days=120, seed=11)
    path = tmp_path / "fake2_5m.parquet"
    bars.to_parquet(path, index=False)
    result = analyze_asset("FAKE2", path, annual_days=365)
    assert result["n_days"] == 119
    assert result["n_train"] == 95
    assert result["n_test"] == 24
    assert result["bars_per_day_median"] == 288
    assert np.isfinite(result["summary"].persistence)
    assert result["summary"].persistence > 0.0
    for metrics in result["metrics"].values():
        assert np.isfinite(metrics["qlike"])
        assert np.isfinite(metrics["mse"])
        assert 0.0 <= metrics["mz"]["r_squared"] <= 1.0
    assert np.isfinite(result["dm_stat"])
    assert 0.0 <= result["dm_pvalue"] <= 1.0


def test_available_assets_skips_missing_files(tmp_path):
    # An asset whose Parquet file is absent is skipped with a warning,
    # not a crash, and an asset whose file exists is kept.
    present = tmp_path / "present.parquet"
    present.write_text("x")
    assets = [
        ("HAVE", "present.parquet", 365),
        ("MISSING", "absent.parquet", 252),
    ]
    kept = available_assets(assets, tmp_path)
    assert kept == [("HAVE", "present.parquet", 365)]


def test_available_assets_skips_some_of_many(tmp_path):
    # With five assets listed and only two files present, the three
    # missing ones are skipped and the two present ones are kept, in
    # listed order.
    for name in ("btc_usd_5m.parquet", "sol_usd_5m.parquet"):
        (tmp_path / name).write_text("x")
    assets = [
        ("BTC", "btc_usd_5m.parquet", 365),
        ("ETH", "eth_usd_5m.parquet", 365),
        ("BNB", "bnb_usd_5m.parquet", 365),
        ("XRP", "xrp_usd_5m.parquet", 365),
        ("SOL", "sol_usd_5m.parquet", 365),
        ("EUR/USD", "eurusd_5m.parquet", 252),
    ]
    kept = available_assets(assets, tmp_path)
    assert kept == [
        ("BTC", "btc_usd_5m.parquet", 365),
        ("SOL", "sol_usd_5m.parquet", 365),
    ]


def test_assets_list_covers_the_top_five_plus_fx():
    # Stage 0A is configured for the top five cryptocurrencies by
    # market capitalisation, excluding stablecoins, in ranking order,
    # plus EUR/USD. Crypto annualises on 365 days, FX on 252.
    from stage0a import ASSETS

    assert [name for name, _, _ in ASSETS] == [
        "BTC", "ETH", "BNB", "XRP", "SOL", "EUR/USD",
    ]
    assert [days for _, _, days in ASSETS] == [
        365, 365, 365, 365, 365, 252,
    ]
