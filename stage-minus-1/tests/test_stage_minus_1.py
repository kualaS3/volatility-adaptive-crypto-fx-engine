"""Tests for the Stage -1 pipeline on synthetic data."""

from datetime import date

import numpy as np
import pandas as pd

from black_scholes import bs_call, bs_put
from stage_minus_1 import (
    analyze_asset_30d,
    analyze_snapshot,
    build_summary_markdown,
    score_vol_forecasts,
    verdict_paragraph,
)


def _synthetic_frame(n_days=300, seed=11):
    """A synthetic daily frame: close, return and realised variance."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days, freq="D")
    returns = rng.normal(0.1, 3.0, size=n_days)
    close = 60000.0 * np.exp(np.cumsum(returns / 100.0))
    rv = np.abs(rng.normal(9.0, 4.0, size=n_days))
    frame = pd.DataFrame(
        {"close": close, "ret": returns, "rv": rv, "n_bars": 288},
        index=dates.date)
    frame.index.name = "date"
    return frame


def _synthetic_dvol(frame, seed=11):
    """A synthetic DVOL series covering the frame dates."""
    rng = np.random.default_rng(seed + 1)
    values = rng.normal(55.0, 8.0, size=len(frame))
    return pd.Series(np.clip(values, 20.0, 120.0), index=frame.index)


def test_score_vol_forecasts_prefers_the_better_forecaster():
    """On synthetic data the accurate forecaster wins on QLIKE.

    Realised variance is the GARCH variance plus small noise, and
    the implied variance is 50 percent too high, so the GARCH
    QLIKE is lower and the mean gap of forecast minus implied is
    negative, which is the vol risk premium headwind.
    """
    rng = np.random.default_rng(3)
    n = 200
    garch_var = rng.uniform(5.0, 20.0, size=n)
    realised_var = garch_var * rng.normal(1.0, 0.1, size=n)
    realised_var = np.clip(realised_var, 0.5, None)
    iv_var = garch_var * 1.5
    factor = 30.0 / 365.0
    garch_vol = np.sqrt(garch_var / factor)
    iv_vol = np.sqrt(iv_var / factor)
    result = score_vol_forecasts(realised_var, garch_var, iv_var,
                                 garch_vol, iv_vol)
    assert result["qlike_garch"] < result["qlike_iv"]
    assert result["mean_gap_vol_points"] < 0
    assert 0.0 <= result["dm_pvalue"] <= 1.0


def test_analyze_asset_30d_end_to_end_synthetic():
    """The full 30-day analysis runs on synthetic data.

    Checks the shape of the result: a horse race with a positive
    day count, a fitted GARCH summary, repricing edges and expiry
    payoffs that are all finite numbers.
    """
    frame = _synthetic_frame()
    dvol = _synthetic_dvol(frame)
    result = analyze_asset_30d("BTC", frame, dvol)
    assert result["name"] == "BTC"
    assert result["race"]["n_days"] > 0
    assert result["race"]["n_days"] <= 60
    assert 0.0 < result["fit"].persistence < 1.5
    for key in ["straddle_edge_mean", "strangle_edge_mean",
                "straddle_net_mean", "strangle_net_mean"]:
        assert np.isfinite(result[key])
    assert 0 <= result["race"]["n_signal_days"] <= result["race"]["n_days"]
    assert np.isfinite(result["mean_abs_ret_30d"])
    assert np.isfinite(result["mean_expected_abs_ret_30d"])
    assert np.isfinite(result["mean_log_ret_30d"])


def test_analyze_snapshot_repricing_sign():
    """Snapshot repricing: a forecast above IV gives a positive edge.

    Mark prices are built from Black-Scholes at the mark IV, so
    when the forecast vol is set above that IV the model value at
    forecast vol exceeds the market price for both structures.
    """
    spot = 60000.0
    time_years = 7.0 / 365.0
    iv = 0.55
    rows = []
    for strike in [54000.0, 57000.0, 60000.0, 63000.0, 66000.0]:
        for option_type in ("C", "P"):
            price = (bs_call(spot, strike, time_years, iv)
                     if option_type == "C"
                     else bs_put(spot, strike, time_years, iv))
            rows.append({
                "currency": "BTC",
                "instrument_name": f"BTC-04DEC26-{int(strike)}-"
                                   f"{option_type}",
                "strike": strike,
                "expiry": date(2026, 12, 4),
                "option_type": option_type,
                "mark_iv": iv * 100.0,
                "mark_price": price / spot,
                "best_bid_price": price / spot * 0.98,
                "best_ask_price": price / spot * 1.02,
                "open_interest": 100.0,
                "underlying_price": spot,
            })
    snapshot = pd.DataFrame(rows)
    frame = _synthetic_frame(n_days=30)
    latest = {
        "omega": 0.5,
        "alpha": 0.1,
        "beta": 0.8,
        "var_one_step": 16.0,
    }
    results = analyze_snapshot("BTC", snapshot, latest, frame,
                               date(2026, 11, 27))
    assert len(results) == 1
    snap = results[0]
    assert snap["days_to_expiry"] == 7
    assert snap["forecast_vol"] > iv * 100.0
    assert snap["straddle_edge"] > 0
    assert snap["strangle_edge"] > 0
    assert snap["atm_strike"] == 60000.0


def test_build_summary_markdown_contains_results():
    """The summary builder renders the computed numbers."""
    frame = _synthetic_frame()
    dvol = _synthetic_dvol(frame)
    result = analyze_asset_30d("BTC", frame, dvol)
    snapshots = {"BTC": [{
        "expiry": date(2026, 12, 4),
        "days_to_expiry": 7,
        "spot": 60000.0,
        "forecast_vol": 55.0,
        "atm_strike": 60000.0,
        "atm_call_iv": 52.0,
        "atm_put_iv": 53.0,
        "recent_realised_7d": 48.0,
        "market_straddle": 2000.0,
        "model_straddle": 2100.0,
        "straddle_edge": 100.0,
        "half_spread_straddle": 20.0,
        "strangle_call_strike": 63000.0,
        "strangle_call_delta": 0.25,
        "strangle_put_strike": 57000.0,
        "strangle_put_delta": -0.25,
        "market_strangle": 800.0,
        "model_strangle": 850.0,
        "strangle_edge": 50.0,
        "half_spread_strangle": 10.0,
    }]}
    dvol_info = {"BTC": {"rows": 500, "first_date": "2024-01-01",
                         "last_date": "2026-11-27"},
                 "ETH": {"rows": 500, "first_date": "2024-01-01",
                         "last_date": "2026-11-27"}}
    text = build_summary_markdown([result], snapshots, dvol_info,
                                  "2026-11-27", "2026-11-23")
    assert "# Stage -1" in text
    assert "## BTC" in text
    assert "## Verdict" in text
    assert "QLIKE" in text
    assert "unhedged" in text


def _fake_result(name, qlike_garch, qlike_iv, gap, signal_net):
    """A minimal result dict for testing the verdict wording."""
    return {
        "name": name,
        "race": {
            "qlike_garch": qlike_garch,
            "qlike_iv": qlike_iv,
            "dm_pvalue": 0.01,
            "mean_gap_vol_points": gap,
            "mean_iv_vol": 40.0,
            "mean_realised_vol": 39.0,
            "n_days": 100,
            "n_signal_days": 99,
        },
        "straddle_net_signal_mean": signal_net,
        "strangle_net_signal_mean": signal_net,
        "mean_abs_ret_30d": 11.0,
        "mean_expected_abs_ret_30d": 8.5,
    }


def test_verdict_no_vol_edge_when_iv_wins():
    """IV winning the race plus a positive payoff is not a vol edge.

    When implied vol beats GARCH and prices realised variance
    fairly, the verdict says there is no vol edge even though the
    unhedged payoff is positive, and it names the fat moves as the
    source of the payoff.
    """
    text = verdict_paragraph(
        [_fake_result("BTC", 0.12, 0.06, 7.0, 100.0)])
    assert "no vol edge" in text
    assert "above implied vol" in text
    assert "fat moves" in text
    assert "barely filters" in text


def test_verdict_mixed_when_garch_wins():
    """GARCH winning the race gives the mixed verdict."""
    text = verdict_paragraph(
        [_fake_result("BTC", 0.05, 0.06, -3.0, 100.0)])
    assert "mixed" in text
    assert "below implied vol" in text
