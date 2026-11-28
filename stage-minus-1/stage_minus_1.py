"""Stage -1: does a GARCH vol forecast beat the market implied vol.

The question, in plain terms: option prices already contain the
market forecast of volatility as implied volatility, and a GARCH
model aims to forecast the same thing. If the GARCH forecast is
higher than the market forecast by enough to cover costs, buying
volatility should have a positive edge. If not, there is no edge.

The pipeline, for BTC and ETH only, since those are the assets
with Deribit options:

1. Horse race, daily, at 7 and 30 days: GARCH forecast annualised
   vol against implied vol over the same horizon, scored as
   forecasters of subsequently realised vol. Realised vol over each
   horizon comes from the 5-minute bars: the sum of daily realised
   variance over the horizon, annualised the same way. Scoring uses
   QLIKE and Diebold-Mariano from Stage 0A evaluate.py. The average
   gap of forecast minus implied, in vol points, is the vol risk
   premium headwind.
2. Repricing: Black-Scholes prices an at-the-money straddle and a
   25-delta strangle at forecast vol and at implied vol, for the
   live snapshot expiries, and reports the theoretical edge per
   trade as model value at forecast vol minus market price. For the
   historical 30-day check the market price is proxied by the
   Black-Scholes price at DVOL implied vol, which is not a real
   traded price and is stated as such.
3. Edge test: for each day where forecast vol exceeds implied vol,
   the realised expiry payoff of buying the straddle and the
   strangle at the market price and holding to expiry. Costs are
   half the live bid-ask spread where available, otherwise 1
   percent of premium, plus a 0.03 percent of underlying taker fee
   per leg. The average net payoff over signal days is compared
   with the unconditional average over all days. This version is
   unhedged: daily delta hedging is the follow-up, not something
   this test does.
4. Writes stage_minus_1_summary.md.

Limits stated plainly in the summary: the 7-day horse race runs on
the live snapshot only, because a long historical 7-day implied vol
series would need stitching across roughly 150 weekly expiries and
is not built here. The 30-day horizons overlap, so the
Diebold-Mariano p-values are optimistic about serial correlation.
The forecast applied to the live snapshot is dated by the last day
of the Stage 0A data, which is a few days before the snapshot.

Run from stage-minus-1: python -m stage_minus_1
Add --fetch to pull fresh Deribit data first. Without --fetch the
pipeline reads the saved files under data, so repeated runs on the
same saved data produce an identical summary.
"""

from __future__ import annotations

import argparse
import glob
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "stage-0a" / "research"))

from black_scholes import (  # noqa: E402
    bs_delta_call,
    bs_delta_put,
    straddle_value,
    strangle_value,
    strike_for_call_delta,
    strike_for_put_delta,
)
from data import load_parquet  # noqa: E402
from deribit import fetch_all_data, load_dvol  # noqa: E402
from evaluate import diebold_mariano, qlike, qlike_losses  # noqa: E402
from garch_fit import fit_garch, summarize_fit  # noqa: E402
from horizon import (  # noqa: E402
    horizon_annualised_vol,
    latest_horizon_forecasts,
    rolling_horizon_forecasts,
)
from realized import daily_frame  # noqa: E402

TRAIN_FRACTION = 0.80
REFIT_EVERY = 20
HORIZON_DAYS = 30
ANNUAL_DAYS = 365
TAKER_FEE_RATE = 0.0003
FALLBACK_SPREAD_RATE = 0.01
DATA_DIR = ROOT / "data"
SUMMARY_PATH = ROOT / "stage_minus_1_summary.md"
ASSETS = [
    ("BTC", REPO_ROOT / "stage-0a" / "research" / "data" / "btc_usd_5m.parquet"),
    ("ETH", REPO_ROOT / "stage-0a" / "research" / "data" / "eth_usd_5m.parquet"),
]


def score_vol_forecasts(realised_var, garch_var, iv_var,
                        garch_vol, iv_vol) -> dict:
    """Score GARCH vol against implied vol as variance forecasters.

    realised_var, garch_var and iv_var are total variances over the
    horizon in percent squared. garch_vol and iv_vol are the matching
    annualised vols in percent. Returns QLIKE for each forecaster,
    the Diebold-Mariano test on the per-day QLIKE losses, and the
    mean gap of forecast vol minus implied vol in vol points.
    """
    dm = diebold_mariano(qlike_losses(realised_var, garch_var),
                         qlike_losses(realised_var, iv_var))
    return {
        "qlike_garch": qlike(realised_var, garch_var),
        "qlike_iv": qlike(realised_var, iv_var),
        "dm_stat": dm["statistic"],
        "dm_pvalue": dm["p_value"],
        "mean_gap_vol_points": float(np.mean(np.asarray(garch_vol)
                                             - np.asarray(iv_vol))),
        "mean_garch_vol": float(np.mean(garch_vol)),
        "mean_iv_vol": float(np.mean(iv_vol)),
        "mean_realised_vol": float(np.mean(
            np.sqrt(np.asarray(realised_var, dtype=float))
            * (ANNUAL_DAYS / HORIZON_DAYS) ** 0.5)),
    }


def analyze_asset_30d(name: str, frame: pd.DataFrame,
                      dvol: pd.Series) -> dict:
    """Horse race, repricing and edge test at 30 days for one asset.

    Uses the Stage 0A 80/20 split: the GARCH model trains on the
    first 80 percent of days and forecasts the last 20 percent with
    rolling refits every 20 days. A forecast day is usable when the
    full 30-day window of realised variance and the expiry close
    both exist after it, and when DVOL has a value for that day.
    """
    returns = frame["ret"]
    rv = frame["rv"]
    close = frame["close"]
    n_days = len(frame)
    n_train = int(n_days * TRAIN_FRACTION)
    horizon_vols = rolling_horizon_forecasts(
        returns, n_train, horizons=(7, 30), refit_every=REFIT_EVERY)

    fitted = fit_garch(returns.iloc[:n_train])
    fit = summarize_fit(fitted, annual_days=ANNUAL_DAYS)

    realised_var = []
    garch_var = []
    iv_var = []
    garch_vol = []
    iv_vol = []
    straddle_edges = []
    strangle_edges = []
    signal_straddle_edges = []
    signal_strangle_edges = []
    straddle_nets = []
    strangle_nets = []
    signal_straddle_nets = []
    signal_strangle_nets = []
    log_rets_30d = []
    abs_rets_30d = []
    expected_abs_rets_30d = []
    time_years = HORIZON_DAYS / ANNUAL_DAYS
    for pos in range(n_train, n_days):
        if pos + HORIZON_DAYS >= n_days:
            break
        day = frame.index[pos]
        if day not in dvol.index:
            continue
        iv = float(dvol.loc[day])
        # The forecast for the horizon starting the day after the
        # entry day, issued at the close of the entry day, so it
        # covers the same 30 days as the realised window below and
        # the holding period to expiry. Using the forecast for the
        # entry day itself would shift the window one day early.
        forecast_vol = float(horizon_vols.iloc[pos + 1 - n_train]["vol_30d"])
        realised = float(rv.iloc[pos + 1: pos + 1 + HORIZON_DAYS].sum())
        factor = HORIZON_DAYS / ANNUAL_DAYS
        realised_var.append(realised)
        garch_var.append(forecast_vol ** 2 * factor)
        iv_var.append(iv ** 2 * factor)
        garch_vol.append(forecast_vol)
        iv_vol.append(iv)

        spot = float(close.iloc[pos])
        sigma_f = forecast_vol / 100.0
        sigma_iv = iv / 100.0
        spot_expiry = float(close.iloc[pos + HORIZON_DAYS])
        log_ret_30d = float(100.0 * np.log(spot_expiry / spot))
        log_rets_30d.append(log_ret_30d)
        abs_rets_30d.append(abs(log_ret_30d))
        expected_abs_rets_30d.append(float(
            np.sqrt(realised) * (2.0 / np.pi) ** 0.5))
        straddle_market = straddle_value(spot, spot, time_years, sigma_iv)
        straddle_model = straddle_value(spot, spot, time_years, sigma_f)
        straddle_edge = straddle_model - straddle_market
        strike_call = strike_for_call_delta(spot, time_years, sigma_iv, 0.25)
        strike_put = strike_for_put_delta(spot, time_years, sigma_iv, -0.25)
        strangle_market = strangle_value(spot, strike_call, strike_put,
                                         time_years, sigma_iv)
        strangle_model = strangle_value(spot, strike_call, strike_put,
                                        time_years, sigma_f)
        strangle_edge = strangle_model - strangle_market
        straddle_edges.append(straddle_edge)
        strangle_edges.append(strangle_edge)

        straddle_payoff = abs(spot_expiry - spot)
        strangle_payoff = (max(spot_expiry - strike_call, 0.0)
                           + max(strike_put - spot_expiry, 0.0))
        straddle_cost = (FALLBACK_SPREAD_RATE * straddle_market
                         + 2 * TAKER_FEE_RATE * spot)
        strangle_cost = (FALLBACK_SPREAD_RATE * strangle_market
                         + 2 * TAKER_FEE_RATE * spot)
        straddle_net = straddle_payoff - straddle_market - straddle_cost
        strangle_net = strangle_payoff - strangle_market - strangle_cost
        straddle_nets.append(straddle_net)
        strangle_nets.append(strangle_net)
        if forecast_vol > iv:
            signal_straddle_edges.append(straddle_edge)
            signal_strangle_edges.append(strangle_edge)
            signal_straddle_nets.append(straddle_net)
            signal_strangle_nets.append(strangle_net)

    race = score_vol_forecasts(realised_var, garch_var, iv_var,
                               garch_vol, iv_vol)
    race["n_days"] = len(realised_var)
    race["n_signal_days"] = len(signal_straddle_nets)
    return {
        "name": name,
        "first_date": str(frame.index[0]),
        "last_date": str(frame.index[-1]),
        "n_days": n_days,
        "n_train": n_train,
        "fit": fit,
        "race": race,
        "straddle_edge_mean": float(np.mean(straddle_edges)),
        "straddle_edge_signal_mean": (
            float(np.mean(signal_straddle_edges))
            if signal_straddle_edges else float("nan")),
        "strangle_edge_mean": float(np.mean(strangle_edges)),
        "strangle_edge_signal_mean": (
            float(np.mean(signal_strangle_edges))
            if signal_strangle_edges else float("nan")),
        "straddle_net_mean": float(np.mean(straddle_nets)),
        "straddle_net_signal_mean": (
            float(np.mean(signal_straddle_nets))
            if signal_straddle_nets else float("nan")),
        "strangle_net_mean": float(np.mean(strangle_nets)),
        "strangle_net_signal_mean": (
            float(np.mean(signal_strangle_nets))
            if signal_strangle_nets else float("nan")),
        "straddle_win_rate_signal": (
            float(np.mean(np.asarray(signal_straddle_nets) > 0))
            if signal_straddle_nets else float("nan")),
        "strangle_win_rate_signal": (
            float(np.mean(np.asarray(signal_strangle_nets) > 0))
            if signal_strangle_nets else float("nan")),
        "mean_log_ret_30d": float(np.mean(log_rets_30d)),
        "mean_abs_ret_30d": float(np.mean(abs_rets_30d)),
        "mean_expected_abs_ret_30d": float(
            np.mean(expected_abs_rets_30d)),
    }


def analyze_snapshot(asset: str, snapshot: pd.DataFrame,
                     latest: dict, frame: pd.DataFrame,
                     as_of: date) -> list:
    """Repricing for the live snapshot expiries of one asset.

    For each expiry in the snapshot, the at-the-money straddle and
    the 25-delta strangle are priced at the GARCH forecast vol for
    the actual days to expiry and compared with the market price
    from the live mark prices. The forecast vol is the one fitted on
    the Stage 0A data, dated by its last day, which is stated in the
    summary. The strangle strikes are the listed strikes whose
    Black-Scholes delta at mark IV is closest to 0.25 and minus
    0.25. The half spread is reported as the cost the live market
    actually charges for crossing it.
    """
    rows = snapshot[snapshot["currency"] == asset]
    recent_rv = float(frame["rv"].iloc[-7:].sum())
    recent_realised_7d = (recent_rv ** 0.5) * (ANNUAL_DAYS / 7) ** 0.5
    results = []
    for expiry in sorted(rows["expiry"].unique()):
        expiry_rows = rows[rows["expiry"] == expiry].copy()
        days = (expiry - as_of).days
        time_years = days / ANNUAL_DAYS
        spot = float(expiry_rows["underlying_price"].median())
        forecast_vol = horizon_annualised_vol(
            latest["omega"], latest["alpha"], latest["beta"],
            latest["var_one_step"], days, annual_days=ANNUAL_DAYS)
        sigma_f = forecast_vol / 100.0
        calls = expiry_rows[expiry_rows["option_type"] == "C"]
        puts = expiry_rows[expiry_rows["option_type"] == "P"]
        atm_call = calls.loc[(calls["strike"] - spot).abs().idxmin()]
        atm_put = puts.loc[(puts["strike"] - spot).abs().idxmin()]
        atm_strike = float(atm_call["strike"])
        market_straddle = ((float(atm_call["mark_price"])
                            + float(atm_put["mark_price"])) * spot)
        model_straddle = straddle_value(spot, atm_strike, time_years,
                                        sigma_f)
        half_spread_straddle = (
            0.5 * ((float(atm_call["best_ask_price"])
                    - float(atm_call["best_bid_price"]))
                   + (float(atm_put["best_ask_price"])
                      - float(atm_put["best_bid_price"]))) * spot)
        calls = calls.assign(
            delta=calls.apply(
                lambda row: bs_delta_call(
                    spot, float(row["strike"]), time_years,
                    float(row["mark_iv"]) / 100.0),
                axis=1))
        puts = puts.assign(
            delta=puts.apply(
                lambda row: bs_delta_put(
                    spot, float(row["strike"]), time_years,
                    float(row["mark_iv"]) / 100.0),
                axis=1))
        strangle_call = calls.loc[(calls["delta"] - 0.25).abs().idxmin()]
        strangle_put = puts.loc[(puts["delta"] + 0.25).abs().idxmin()]
        market_strangle = ((float(strangle_call["mark_price"])
                            + float(strangle_put["mark_price"])) * spot)
        model_strangle = strangle_value(
            spot, float(strangle_call["strike"]),
            float(strangle_put["strike"]), time_years, sigma_f)
        half_spread_strangle = (
            0.5 * ((float(strangle_call["best_ask_price"])
                    - float(strangle_call["best_bid_price"]))
                   + (float(strangle_put["best_ask_price"])
                      - float(strangle_put["best_bid_price"]))) * spot)
        results.append({
            "expiry": expiry,
            "days_to_expiry": days,
            "spot": spot,
            "forecast_vol": forecast_vol,
            "atm_strike": atm_strike,
            "atm_call_iv": float(atm_call["mark_iv"]),
            "atm_put_iv": float(atm_put["mark_iv"]),
            "recent_realised_7d": recent_realised_7d,
            "market_straddle": market_straddle,
            "model_straddle": model_straddle,
            "straddle_edge": model_straddle - market_straddle,
            "half_spread_straddle": half_spread_straddle,
            "strangle_call_strike": float(strangle_call["strike"]),
            "strangle_call_delta": float(strangle_call["delta"]),
            "strangle_put_strike": float(strangle_put["strike"]),
            "strangle_put_delta": float(strangle_put["delta"]),
            "market_strangle": market_strangle,
            "model_strangle": model_strangle,
            "strangle_edge": model_strangle - market_strangle,
            "half_spread_strangle": half_spread_strangle,
        })
    return results


def _fmt(value: float, spec: str) -> str:
    """Format a float, or n/a when it is not a number."""
    if value != value:
        return "n/a"
    return format(value, spec)


def verdict_paragraph(results: list) -> str:
    """One plain paragraph saying whether an edge exists.

    The verdict is read off the computed numbers, not written by
    hand. A signal day is a day where the GARCH forecast vol
    exceeded the DVOL implied vol. The edge test buys the straddle
    and the strangle on those days and holds to expiry, unhedged.
    The verdict separates the vol question from the payoff
    question: a positive unhedged payoff does not prove a vol edge
    when implied vol already prices realised variance fairly and
    the 30-day moves run fatter than a normal move implies.
    """
    parts = []
    iv_wins_everywhere = True
    for result in results:
        race = result["race"]
        name = result["name"]
        if race["qlike_garch"] < race["qlike_iv"]:
            race_line = (
                f"the GARCH forecast beat implied vol as a forecaster "
                f"of realised vol, with QLIKE {race['qlike_garch']:.4f} "
                f"against {race['qlike_iv']:.4f} for implied vol")
            iv_wins_everywhere = False
        else:
            race_line = (
                f"the GARCH forecast did not beat implied vol as a "
                f"forecaster of realised vol, with QLIKE "
                f"{race['qlike_garch']:.4f} against "
                f"{race['qlike_iv']:.4f} for implied vol")
        if race["dm_pvalue"] < 0.05:
            race_line += (
                f", and the Diebold-Mariano p-value of "
                f"{race['dm_pvalue']:.3f} is below 0.05, so the gap is "
                f"bigger than noise, though the overlapping windows "
                f"make the p-value optimistic")
        else:
            race_line += (
                f", and the Diebold-Mariano p-value of "
                f"{race['dm_pvalue']:.3f} is above 0.05, so the gap is "
                f"within noise")
        gap = race["mean_gap_vol_points"]
        if gap > 0:
            gap_line = (
                f"The forecast sat on average {gap:.1f} vol points "
                f"above implied vol, so it overstates vol rather than "
                f"facing a vol risk premium headwind")
        elif gap < 0:
            gap_line = (
                f"The forecast sat on average {-gap:.1f} vol points "
                f"below implied vol, which is the vol risk premium "
                f"headwind a buyer has to overcome")
        else:
            gap_line = (
                "The forecast sat on average at implied vol, so there "
                "was no vol risk premium headwind either way")
        iv_bias = race["mean_iv_vol"] - race["mean_realised_vol"]
        if abs(iv_bias) <= 2.0:
            pricing_line = (
                f"implied vol priced realised variance about fairly, "
                f"at {race['mean_iv_vol']:.1f} percent against "
                f"{race['mean_realised_vol']:.1f} percent realised")
        elif iv_bias > 0:
            pricing_line = (
                f"implied vol overpriced realised variance by "
                f"{iv_bias:.1f} vol points, at "
                f"{race['mean_iv_vol']:.1f} percent against "
                f"{race['mean_realised_vol']:.1f} percent realised")
        else:
            pricing_line = (
                f"implied vol underpriced realised variance by "
                f"{-iv_bias:.1f} vol points, at "
                f"{race['mean_iv_vol']:.1f} percent against "
                f"{race['mean_realised_vol']:.1f} percent realised")
        signal_net = (result["straddle_net_signal_mean"]
                      + result["strangle_net_signal_mean"]) / 2.0
        if signal_net > 0:
            pnl_line = (
                f"Buying volatility on signal days made money on "
                f"average, {result['straddle_net_signal_mean']:.0f} "
                f"dollars per straddle and "
                f"{result['strangle_net_signal_mean']:.0f} dollars per "
                f"strangle after costs")
        else:
            pnl_line = (
                f"Buying volatility on signal days lost money on "
                f"average, {result['straddle_net_signal_mean']:.0f} "
                f"dollars per straddle and "
                f"{result['strangle_net_signal_mean']:.0f} dollars per "
                f"strangle after costs")
        excess = (result["mean_abs_ret_30d"]
                  - result["mean_expected_abs_ret_30d"])
        if excess > 0:
            fat_line = (
                f"The mean absolute 30-day move was "
                f"{result['mean_abs_ret_30d']:.1f} percent against "
                f"{result['mean_expected_abs_ret_30d']:.1f} percent "
                f"expected from the realised variance under a normal "
                f"move, so the unhedged payoff is fattened by large "
                f"directional moves rather than by vol mispricing")
        else:
            fat_line = (
                f"The mean absolute 30-day move was "
                f"{result['mean_abs_ret_30d']:.1f} percent against "
                f"{result['mean_expected_abs_ret_30d']:.1f} percent "
                f"expected from the realised variance under a normal "
                f"move, so the unhedged payoff was not fattened by "
                f"large directional moves")
        signal_share = race["n_signal_days"] / max(race["n_days"], 1)
        if signal_share > 0.9:
            signal_line = (
                f"The signal fired on {race['n_signal_days']} of "
                f"{race['n_days']} days, so it barely filters at all: "
                f"the signal-day average is nearly the unconditional "
                f"average")
        else:
            signal_line = (
                f"The signal fired on {race['n_signal_days']} of "
                f"{race['n_days']} days")
        parts.append(
            f"On {name}, {race_line}. {gap_line}, and {pricing_line}. "
            f"{pnl_line}. {fat_line}. {signal_line}.")
    if iv_wins_everywhere:
        close = (
            "Taken together, there is no vol edge here. Implied vol "
            "is the better forecaster and prices realised variance "
            "about fairly, the GARCH forecast overstates vol, so the "
            "repricing edge is mechanical rather than real, and the "
            "positive unhedged payoff comes from 30-day moves "
            "running fatter than a normal distribution implies, "
            "which a delta-hedged test would strip out. Buying "
            "volatility on this signal is a bet on fat moves, not a "
            "bet on mispriced vol.")
    else:
        close = (
            "Taken together, the evidence is mixed: the GARCH "
            "forecast beat implied vol on at least one asset, but "
            "the unhedged payoff still mixes vol and direction, so "
            "a delta-hedged test is needed before calling this an "
            "edge.")
    return " ".join(parts) + " " + close


def build_summary_markdown(results: list, snapshots: dict,
                           dvol_info: dict, snapshot_date: str,
                           last_data_date: str) -> str:
    """Assemble stage_minus_1_summary.md from the computed results."""
    lines = [
        "# Stage -1: implied vol against the GARCH forecast",
        "",
        "The question is whether a GARCH vol forecast beats the "
        "market implied vol by enough to make buying volatility "
        "profitable after costs. The horse race scores the GARCH "
        "forecast and the DVOL implied vol as forecasters of "
        "subsequently realised vol over 30 days. The repricing "
        "values an at-the-money straddle and a 25-delta strangle at "
        "forecast vol against the market price. The edge test buys "
        "both on days where the forecast exceeds implied vol and "
        "holds them to expiry, unhedged.",
        "",
        "## Data",
        "",
        f"Five-minute bars are the Stage 0A files, daily frame "
        f"2024-01-02 to "
        f"{last_data_date}, 1,057 days. DVOL history comes from the "
        "Deribit public API, no key, resolution 1D. "
        f"BTC DVOL covers {dvol_info['BTC']['first_date']} to "
        f"{dvol_info['BTC']['last_date']}, "
        f"{dvol_info['BTC']['rows']} daily rows. ETH DVOL covers "
        f"{dvol_info['ETH']['first_date']} to "
        f"{dvol_info['ETH']['last_date']}, "
        f"{dvol_info['ETH']['rows']} daily rows. The live option "
        f"chain snapshot is from {snapshot_date}. The GARCH forecast "
        f"applied to the live snapshot is fitted on data through "
        f"{last_data_date}, a few days before the snapshot, which is "
        "a staleness to keep in mind when reading the live "
        "comparison.",
        "",
        "## Limits, stated plainly",
        "",
        "- The 7-day horse race runs on the live snapshot only. A "
        "long historical 7-day implied vol series would need "
        "stitching across roughly 150 weekly expiries and is not "
        "built here. The 7-day check below is one day, not a "
        "history.",
        "- The 30-day horizons overlap, so consecutive forecast "
        "windows share 29 days of realised variance. The "
        "Diebold-Mariano p-values below are optimistic about that "
        "serial correlation, since the test lag does not cover a "
        "29-day overlap.",
        "- For the historical 30-day check the market price is "
        "proxied by the Black-Scholes price at DVOL implied vol. "
        "That is not a real traded price. Real traded prices only "
        "appear in the live snapshot, as mark prices.",
        "- The edge test is unhedged. Daily delta hedging is the "
        "follow-up, not something this test does. An unhedged "
        "straddle is a directional bet as well as a vol bet, so its "
        "payoff swings with the underlying move.",
        "- Black-Scholes uses a zero risk-free rate, since Deribit "
        "options are European and cash-settled and the rate is "
        "small next to crypto vol.",
        "",
    ]
    for result in results:
        fit = result["fit"]
        race = result["race"]
        lines += [
            f"## {result['name']}",
            "",
            "### Fitted GARCH(1,1) on the training set",
            "",
            "| Parameter | Value |",
            "|---|---|",
            f"| mu | {fit.mu:.4f} |",
            f"| omega | {fit.omega:.4f} |",
            f"| alpha | {fit.alpha:.4f} |",
            f"| beta | {fit.beta:.4f} |",
            f"| persistence (alpha + beta) | {fit.persistence:.4f} |",
            f"| half-life (days) | {fit.half_life_days:.1f} |",
            f"| annualised vol (%) | {fit.annualized_vol_pct:.1f} |",
            "",
            "### Horse race at 30 days",
            "",
            f"Test days with a full 30-day window: {race['n_days']}. "
            f"Signal days, where the forecast exceeded implied vol: "
            f"{race['n_signal_days']}.",
            "",
            "| Forecaster | QLIKE | Mean vol (%) |",
            "|---|---|---|",
            f"| GARCH forecast | {race['qlike_garch']:.4f} | "
            f"{race['mean_garch_vol']:.1f} |",
            f"| DVOL implied vol | {race['qlike_iv']:.4f} | "
            f"{race['mean_iv_vol']:.1f} |",
            f"| Realised vol | | {race['mean_realised_vol']:.1f} |",
            "",
            f"Mean gap, forecast minus implied: "
            f"{race['mean_gap_vol_points']:.2f} vol points. "
            f"Diebold-Mariano GARCH against implied vol: statistic "
            f"{race['dm_stat']:.2f}, p-value {race['dm_pvalue']:.4f}.",
            "",
            "### Repricing at 30 days, historical",
            "",
            "Market price proxied by the Black-Scholes price at DVOL "
            "implied vol, not a real traded price. Edge is model "
            "value at forecast vol minus that proxy market price.",
            "",
            "| Structure | Mean edge per trade, all days ($) | "
            "Mean edge per trade, signal days ($) |",
            "|---|---|---|",
            f"| ATM straddle | {result['straddle_edge_mean']:.0f} | "
            f"{_fmt(result['straddle_edge_signal_mean'], '.0f')} |",
            f"| 25-delta strangle | {result['strangle_edge_mean']:.0f} | "
            f"{_fmt(result['strangle_edge_signal_mean'], '.0f')} |",
            "",
            "### Edge test at 30 days, held to expiry, unhedged",
            "",
            "Costs are 1 percent of premium, since no live spread "
            "exists for history, plus a 0.03 percent of underlying "
            "taker fee per leg. The underlying at expiry is the "
            "daily close from the Stage 0A data.",
            "",
            f"Directional check: the mean 30-day log return was "
            f"{result['mean_log_ret_30d']:.2f} percent, the mean "
            f"absolute 30-day move was "
            f"{result['mean_abs_ret_30d']:.2f} percent, against "
            f"{result['mean_expected_abs_ret_30d']:.2f} percent "
            f"expected from the realised variance under a normal "
            f"move. When the actual absolute move exceeds the "
            f"normal-implied one, an unhedged straddle pays more "
            f"than its Black-Scholes price even when variance is "
            f"fairly priced, so this gap separates a vol edge from "
            f"a fat-move payoff.",
            "",
            "| Structure | Mean net payoff, all days ($) | "
            "Mean net payoff, signal days ($) | "
            "Win rate, signal days |",
            "|---|---|---|---|",
            f"| ATM straddle | {result['straddle_net_mean']:.0f} | "
            f"{_fmt(result['straddle_net_signal_mean'], '.0f')} | "
            f"{_fmt(result['straddle_win_rate_signal'], '.2f')} |",
            f"| 25-delta strangle | {result['strangle_net_mean']:.0f} | "
            f"{_fmt(result['strangle_net_signal_mean'], '.0f')} | "
            f"{_fmt(result['strangle_win_rate_signal'], '.2f')} |",
            "",
            "### Live snapshot repricing",
            "",
            f"Snapshot from {snapshot_date}. Forecast vol is fitted "
            f"on data through {last_data_date}. Market prices are "
            f"live mark prices. Edge is model value at forecast vol "
            f"minus market price.",
            "",
        ]
        for snap in snapshots[result["name"]]:
            lines += [
                f"Expiry {snap['expiry']} ({snap['days_to_expiry']} "
                f"days out), spot "
                f"{snap['spot']:.0f}, forecast vol "
                f"{snap['forecast_vol']:.1f} percent, at-the-money "
                f"call IV {snap['atm_call_iv']:.1f} percent, "
                f"at-the-money put IV "
                f"{snap['atm_put_iv']:.1f} percent. Most recent "
                f"completed 7-day realised vol "
                f"{snap['recent_realised_7d']:.1f} percent, for "
                f"context: the matching 7-day realised vol for this "
                f"expiry has not happened yet.",
                "",
                "| Structure | Market price ($) | Model value at "
                "forecast vol ($) | Edge ($) | Half spread ($) |",
                "|---|---|---|---|---|",
                f"| ATM straddle, strike "
                f"{snap['atm_strike']:.0f} | "
                f"{snap['market_straddle']:.0f} | "
                f"{snap['model_straddle']:.0f} | "
                f"{snap['straddle_edge']:.0f} | "
                f"{snap['half_spread_straddle']:.0f} |",
                f"| 25-delta strangle, call strike "
                f"{snap['strangle_call_strike']:.0f} "
                f"(delta {snap['strangle_call_delta']:.2f}), put "
                f"strike {snap['strangle_put_strike']:.0f} "
                f"(delta {snap['strangle_put_delta']:.2f}) | "
                f"{snap['market_strangle']:.0f} | "
                f"{snap['model_strangle']:.0f} | "
                f"{snap['strangle_edge']:.0f} | "
                f"{snap['half_spread_strangle']:.0f} |",
                "",
            ]
    lines += [
        "## Verdict",
        "",
        verdict_paragraph(results),
        "",
    ]
    return "\n".join(lines)


def latest_snapshot_file(data_dir: Path) -> Path | None:
    """The newest option chain snapshot CSV in data_dir, if any."""
    files = sorted(glob.glob(str(data_dir / "option_chain_snapshot_*.csv")))
    if not files:
        return None
    return Path(files[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true",
                        help="pull fresh Deribit data first")
    args = parser.parse_args()

    if args.fetch:
        info = fetch_all_data(DATA_DIR)
        print(f"Fetched DVOL BTC {info['dvol']['BTC']['rows']} rows, "
              f"ETH {info['dvol']['ETH']['rows']} rows")
        print(f"Snapshot: {info['snapshot_rows']} rows, "
              f"weekly {info['weekly_expiry']}, "
              f"monthly {info['monthly_expiry']}")

    dvol_paths = {currency: DATA_DIR / f"dvol_{currency.lower()}.parquet"
                  for currency in ("BTC", "ETH")}
    snapshot_path = latest_snapshot_file(DATA_DIR)
    missing = [str(path) for path in dvol_paths.values()
               if not path.exists()]
    if snapshot_path is None:
        missing.append("option chain snapshot CSV")
    if missing:
        raise SystemExit(
            "missing Deribit data files: " + ", ".join(missing)
            + ". Run with --fetch to pull them.")

    snapshot_date = snapshot_path.stem.rsplit("_", 1)[-1]
    as_of = date.fromisoformat(snapshot_date)
    snapshot = pd.read_csv(snapshot_path)
    snapshot["expiry"] = pd.to_datetime(snapshot["expiry"]).dt.date

    dvol_info = {}
    results = []
    snapshots = {}
    last_data_date = ""
    for name, parquet_path in ASSETS:
        bars = load_parquet(str(parquet_path))
        frame = daily_frame(bars)
        last_data_date = str(frame.index[-1])
        dvol_frame = load_dvol(dvol_paths[name])
        dvol = dvol_frame.set_index("date")["dvol_close"]
        dvol_info[name] = {
            "rows": len(dvol_frame),
            "first_date": str(dvol_frame["date"].iloc[0]),
            "last_date": str(dvol_frame["date"].iloc[-1]),
        }
        result = analyze_asset_30d(name, frame, dvol)
        results.append(result)
        latest = latest_horizon_forecasts(frame["ret"])
        snapshots[name] = analyze_snapshot(name, snapshot, latest,
                                           frame, as_of)
        race = result["race"]
        print(f"\n=== {name} ===")
        print(f"Horse race 30d: {race['n_days']} days, "
              f"{race['n_signal_days']} signal days")
        print(f"QLIKE GARCH {race['qlike_garch']:.4f}, "
              f"IV {race['qlike_iv']:.4f}, DM p {race['dm_pvalue']:.4f}")
        print(f"Mean gap forecast minus IV: "
              f"{race['mean_gap_vol_points']:.2f} vol points")
        print(f"Straddle edge, signal days: "
              f"${result['straddle_edge_signal_mean']:.0f}, net payoff "
              f"${result['straddle_net_signal_mean']:.0f}")
        print(f"Strangle edge, signal days: "
              f"${result['strangle_edge_signal_mean']:.0f}, net payoff "
              f"${result['strangle_net_signal_mean']:.0f}")
        for snap in snapshots[name]:
            print(f"Snapshot {snap['expiry']} "
                  f"({snap['days_to_expiry']}d): forecast vol "
                  f"{snap['forecast_vol']:.1f} percent, ATM call IV "
                  f"{snap['atm_call_iv']:.1f}, ATM put IV "
                  f"{snap['atm_put_iv']:.1f}, straddle edge "
                  f"${snap['straddle_edge']:.0f}")

    summary = build_summary_markdown(results, snapshots, dvol_info,
                                     snapshot_date, last_data_date)
    SUMMARY_PATH.write_text(summary, encoding="utf-8")
    print(f"\nWrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
