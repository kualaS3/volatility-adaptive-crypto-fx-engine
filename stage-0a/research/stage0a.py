"""Stage 0A: EWMA vs vanilla GARCH(1,1) on daily realized variance.

For each asset, the five crypto assets then EUR/USD, the pipeline is:

1. Load the 5-minute bars from research/data and build the daily
   frame: percent log returns from the daily closes and daily
   realized variance from the squared 5-minute returns.
2. Split 80/20. The first 80 percent of days train the model, the
   last 20 percent are the out-of-sample test set.
3. Fit GARCH(1,1) on the training daily returns and report its
   parameters.
4. Forecast one day ahead across the test set, refitting on an
   expanding window every 20 days, so no forecast ever sees a return
   from its own day or later.
5. Score the GARCH forecasts against a constant-variance baseline
   and an EWMA baseline on the same test days, with QLIKE, MSE and a
   Mincer-Zarnowitz regression, all against next-day realized
   variance. A Diebold-Mariano test then asks whether the QLIKE gap
   between GARCH and the best baseline is bigger than noise.
6. Run the Ljung-Box and ARCH-LM diagnostics on the in-sample
   standardised residuals.
7. Write research/stage0a_summary.md.

Run from the repo root: python -m research.stage0a
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from baselines import constant_variance, ewma_variance
from data import load_parquet
from evaluate import (
    arch_lm,
    diebold_mariano,
    ljung_box_sq,
    mincer_zarnowitz,
    mse,
    qlike,
    qlike_losses,
)
from garch_fit import fit_garch, rolling_forecasts, summarize_fit
from realized import daily_frame

TRAIN_FRACTION = 0.80
REFIT_EVERY = 20
# The top five cryptocurrencies by market capitalisation, excluding
# stablecoins, in ranking order (CoinGecko, 2026-11-24), then EUR/USD.
# Crypto annualises on 365 days, FX on 252 trading days.
ASSETS = [
    ("BTC", "research/data/btc_usd_5m.parquet", 365),
    ("ETH", "research/data/eth_usd_5m.parquet", 365),
    ("BNB", "research/data/bnb_usd_5m.parquet", 365),
    ("XRP", "research/data/xrp_usd_5m.parquet", 365),
    ("SOL", "research/data/sol_usd_5m.parquet", 365),
    ("EUR/USD", "research/data/eurusd_5m.parquet", 252),
]

FX_BLOCKED_NOTE = (
    "EUR/USD is not in the table above: no free 2-year 5-minute source "
    "is reachable from this machine. Dukascopy datafeed answers a few "
    "requests, then returns HTTP 429, then stops answering. histdata.com "
    "does not answer. Kraken EURUSD OHLC answers, but its `since` "
    "pagination is erratic: repeated calls return inconsistent, "
    "overlapping windows that never reach before the last few days, so "
    "no trustworthy 2-year series can be built from it. Yahoo Finance "
    "intraday bars only go back about 60 days. The pipeline skips a "
    "missing data file with a warning, so EUR/USD slots in the moment "
    "a source is found."
)


def analyze_asset(name: str, parquet_path: Path,
                  annual_days: int) -> dict:
    """Run the whole pipeline for one asset and return its results."""
    bars = load_parquet(str(parquet_path))
    frame = daily_frame(bars)
    returns = frame["ret"]
    realized_all = frame["rv"]
    n_days = len(frame)
    n_train = int(n_days * TRAIN_FRACTION)
    train = returns.iloc[:n_train]
    test = returns.iloc[n_train:]
    realized = realized_all.iloc[n_train:]

    fitted = fit_garch(train)
    summary = summarize_fit(fitted, annual_days=annual_days)
    cond_vol = fitted.conditional_volatility
    standardised = train / cond_vol
    lb_stat, lb_pvalue = ljung_box_sq(standardised, lags=10)
    arch_stat, arch_pvalue = arch_lm(standardised, lags=10)

    garch_var = rolling_forecasts(returns, n_train, refit_every=REFIT_EVERY)
    const_var = pd.Series(
        np.full(len(test), constant_variance(train)), index=test.index
    )
    ewma_var = ewma_variance(returns).iloc[n_train:]

    forecasts = {
        "GARCH(1,1)": garch_var,
        "Constant": const_var,
        "EWMA (0.94)": ewma_var,
    }
    metrics = {}
    for label, forecast in forecasts.items():
        metrics[label] = {
            "qlike": qlike(realized, forecast),
            "mse": mse(realized, forecast),
            "mz": mincer_zarnowitz(realized, forecast),
        }
    best_label = min(
        ("Constant", "EWMA (0.94)"),
        key=lambda label: metrics[label]["qlike"],
    )
    best_var = const_var if best_label == "Constant" else ewma_var
    dm = diebold_mariano(
        qlike_losses(realized, garch_var),
        qlike_losses(realized, best_var),
    )

    return {
        "name": name,
        "first_date": str(frame.index[0]),
        "last_date": str(frame.index[-1]),
        "n_bars": len(bars),
        "n_days": n_days,
        "n_train": n_train,
        "n_test": len(test),
        "bars_per_day_median": float(frame["n_bars"].median()),
        "bars_per_day_min": int(frame["n_bars"].min()),
        "summary": summary,
        "lb_stat": lb_stat,
        "lb_pvalue": lb_pvalue,
        "arch_stat": arch_stat,
        "arch_pvalue": arch_pvalue,
        "metrics": metrics,
        "best_baseline": best_label,
        "dm_stat": dm["statistic"],
        "dm_pvalue": dm["p_value"],
    }


def _fmt(value: float, spec: str) -> str:
    """Format a float, or n/a when it is not a number.

    The constant baseline has no identified Mincer-Zarnowitz slope,
    so its slope and intercept are NaN and print as n/a.
    """
    if value != value:  # NaN is the only float unequal to itself
        return "n/a"
    return format(value, spec)


def print_asset(result: dict) -> None:
    """Print one asset's fitted parameters and metric table."""
    s = result["summary"]
    print(f"\n=== {result['name']} ===")
    print(f"Data: {result['n_bars']} 5-minute bars, "
          f"{result['first_date']} to {result['last_date']}, "
          f"{result['n_days']} days, {result['n_train']} train / "
          f"{result['n_test']} test, bars per day median "
          f"{result['bars_per_day_median']:.0f} min "
          f"{result['bars_per_day_min']}")
    print(f"mu={s.mu:.4f}  omega={s.omega:.4f}  alpha={s.alpha:.4f}  "
          f"beta={s.beta:.4f}")
    print(f"persistence={s.persistence:.4f}  "
          f"half-life={s.half_life_days:.1f} days  "
          f"unconditional vol={s.unconditional_daily_vol_pct:.2f}%/day  "
          f"annualised vol={s.annualized_vol_pct:.1f}%")
    print(f"Ljung-Box on squared residuals (lag 10): "
          f"stat={result['lb_stat']:.2f} p={result['lb_pvalue']:.4f}")
    print(f"ARCH-LM on residuals (lag 10): "
          f"stat={result['arch_stat']:.2f} p={result['arch_pvalue']:.4f}")
    print(f"{'model':<12} {'QLIKE':>8} {'MSE':>10} "
          f"{'MZ R2':>7} {'MZ slope':>9} {'MZ int.':>8}")
    for label, m in result["metrics"].items():
        print(f"{label:<12} {m['qlike']:>8.4f} {m['mse']:>10.2f} "
              f"{m['mz']['r_squared']:>7.3f} "
              f"{_fmt(m['mz']['slope'], '.3f'):>9} "
              f"{_fmt(m['mz']['intercept'], '.3f'):>8}")
    print(f"Diebold-Mariano GARCH vs {result['best_baseline']}: "
          f"stat={result['dm_stat']:.2f} p={result['dm_pvalue']:.4f}")


def interpretation(result: dict) -> str:
    """One plain paragraph saying whether GARCH beat the baselines.

    The verdict is read off the QLIKE numbers, not written by hand:
    the best baseline is whichever of Constant and EWMA scored lower,
    and the improvement is the percentage drop in QLIKE from that
    baseline to GARCH. The Diebold-Mariano p-value says whether that
    gap is bigger than noise. The Mincer-Zarnowitz slope says whether
    the GARCH forecasts were calibrated.
    """
    name = result["name"]
    garch = result["metrics"]["GARCH(1,1)"]
    best = result["metrics"][result["best_baseline"]]
    improvement = 100.0 * (best["qlike"] - garch["qlike"]) / best["qlike"]
    if improvement > 0:
        verdict = (
            f"GARCH beat the best baseline ({result['best_baseline']}) "
            f"by {improvement:.1f}% on QLIKE"
        )
    else:
        verdict = (
            f"GARCH did not beat the best baseline "
            f"({result['best_baseline']}). It was {-improvement:.1f}% "
            f"worse on QLIKE"
        )
    if result["dm_pvalue"] < 0.05:
        significance = (
            f"The Diebold-Mariano p-value of {result['dm_pvalue']:.3f} "
            f"is below 0.05, so the gap is bigger than noise and the "
            f"edge is statistically significant"
        )
    else:
        significance = (
            f"The Diebold-Mariano p-value of {result['dm_pvalue']:.3f} "
            f"is above 0.05, so the gap is within noise and the edge "
            f"is not statistically significant on this test set"
        )
    slope = garch["mz"]["slope"]
    if 0.8 <= slope <= 1.2:
        calibration = (
            f"The Mincer-Zarnowitz slope of {slope:.2f} is near 1, so "
            f"the GARCH forecasts were roughly calibrated"
        )
    elif slope < 0.8:
        calibration = (
            f"The Mincer-Zarnowitz slope of {slope:.2f} is below 1, so "
            f"the GARCH forecasts overreacted a little: they swung "
            f"wider than the realised variance did"
        )
    else:
        calibration = (
            f"The Mincer-Zarnowitz slope of {slope:.2f} is above 1, so "
            f"the GARCH forecasts under-reacted: they moved less than "
            f"the realised variance did"
        )
    return (
        f"On {name}, {verdict}. The constant-variance baseline scored "
        f"{result['metrics']['Constant']['qlike']:.3f} on QLIKE and EWMA "
        f"scored {result['metrics']['EWMA (0.94)']['qlike']:.3f}, against "
        f"{garch['qlike']:.3f} for GARCH. {significance}. {calibration}, "
        f"and the R-squared of {garch['mz']['r_squared']:.3f} reflects "
        f"how much of the day-to-day realised variance a variance "
        f"forecast can explain."
    )


def build_summary_markdown(results: list, skipped: list) -> str:
    """Assemble research/stage0a_summary.md from the computed results.

    Assets listed in ASSETS whose data file is missing are named in a
    note at the end, so the write-up records what did not run and why
    instead of silently dropping it.
    """
    lines = [
        "# Stage 0A: EWMA vs GARCH(1,1) on daily realized variance",
        "",
        "Out-of-sample variance forecasts from a GARCH(1,1) fitted on "
        "percent daily log returns, scored against a constant-variance "
        "baseline and an EWMA baseline with lambda 0.94. The realised "
        "variance each day is the sum of the squared 5-minute log "
        "returns inside that day, in percent squared. QLIKE is the "
        "headline metric and MSE is reported alongside it. A "
        "Diebold-Mariano test on the per-day QLIKE losses asks whether "
        "the GARCH edge over the best baseline is bigger than noise.",
        "",
    ]
    for result in results:
        s = result["summary"]
        lines += [
            f"## {result['name']}",
            "",
            f"Data: {result['n_bars']} 5-minute bars, "
            f"{result['first_date']} to {result['last_date']}, "
            f"{result['n_days']} days, {result['n_train']} in the "
            f"training set and {result['n_test']} in the test set. Bars "
            f"per day: median {result['bars_per_day_median']:.0f}, "
            f"minimum {result['bars_per_day_min']}.",
            "",
            "### Fitted GARCH(1,1)",
            "",
            "| Parameter | Value |",
            "|---|---|",
            f"| mu | {s.mu:.4f} |",
            f"| omega | {s.omega:.4f} |",
            f"| alpha | {s.alpha:.4f} |",
            f"| beta | {s.beta:.4f} |",
            f"| persistence (alpha + beta) | {s.persistence:.4f} |",
            f"| half-life (days) | {s.half_life_days:.1f} |",
            f"| unconditional daily vol (%) | "
            f"{s.unconditional_daily_vol_pct:.2f} |",
            f"| annualised vol (%) | {s.annualized_vol_pct:.1f} |",
            "",
            "### Diagnostics on the in-sample fit",
            "",
            f"Ljung-Box on squared standardised residuals, lag 10: "
            f"statistic {result['lb_stat']:.2f}, "
            f"p-value {result['lb_pvalue']:.4f}. "
            f"ARCH-LM on standardised residuals, lag 10: statistic "
            f"{result['arch_stat']:.2f}, "
            f"p-value {result['arch_pvalue']:.4f}.",
            "",
            "### Out-of-sample metrics",
            "",
            "| Model | QLIKE | MSE | MZ R-squared | MZ slope | MZ intercept |",
            "|---|---|---|---|---|---|",
        ]
        for label, m in result["metrics"].items():
            lines.append(
                f"| {label} | {m['qlike']:.4f} | {m['mse']:.2f} | "
                f"{m['mz']['r_squared']:.3f} | "
                f"{_fmt(m['mz']['slope'], '.3f')} | "
                f"{_fmt(m['mz']['intercept'], '.3f')} |"
            )
        lines += [
            "",
            f"Diebold-Mariano GARCH vs {result['best_baseline']}: "
            f"statistic {result['dm_stat']:.2f}, "
            f"p-value {result['dm_pvalue']:.4f}.",
            "",
            interpretation(result),
            "",
        ]
    if skipped:
        lines += [
            "## Skipped assets",
            "",
            f"The following listed assets had no data file and were "
            f"skipped: {', '.join(name for name, _, _ in skipped)}.",
            "",
            FX_BLOCKED_NOTE,
            "",
        ]
    return "\n".join(lines)


def available_assets(assets, root: Path) -> list:
    """The assets whose data file exists, warning about the rest.

    An asset with no Parquet file is skipped with a plain warning
    rather than crashing the run: EUR/USD has no usable free 5-minute
    source from this machine yet, so Stage 0A runs on the crypto assets
    alone until an FX source is found. The skipped asset stays listed
    in ASSETS, so it slots back in the moment its file appears.
    """
    available = []
    for asset in assets:
        name, relative_path, annual_days = asset
        if (root / relative_path).exists():
            available.append(asset)
        else:
            print(f"WARNING: skipping {name}: no data file at "
                  f"{relative_path}")
    return available


def main() -> None:
    assets = available_assets(ASSETS, ROOT)
    if not assets:
        raise SystemExit("no data files found under research/data: run "
                         "scripts/fetch_research_data.py first")
    skipped = [asset for asset in ASSETS if asset not in assets]
    results = []
    for name, relative_path, annual_days in assets:
        result = analyze_asset(name, ROOT / relative_path, annual_days)
        results.append(result)
        print_asset(result)
    summary_path = ROOT / "research" / "stage0a_summary.md"
    summary_path.write_text(
        build_summary_markdown(results, skipped), encoding="utf-8"
    )
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
