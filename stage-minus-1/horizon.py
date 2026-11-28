"""GARCH(1,1) multi-horizon forecast volatility.

Stage 0A forecasts one day ahead. Stage -1 needs the same forecast
over a whole option horizon, 7 days for a weekly expiry and 30 days
for a monthly expiry. The multi-step forecast is analytic. The
expected variance h days ahead is the unconditional variance plus
persistence to the power h minus 1 times the gap between the
one-step forecast variance and the unconditional variance:

    E[sigma_{t+h}^2] = var_bar + persistence^(h-1) * (var_1 - var_bar)

Summing that over the horizon gives the total expected variance,
which is annualised with sqrt(365 / horizon_days) for crypto.

The same discipline as Stage 0A applies: rolling refits every 20
days on an expanding window, no lookahead, and a forecast for a
horizon uses only returns strictly before that horizon starts.
The Stage 0A modules are reused through a documented sys.path
insert pointing at stage-0a/research, which is where data.py,
realized.py, garch_fit.py and evaluate.py live.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_STAGE0A_RESEARCH = Path(__file__).resolve().parents[1] / "stage-0a" / "research"
if str(_STAGE0A_RESEARCH) not in sys.path:
    sys.path.insert(0, str(_STAGE0A_RESEARCH))

from garch_fit import fit_garch, next_variance  # noqa: E402


def horizon_total_variance(omega: float, alpha: float, beta: float,
                           var_one_step: float,
                           horizon_days: int) -> float:
    """Total expected variance over the next horizon_days days.

    The inputs are in percent squared, the same units as the Stage 0A
    one-step forecast variance. The closed form sums the analytic
    multi-step expectation over the horizon. When persistence is 1,
    the unconditional variance does not exist, so the fallback sums
    the unit-persistence recursion instead: each step adds omega to
    the previous expected variance.
    """
    persistence = alpha + beta
    if persistence >= 1.0:
        total = 0.0
        expected = var_one_step
        for _ in range(horizon_days):
            total += expected
            expected += omega
        return total
    var_uncond = omega / (1.0 - persistence)
    geometric = (1.0 - persistence ** horizon_days) / (1.0 - persistence)
    return horizon_days * var_uncond + (var_one_step - var_uncond) * geometric


def horizon_annualised_vol(omega: float, alpha: float, beta: float,
                           var_one_step: float, horizon_days: int,
                           annual_days: int = 365) -> float:
    """Annualised forecast vol in percent over a horizon.

    Takes the total expected variance over the horizon and
    annualises it with sqrt(annual_days / horizon_days). Crypto
    uses 365 days, since it trades every day of the year.
    """
    total = horizon_total_variance(omega, alpha, beta, var_one_step,
                                   horizon_days)
    return float((total ** 0.5) * (annual_days / horizon_days) ** 0.5)


def rolling_horizon_forecasts(returns: pd.Series, n_train: int,
                              horizons: tuple = (7, 30),
                              refit_every: int = 20,
                              annual_days: int = 365) -> pd.DataFrame:
    """Annualised multi-horizon forecast vols over the test set.

    Mirrors the Stage 0A rolling_forecasts loop exactly: the model is
    refit on an expanding window every refit_every test days, always
    on returns strictly before the day being forecast, and between
    refits the one-step variance steps forward with next_variance
    using each new return as it arrives. Each test day then gets an
    annualised forecast vol per horizon, from the analytic multi-step
    formula applied to that day one-step variance and the current
    fit parameters. Returns a DataFrame indexed by the test dates
    with one vol_Hd column per horizon, in percent.
    """
    test = returns.iloc[n_train:]
    rows = []
    fitted = None
    last_variance = None
    omega = alpha = beta = float("nan")
    for i in range(len(test)):
        if i % refit_every == 0:
            fitted = fit_garch(returns.iloc[: n_train + i])
            params = fitted.params
            omega = float(params["omega"])
            alpha = float(params["alpha[1]"])
            beta = float(params["beta[1]"])
            last_variance = float(fitted.conditional_volatility.iloc[-1] ** 2)
            last_return = float(returns.iloc[n_train + i - 1])
        else:
            last_return = float(test.iloc[i - 1])
        last_variance = next_variance(omega, alpha, beta,
                                      last_variance, last_return)
        row = {}
        for horizon in horizons:
            row[f"vol_{horizon}d"] = horizon_annualised_vol(
                omega, alpha, beta, last_variance, horizon,
                annual_days=annual_days)
        rows.append(row)
    return pd.DataFrame(rows, index=test.index)


def latest_horizon_forecasts(returns: pd.Series,
                             horizons: tuple = (7, 30),
                             annual_days: int = 365) -> dict:
    """Horizon vols forecast from the last day of a returns series.

    Fits GARCH(1,1) on the whole series, steps the variance one day
    forward with next_variance, and annualises the multi-step
    forecast per horizon. Returns a dict with the fit parameters
    and a vols dict keyed by horizon in days. This is the forecast
    applied to the live option snapshot, and it is dated by the last
    return in the series, which is stated wherever it is used.
    """
    fitted = fit_garch(returns)
    params = fitted.params
    omega = float(params["omega"])
    alpha = float(params["alpha[1]"])
    beta = float(params["beta[1]"])
    last_variance = float(fitted.conditional_volatility.iloc[-1] ** 2)
    last_return = float(returns.iloc[-1])
    var_one_step = next_variance(omega, alpha, beta,
                                 last_variance, last_return)
    vols = {}
    for horizon in horizons:
        vols[horizon] = horizon_annualised_vol(
            omega, alpha, beta, var_one_step, horizon,
            annual_days=annual_days)
    return {
        "mu": float(params["mu"]),
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "persistence": alpha + beta,
        "var_one_step": var_one_step,
        "vols": vols,
    }
