"""Scoring the forecasts and checking the fit.

The realised variance for one day is the sum of the squared 5-minute
returns inside that day, built by research.realized. It is still a
noisy proxy for the true variance, so QLIKE is the headline metric:
it is the loss function that ranks variance forecasts correctly even
when the proxy is noisy. MSE is reported alongside it, but it is
dominated by the few biggest days and is not trusted on its own. The
Mincer-Zarnowitz regression checks calibration: realised variance
regressed on forecast variance should have slope near 1 and intercept
near 0 when the forecasts are unbiased. The Diebold-Mariano test asks
whether the QLIKE gap between two models is bigger than noise: with
only a few hundred test days, a small edge can easily be chance, so
the test is run in Stage 0A already, not left for Stage 0C.

The diagnostics check whether the GARCH fit left anything on the
table. If the squared standardised residuals still cluster, the model
missed volatility dynamics. Both tests report a p-value: below 0.05
the leftover structure is unlikely to be chance.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch


def _as_array(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def qlike(realized_var, forecast_var) -> float:
    """Mean QLIKE loss: mean of RV / sigma^2 - ln(RV / sigma^2) - 1.

    Lower is better. QLIKE punishes an over-forecast and an
    under-forecast asymmetrically, which is the right shape for a
    variance loss, and it stays valid when the realised proxy is
    noisy, unlike MSE. This is the plan form of the loss. It differs
    from the sibling project form, mean of RV / sigma^2 + ln(sigma^2),
    by -ln(RV) - 1, which is the same for every model scored on the
    same test days, so the ranking of models is identical either way.
    """
    realized = _as_array(realized_var)
    forecast = _as_array(forecast_var)
    ratio = realized / forecast
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def qlike_losses(realized_var, forecast_var) -> np.ndarray:
    """Per-day QLIKE losses, one per test day.

    The mean of these is qlike above. The per-day series is what the
    Diebold-Mariano test compares between two models.
    """
    realized = _as_array(realized_var)
    forecast = _as_array(forecast_var)
    ratio = realized / forecast
    return ratio - np.log(ratio) - 1.0


def mse(realized_var, forecast_var) -> float:
    """Mean squared error between realised and forecast variance.

    Lower is better, but a handful of huge return days dominate it,
    so it is a secondary metric here, not the headline.
    """
    realized = _as_array(realized_var)
    forecast = _as_array(forecast_var)
    return float(np.mean((realized - forecast) ** 2))


def mincer_zarnowitz(realized_var, forecast_var) -> dict:
    """Regress realised variance on forecast variance.

    Returns a dict with r_squared, slope and intercept. A slope near
    1 and an intercept near 0 mean the forecasts are unbiased: on
    average, a forecast of X is followed by a realised variance of X.
    A slope below 1 means the forecasts overreact, swinging too wide.
    When the forecast never varies, as with the constant baseline,
    the slope is unidentified, so slope and intercept come back as
    NaN and r_squared is 0: a flat forecast explains nothing.
    """
    realized = _as_array(realized_var)
    forecast = _as_array(forecast_var)
    if float(np.ptp(forecast)) == 0.0:
        return {"r_squared": 0.0, "slope": float("nan"),
                "intercept": float("nan")}
    design = sm.add_constant(forecast, has_constant="add")
    result = sm.OLS(realized, design).fit()
    return {
        "r_squared": float(result.rsquared),
        "slope": float(result.params[1]),
        "intercept": float(result.params[0]),
    }


def diebold_mariano(loss_a, loss_b) -> dict:
    """Diebold-Mariano test on two per-day loss series.

    Returns a dict with statistic and p_value (two-sided). The test
    compares the mean loss difference against a Newey-West
    heteroskedasticity and autocorrelation consistent standard error
    with a Bartlett kernel, so serial correlation in the daily loss
    differences does not inflate the result. The lag follows the
    Newey-West rule, floor(4 * (T / 100) ^ (2 / 9)). A negative
    statistic means the first model (a) had the lower mean loss.
    The null is equal predictive accuracy. A small p-value says the
    gap is bigger than noise.
    """
    a = _as_array(loss_a)
    b = _as_array(loss_b)
    d = a - b
    n = len(d)
    mean = float(np.mean(d))
    lag = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    var = float(np.var(d, ddof=1))
    hac = var
    for k in range(1, lag + 1):
        weight = 1.0 - k / (lag + 1.0)
        cov = float(np.mean((d[k:] - mean) * (d[:-k] - mean)))
        hac += 2.0 * weight * cov
    se = float(np.sqrt(hac / n))
    if se == 0.0:
        # No variation in the loss difference. A zero mean difference
        # means the models tied exactly. A nonzero one means the gap
        # is certain rather than noise.
        if mean == 0.0:
            return {"statistic": 0.0, "p_value": 1.0}
        return {
            "statistic": float(np.sign(mean) * np.inf),
            "p_value": 0.0,
        }
    stat = mean / se
    p_value = float(2.0 * norm.sf(abs(stat)))
    return {"statistic": stat, "p_value": p_value}


def ljung_box_sq(standardized_residuals, lags: int = 10) -> tuple:
    """Ljung-Box test on squared standardised residuals.

    Returns (statistic, p-value) at the requested lag. A small
    p-value means the squared residuals still cluster, so the model
    missed some volatility dynamics.
    """
    values = _as_array(standardized_residuals)
    table = acorr_ljungbox(values ** 2, lags=[lags], return_df=True)
    return float(table["lb_stat"].iloc[0]), float(table["lb_pvalue"].iloc[0])


def arch_lm(standardized_residuals, lags: int = 10) -> tuple:
    """Engle ARCH-LM test on standardised residuals.

    Returns (statistic, p-value) at the requested lag. A small
    p-value means ARCH effects remain in the residuals, so the
    conditional variance model is incomplete.
    """
    values = _as_array(standardized_residuals)
    with warnings.catch_warnings():
        # het_arch warns that a future statsmodels release changes its
        # return type. The tuple it returns today is what we use.
        warnings.simplefilter("ignore", FutureWarning)
        stat, pvalue, _, _ = het_arch(values, nlags=lags)
    return float(stat), float(pvalue)
