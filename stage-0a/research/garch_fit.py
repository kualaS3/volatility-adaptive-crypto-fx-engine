"""The GARCH(1,1) model and the numbers derived from its fit.

The model says the return on day t is a constant mean plus noise whose
variance moves over time:

    r_t = mu + sigma_t * z_t,  z_t standard normal
    sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2

omega is the floor the variance reverts to, alpha is how hard yesterday
shocks the variance today, and beta is how much of yesterday variance
carries over. alpha plus beta is the persistence: how slowly a shock
dies out. When persistence is close to 1, volatility stays high for a
long time after a shock, which is the usual finding for crypto.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model


@dataclass
class FitSummary:
    """One fitted GARCH(1,1), with the derived quantities filled in."""

    mu: float
    omega: float
    alpha: float
    beta: float
    persistence: float
    half_life_days: float
    unconditional_daily_vol_pct: float
    annualized_vol_pct: float


def fit_garch(returns: pd.Series):
    """Fit a GARCH(1,1) with a constant mean and normal errors.

    Returns the fitted result object from the arch package. The caller
    reads params and conditional volatility off it, or passes it to
    summarize_fit and forecast_next_variance below.
    """
    model = arch_model(returns, mean="Constant", vol="Garch", p=1, q=1,
                       dist="normal")
    return model.fit(disp="off")


def forecast_next_variance(fitted_result) -> float:
    """One-step-ahead variance forecast for the next day, in percent squared."""
    forecast = fitted_result.forecast(horizon=1)
    return float(forecast.variance.iloc[-1, 0])


def next_variance(omega: float, alpha: float, beta: float,
                  last_variance: float, last_return: float) -> float:
    """One step of the GARCH variance recursion.

    The variance for tomorrow is omega plus alpha times the squared
    return of today plus beta times the variance of today. This is
    the step the model itself takes, and it is what lets a forecast
    react to each new return without refitting the parameters.
    """
    return omega + alpha * last_return ** 2 + beta * last_variance


def rolling_forecasts(returns: pd.Series, n_train: int,
                      refit_every: int = 20) -> pd.Series:
    """One-step variance forecasts over the test set, refit as it goes.

    The model is refit on an expanding window every `refit_every`
    test days, always on returns strictly before the day being
    forecast, so there is no lookahead. Between refits the forecast
    still moves every day: it steps forward with next_variance, using
    each new return as it arrives. Returns a Series of forecast
    variances in percent squared, indexed by the test dates.
    """
    test = returns.iloc[n_train:]
    forecasts = []
    fitted = None
    last_variance = None
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
        forecasts.append(last_variance)
    return pd.Series(forecasts, index=test.index)


def summarize_fit(fitted_result, annual_days: int = 365) -> FitSummary:
    """Pull the parameters out of a fit and derive the readable numbers.

    Persistence is alpha plus beta. The half-life is how many days a
    variance shock takes to halve, from ln(0.5) / ln(persistence).
    The unconditional daily vol is sqrt(omega / (1 - persistence)), in
    percent per day. The annualised vol multiplies by sqrt(annual_days):
    365 for crypto, which trades every day of the year, and 252 for
    FX, which trades on business days.
    """
    params = fitted_result.params
    mu = float(params["mu"])
    omega = float(params["omega"])
    alpha = float(params["alpha[1]"])
    beta = float(params["beta[1]"])
    persistence = alpha + beta
    if persistence >= 1.0:
        half_life = float("inf")
    else:
        half_life = float(np.log(0.5) / np.log(persistence))
    if persistence >= 1.0:
        unconditional = float("nan")
    else:
        unconditional = float(np.sqrt(omega / (1.0 - persistence)))
    return FitSummary(
        mu=mu,
        omega=omega,
        alpha=alpha,
        beta=beta,
        persistence=persistence,
        half_life_days=half_life,
        unconditional_daily_vol_pct=unconditional,
        annualized_vol_pct=unconditional * float(np.sqrt(annual_days)),
    )
