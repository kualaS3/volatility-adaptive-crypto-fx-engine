"""The two simple forecasters GARCH is compared against.

A model that cannot beat a dumb baseline is not a model worth using,
so both baselines are scored on the same test days with the same
metrics. Constant variance says tomorrow will be as volatile as the
training period was on average. EWMA says tomorrow will be close to
today, with yesterday shock fading at rate lambda. EWMA is the
RiskMetrics standard, lambda 0.94 for daily data, and it is a genuinely
hard baseline to beat on a one-day horizon.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def constant_variance(train_returns: pd.Series) -> float:
    """The training-set variance, one number used for every test day.

    The variance is the sample variance of the training returns, in
    percent squared, matching the units of the GARCH forecasts.
    """
    return float(train_returns.var(ddof=1))


def ewma_variance(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """EWMA variance forecasts, one per day, in percent squared.

    The value dated day t is the forecast for day t, built only from
    returns before t, so there is no lookahead. The recursion is
    sigma_t^2 = lam * sigma_{t-1}^2 + (1 - lam) * r_{t-1}^2, started
    from the variance of the first 20 returns. The first value is
    therefore the variance of those first 20 returns, not a forecast.
    """
    values = np.asarray(returns, dtype=float)
    out = np.empty_like(values)
    warmup = values[:20]
    out[0] = float(np.var(warmup, ddof=1))
    for t in range(1, len(values)):
        out[t] = lam * out[t - 1] + (1.0 - lam) * values[t - 1] ** 2
    return pd.Series(out, index=returns.index)
