"""Tests for the GARCH fit and the baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from baselines import constant_variance, ewma_variance
from garch_fit import fit_garch, rolling_forecasts, summarize_fit


def test_ewma_recursion_against_hand_computation():
    # Returns 1, 2, 3, 4 with lambda 0.5. The warmup variance is the
    # variance of the first 20 returns, which here is the variance of
    # all four, and each later value follows the recursion exactly.
    returns = pd.Series([1.0, 2.0, 3.0, 4.0])
    out = ewma_variance(returns, lam=0.5)
    warmup = float(np.var([1.0, 2.0, 3.0, 4.0], ddof=1))
    expected = [warmup]
    for t in range(1, 4):
        expected.append(0.5 * expected[-1] + 0.5 * returns.iloc[t - 1] ** 2)
    assert out.tolist() == [pytest.approx(v) for v in expected]


def test_constant_variance_is_the_training_variance():
    returns = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    assert constant_variance(returns) == pytest.approx(
        float(returns.var(ddof=1))
    )


def test_garch_recovers_known_parameters_on_simulated_data():
    # Simulate a GARCH(1,1) with omega 0.1, alpha 0.1, beta 0.85 on
    # percent returns, then fit and check the persistence comes back
    # near the true 0.95. Estimation noise is real, so the tolerance
    # is loose: the point is the machinery works, not that it is exact.
    rng = np.random.default_rng(7)
    n = 1000
    omega, alpha, beta = 0.1, 0.1, 0.85
    sigma2 = np.empty(n)
    returns = np.empty(n)
    sigma2[0] = omega / (1.0 - alpha - beta)
    returns[0] = np.sqrt(sigma2[0]) * rng.standard_normal()
    for t in range(1, n):
        sigma2[t] = omega + alpha * returns[t - 1] ** 2 + beta * sigma2[t - 1]
        returns[t] = np.sqrt(sigma2[t]) * rng.standard_normal()
    fitted = fit_garch(pd.Series(returns))
    summary = summarize_fit(fitted)
    assert summary.omega > 0.0
    assert abs(summary.persistence - 0.95) < 0.15
    assert summary.half_life_days > 0.0


def test_rolling_forecasts_shape_and_positivity():
    # A short synthetic return series: the forecasts must cover the
    # test set exactly, all be positive, and the first forecast must
    # use only training data, so it is finite.
    rng = np.random.default_rng(3)
    returns = pd.Series(rng.standard_normal(100) * 2.0)
    forecasts = rolling_forecasts(returns, n_train=80, refit_every=20)
    assert len(forecasts) == 20
    assert (forecasts > 0).all()
    assert np.isfinite(forecasts).all()
