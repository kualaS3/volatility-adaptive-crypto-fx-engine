"""Tests for the multi-horizon GARCH forecast vol."""

import numpy as np
import pandas as pd

from horizon import (
    horizon_annualised_vol,
    horizon_total_variance,
    latest_horizon_forecasts,
    rolling_horizon_forecasts,
)


def test_horizon_total_variance_matches_hand_recursion():
    """Closed form total variance against a hand-stepped recursion.

    The hand recursion steps the expected variance forward one day
    at a time: each step is omega plus persistence times the
    previous expected variance, since the expected squared return
    equals the expected variance. Summing those steps over the
    horizon must match the closed form.
    """
    omega = 0.5
    alpha = 0.1
    beta = 0.8
    var_one_step = 4.0
    horizon = 5
    expected = var_one_step
    total = 0.0
    for _ in range(horizon):
        total += expected
        expected = omega + (alpha + beta) * expected
    result = horizon_total_variance(omega, alpha, beta, var_one_step,
                                    horizon)
    assert abs(result - total) < 1e-9


def test_horizon_total_variance_unit_persistence():
    """With persistence 1 each step adds omega to the variance."""
    omega = 0.5
    alpha = 0.2
    beta = 0.8
    var_one_step = 4.0
    horizon = 4
    expected = var_one_step
    total = 0.0
    for _ in range(horizon):
        total += expected
        expected += omega
    result = horizon_total_variance(omega, alpha, beta, var_one_step,
                                    horizon)
    assert abs(result - total) < 1e-9


def test_one_day_horizon_is_the_one_step_vol():
    """A 1-day horizon annualises to the one-step vol itself."""
    var_one_step = 6.76
    vol = horizon_annualised_vol(0.5, 0.1, 0.8, var_one_step, 1)
    expected = var_one_step ** 0.5 * 365.0 ** 0.5
    assert abs(vol - expected) < 1e-9


def test_longer_horizon_pulls_toward_unconditional():
    """A long horizon vol sits between the one-step and the mean.

    With a one-step variance far above the unconditional variance,
    the 30-day vol is below the 1-day vol and above the
    unconditional annualised vol, since the shock decays over the
    horizon.
    """
    omega = 0.5
    alpha = 0.1
    beta = 0.8
    var_one_step = 25.0
    var_uncond = omega / (1.0 - alpha - beta)
    vol_1d = horizon_annualised_vol(omega, alpha, beta, var_one_step, 1)
    vol_30d = horizon_annualised_vol(omega, alpha, beta, var_one_step, 30)
    vol_uncond = var_uncond ** 0.5 * 365.0 ** 0.5
    assert vol_uncond < vol_30d < vol_1d


def _synthetic_returns(n=300, seed=7):
    rng = np.random.default_rng(seed)
    values = rng.normal(0.0, 3.0, size=n)
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    return pd.Series(values, index=dates.date)


def test_rolling_horizon_forecasts_shape_and_sign():
    """Rolling forecasts cover the test set with positive vols."""
    returns = _synthetic_returns()
    n_train = 240
    frame = rolling_horizon_forecasts(returns, n_train, horizons=(7, 30),
                                      refit_every=20)
    assert len(frame) == len(returns) - n_train
    assert list(frame.columns) == ["vol_7d", "vol_30d"]
    assert frame.index[0] == returns.index[n_train]
    assert np.isfinite(frame.values).all()
    assert (frame.values > 0).all()


def test_latest_horizon_forecasts_keys():
    """The latest forecast returns params and vols per horizon."""
    returns = _synthetic_returns()
    result = latest_horizon_forecasts(returns, horizons=(7, 30))
    assert set(result["vols"].keys()) == {7, 30}
    assert result["vols"][7] > 0
    assert result["vols"][30] > 0
    assert 0.0 < result["persistence"] < 1.5
