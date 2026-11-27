"""Tests for the evaluation metrics.

The expected values are hand-computed from the definitions, so the
tests check the maths rather than the code agreeing with itself.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from evaluate import (
    arch_lm,
    diebold_mariano,
    ljung_box_sq,
    mincer_zarnowitz,
    mse,
    qlike,
    qlike_losses,
)


def test_qlike_is_zero_for_a_perfect_forecast():
    # Realised equal to forecast gives ratio 1, and 1 - ln(1) - 1 is 0.
    assert qlike([4.0, 9.0], [4.0, 9.0]) == pytest.approx(0.0)
    # One day: realised 4, forecast 1, ratio 4, loss 4 - ln(4) - 1.
    assert qlike([4.0], [1.0]) == pytest.approx(3.0 - math.log(4.0))
    # The mean of the per-day losses is the headline number.
    losses = qlike_losses([4.0, 9.0], [1.0, 3.0])
    assert losses.tolist() == [
        pytest.approx(3.0 - math.log(4.0)),
        pytest.approx(2.0 - math.log(3.0)),
    ]
    assert qlike([4.0, 9.0], [1.0, 3.0]) == pytest.approx(
        float(np.mean(losses))
    )


def test_mse_is_the_mean_squared_gap():
    assert mse([4.0, 9.0], [4.0, 9.0]) == pytest.approx(0.0)
    assert mse([4.0], [1.0]) == pytest.approx(9.0)
    assert mse([4.0, 9.0], [2.0, 6.0]) == pytest.approx(
        ((4.0 - 2.0) ** 2 + (9.0 - 6.0) ** 2) / 2.0
    )


def test_mincer_zarnowitz_on_a_perfect_forecast():
    # Realised equal to forecast is a line of slope 1 through the
    # origin, so the regression recovers slope 1, intercept 0, R2 1.
    realised = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = mincer_zarnowitz(realised, realised)
    assert result["slope"] == pytest.approx(1.0)
    assert result["intercept"] == pytest.approx(0.0)
    assert result["r_squared"] == pytest.approx(1.0)


def test_mincer_zarnowitz_constant_forecast_has_no_slope():
    # A forecast that never varies has nothing to regress on.
    result = mincer_zarnowitz([1.0, 2.0, 3.0], [2.0, 2.0, 2.0])
    assert result["r_squared"] == 0.0
    assert math.isnan(result["slope"])
    assert math.isnan(result["intercept"])


def test_diebold_mariano_identical_losses_give_no_difference():
    # The same loss series twice has a zero mean difference.
    result = diebold_mariano([1.0, 2.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 2.0, 1.0])
    assert result["statistic"] == pytest.approx(0.0)
    assert result["p_value"] == pytest.approx(1.0)


def test_diebold_mariano_clear_winner_is_significant():
    # Model a beats model b every day, by 1.0 or 1.2 in turn, so the
    # loss difference varies and the standard error stays finite. The
    # statistic must be strongly negative and the p-value tiny.
    loss_a = [1.0, 1.2, 1.0, 1.2, 1.0, 1.2, 1.0, 1.2]
    loss_b = [2.0, 2.4, 2.0, 2.4, 2.0, 2.4, 2.0, 2.4]
    result = diebold_mariano(loss_a, loss_b)
    assert result["statistic"] < -5.0
    assert result["p_value"] < 0.001


def test_diagnostics_return_finite_values_on_noise():
    # White noise standardised residuals have no structure left, so
    # both tests must return a finite statistic and a p-value in
    # [0, 1] rather than crashing.
    rng = np.random.default_rng(11)
    noise = pd.Series(rng.standard_normal(500))
    lb_stat, lb_p = ljung_box_sq(noise)
    arch_stat, arch_p = arch_lm(noise)
    assert np.isfinite(lb_stat) and 0.0 <= lb_p <= 1.0
    assert np.isfinite(arch_stat) and 0.0 <= arch_p <= 1.0
