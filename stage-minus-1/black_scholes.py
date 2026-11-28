"""Black-Scholes pricing for Deribit BTC and ETH options.

Deribit BTC and ETH options are European and cash-settled, so the
plain Black-Scholes formula applies. The risk-free rate is taken as
zero and that assumption is stated everywhere the prices are used:
with a zero rate the forward price equals the spot price, so the
at-the-money strike is the spot price and put-call parity is
call minus put equals spot minus strike. All prices are in the same
units as the spot price, so a spot in dollars gives a price in
dollars. Volatilities are decimals, not percent, and time is in
years as days to expiry divided by 365 for crypto.
"""

from __future__ import annotations

import math
from statistics import NormalDist

_DIST = NormalDist()


def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return _DIST.cdf(x)


def _norm_ppf(p: float) -> float:
    """Standard normal quantile function."""
    return _DIST.inv_cdf(p)


def bs_call(spot: float, strike: float, time_years: float,
            sigma: float) -> float:
    """European call price with a zero risk-free rate."""
    if time_years <= 0.0 or sigma <= 0.0:
        return max(spot - strike, 0.0)
    d1 = (math.log(spot / strike) + 0.5 * sigma ** 2 * time_years) / (
        sigma * math.sqrt(time_years))
    d2 = d1 - sigma * math.sqrt(time_years)
    return spot * _norm_cdf(d1) - strike * _norm_cdf(d2)


def bs_put(spot: float, strike: float, time_years: float,
           sigma: float) -> float:
    """European put price with a zero risk-free rate."""
    if time_years <= 0.0 or sigma <= 0.0:
        return max(strike - spot, 0.0)
    d1 = (math.log(spot / strike) + 0.5 * sigma ** 2 * time_years) / (
        sigma * math.sqrt(time_years))
    d2 = d1 - sigma * math.sqrt(time_years)
    return strike * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def bs_delta_call(spot: float, strike: float, time_years: float,
                  sigma: float) -> float:
    """Call delta with a zero risk-free rate, equal to N(d1)."""
    if time_years <= 0.0 or sigma <= 0.0:
        return 1.0 if spot > strike else 0.0
    d1 = (math.log(spot / strike) + 0.5 * sigma ** 2 * time_years) / (
        sigma * math.sqrt(time_years))
    return _norm_cdf(d1)


def bs_delta_put(spot: float, strike: float, time_years: float,
                 sigma: float) -> float:
    """Put delta with a zero risk-free rate, equal to N(d1) minus 1."""
    return bs_delta_call(spot, strike, time_years, sigma) - 1.0


def strike_for_call_delta(spot: float, time_years: float, sigma: float,
                          target_delta: float) -> float:
    """Strike whose call delta equals target_delta.

    With a zero rate, delta is N(d1), so d1 is the quantile of the
    target delta, and the strike follows from the d1 formula in
    closed form. No solver is needed.
    """
    d1 = _norm_ppf(target_delta)
    return spot * math.exp(-d1 * sigma * math.sqrt(time_years)
                           + 0.5 * sigma ** 2 * time_years)


def strike_for_put_delta(spot: float, time_years: float, sigma: float,
                         target_delta: float) -> float:
    """Strike whose put delta equals target_delta, a negative number.

    Put delta is N(d1) minus 1, so N(d1) is 1 plus the target, and
    the strike follows from the same closed form as the call.
    """
    d1 = _norm_ppf(1.0 + target_delta)
    return spot * math.exp(-d1 * sigma * math.sqrt(time_years)
                           + 0.5 * sigma ** 2 * time_years)


def straddle_value(spot: float, strike: float, time_years: float,
                   sigma: float) -> float:
    """Value of a long at-the-money straddle, call plus put."""
    return (bs_call(spot, strike, time_years, sigma)
            + bs_put(spot, strike, time_years, sigma))


def strangle_value(spot: float, strike_call: float, strike_put: float,
                   time_years: float, sigma: float) -> float:
    """Value of a long strangle, out-of-the-money call plus put."""
    return (bs_call(spot, strike_call, time_years, sigma)
            + bs_put(spot, strike_put, time_years, sigma))


def call_payoff(spot_expiry: float, strike: float) -> float:
    """Payoff of a long call held to expiry."""
    return max(spot_expiry - strike, 0.0)


def put_payoff(spot_expiry: float, strike: float) -> float:
    """Payoff of a long put held to expiry."""
    return max(strike - spot_expiry, 0.0)
