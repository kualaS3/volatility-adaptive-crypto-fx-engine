"""Tests for Black-Scholes pricing and the vol structures."""

import math

from black_scholes import (
    bs_call,
    bs_delta_call,
    bs_delta_put,
    bs_put,
    straddle_value,
    strangle_value,
    strike_for_call_delta,
    strike_for_put_delta,
)


def _norm_cdf(x: float) -> float:
    """Independent normal CDF from math.erf for hand checks."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def test_put_call_parity():
    """Call minus put equals spot minus strike at a zero rate."""
    for spot, strike, time_years, sigma in [
        (60000.0, 60000.0, 30.0 / 365.0, 0.6),
        (60000.0, 70000.0, 7.0 / 365.0, 0.8),
        (1700.0, 1500.0, 30.0 / 365.0, 0.9),
        (60000.0, 50000.0, 1.0, 0.4),
    ]:
        parity = (bs_call(spot, strike, time_years, sigma)
                  - bs_put(spot, strike, time_years, sigma))
        assert abs(parity - (spot - strike)) < 1e-8


def test_atm_straddle_hand_check():
    """ATM straddle against a hand computation with math.erf.

    At the money with a zero rate, d1 is half sigma root T and d2
    is minus d1, and the call equals the put, so the straddle is
    twice the call. The expected value is built from math.erf here,
    not from the module under test.
    """
    spot = 100.0
    strike = 100.0
    time_years = 30.0 / 365.0
    sigma = 0.5
    d1 = 0.5 * sigma * math.sqrt(time_years)
    d2 = -d1
    expected_call = spot * _norm_cdf(d1) - strike * _norm_cdf(d2)
    expected_put = strike * _norm_cdf(-d2) - spot * _norm_cdf(-d1)
    assert abs(bs_call(spot, strike, time_years, sigma)
               - expected_call) < 1e-10
    assert abs(bs_put(spot, strike, time_years, sigma)
               - expected_put) < 1e-10
    assert abs(straddle_value(spot, strike, time_years, sigma)
               - 2.0 * expected_call) < 1e-10
    assert abs(expected_call - expected_put) < 1e-10


def test_expired_option_is_intrinsic_value():
    """At expiry the call and put are worth their intrinsic value."""
    assert bs_call(105.0, 100.0, 0.0, 0.5) == 5.0
    assert bs_put(95.0, 100.0, 0.0, 0.5) == 5.0
    assert bs_call(95.0, 100.0, 0.0, 0.5) == 0.0
    assert bs_put(105.0, 100.0, 0.0, 0.5) == 0.0


def test_delta_strike_solver():
    """The strike from the delta solver really has that delta."""
    spot = 60000.0
    time_years = 30.0 / 365.0
    sigma = 0.6
    call_strike = strike_for_call_delta(spot, time_years, sigma, 0.25)
    assert call_strike > spot
    assert abs(bs_delta_call(spot, call_strike, time_years, sigma)
               - 0.25) < 1e-9
    put_strike = strike_for_put_delta(spot, time_years, sigma, -0.25)
    assert put_strike < spot
    assert abs(bs_delta_put(spot, put_strike, time_years, sigma)
               + 0.25) < 1e-9


def test_strangle_cheaper_than_straddle():
    """A 25-delta strangle costs less than the ATM straddle."""
    spot = 60000.0
    time_years = 30.0 / 365.0
    sigma = 0.6
    call_strike = strike_for_call_delta(spot, time_years, sigma, 0.25)
    put_strike = strike_for_put_delta(spot, time_years, sigma, -0.25)
    strangle = strangle_value(spot, call_strike, put_strike,
                              time_years, sigma)
    straddle = straddle_value(spot, spot, time_years, sigma)
    assert 0.0 < strangle < straddle


def test_higher_vol_raises_straddle_value():
    """A straddle priced at higher vol is worth more.

    This is the property the repricing edge rests on: when the
    forecast vol exceeds implied vol, the model value exceeds the
    market price proxied at implied vol.
    """
    spot = 60000.0
    time_years = 30.0 / 365.0
    low = straddle_value(spot, spot, time_years, 0.5)
    high = straddle_value(spot, spot, time_years, 0.7)
    assert high > low
