"""Tests for the realized variance builder.

The expected numbers are computed from the definition with plain
math, not by calling the code under test, so a wrong implementation
cannot agree with them by construction.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from realized import daily_frame, five_minute_returns


def _bars(closes, start=datetime(2024, 1, 1, tzinfo=timezone.utc)):
    """Bars with the given closes, one 5-minute bar after the next."""
    times = [start + timedelta(minutes=5 * i) for i in range(len(closes))]
    return pd.DataFrame(
        {
            "open_time": times,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1.0] * len(closes),
        }
    )


def test_five_minute_returns_match_hand_computation():
    # Closes 100, 101, 100.5, 102 give three percent log returns.
    bars = _bars([100.0, 101.0, 100.5, 102.0])
    returns = five_minute_returns(bars)
    expected = [
        100.0 * math.log(101.0 / 100.0),
        100.0 * math.log(100.5 / 101.0),
        100.0 * math.log(102.0 / 100.5),
    ]
    assert returns.tolist() == [pytest.approx(v) for v in expected]


def test_realized_variance_is_the_sum_of_squared_returns():
    # Day 1 has four closes, day 2 has two. Day 2 RV is the sum of
    # the squared 5-minute returns landing on day 2, including the
    # first return of the day, which comes from day 1 last close.
    day1 = _bars([100.0, 101.0, 100.5, 102.0])
    day2 = _bars(
        [103.0, 101.0], start=datetime(2024, 1, 2, tzinfo=timezone.utc)
    )
    frame = daily_frame(pd.concat([day1, day2], ignore_index=True))
    assert len(frame) == 1
    assert frame.index.tolist() == [datetime(2024, 1, 2).date()]
    rv_expected = (
        (100.0 * math.log(103.0 / 102.0)) ** 2
        + (100.0 * math.log(101.0 / 103.0)) ** 2
    )
    assert frame["rv"].iloc[0] == pytest.approx(rv_expected)
    assert frame["n_bars"].iloc[0] == 2
    assert frame["ret"].iloc[0] == pytest.approx(
        100.0 * math.log(101.0 / 102.0)
    )


def test_daily_returns_and_rv_align_by_date():
    # Three days of bars with distinct closes per day. The daily
    # frame must have two rows (the first day has no previous close),
    # returns and RV must share the same dates, and the daily return
    # must be the log change of the daily closes.
    day1 = [100.0] * 4
    day2 = [101.0, 102.0, 101.0, 103.0]
    day3 = [103.0, 104.0, 102.0, 105.0]
    closes = day1 + day2 + day3
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    times = []
    for day in range(3):
        for i in range(4):
            times.append(start + timedelta(days=day, minutes=5 * i))
    bars = pd.DataFrame(
        {
            "open_time": times,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [1.0] * len(closes),
        }
    )
    frame = daily_frame(bars)
    assert len(frame) == 2
    assert frame.index.tolist() == [
        datetime(2024, 1, 2).date(),
        datetime(2024, 1, 3).date(),
    ]
    assert frame["ret"].iloc[0] == pytest.approx(
        100.0 * math.log(103.0 / 100.0)
    )
    assert frame["ret"].iloc[1] == pytest.approx(
        100.0 * math.log(105.0 / 103.0)
    )
    assert frame["n_bars"].tolist() == [4, 4]
    # The day 2 RV is the sum of the squared 5-minute returns that
    # land on day 2, including the jump from day 1 last close.
    day2_rv = (
        (100.0 * math.log(101.0 / 100.0)) ** 2
        + (100.0 * math.log(102.0 / 101.0)) ** 2
        + (100.0 * math.log(101.0 / 102.0)) ** 2
        + (100.0 * math.log(103.0 / 101.0)) ** 2
    )
    assert frame["rv"].iloc[0] == pytest.approx(day2_rv)


def test_daily_frame_counts_missing_bars():
    # A day with only two bars still gets a row, and n_bars says 2.
    bars = _bars([100.0, 101.0])
    bars2 = _bars(
        [102.0, 103.0],
        start=datetime(2024, 1, 2, tzinfo=timezone.utc),
    )
    frame = daily_frame(pd.concat([bars, bars2], ignore_index=True))
    assert len(frame) == 1
    assert frame["n_bars"].iloc[0] == 2
