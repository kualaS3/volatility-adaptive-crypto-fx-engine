"""Tests for the Deribit fetchers, using canned API payloads."""

from datetime import date

import deribit
from deribit import (
    chain_row_from_ticker,
    dvol_frame_from_rows,
    fetch_dvol_history,
    last_friday_of_month,
    nearest_monthly_expiry,
    nearest_weekly_expiry,
    parse_expiry,
    parse_option_type,
    parse_strike,
)


def test_parse_instrument_name():
    """Expiry, strike and type come out of an instrument name."""
    name = "BTC-25DEC26-90000-C"
    assert parse_expiry(name) == date(2026, 12, 25)
    assert parse_strike(name) == 90000.0
    assert parse_option_type(name) == "C"
    assert parse_option_type("ETH-4DEC26-2500-P") == "P"


def test_last_friday_of_month():
    """December 2026 ends on a Friday, November ends on the 27th."""
    assert last_friday_of_month(2026, 12) == date(2026, 12, 25)
    assert last_friday_of_month(2026, 11) == date(2026, 11, 27)
    assert last_friday_of_month(2027, 1) == date(2027, 1, 29)


def test_nearest_weekly_expiry():
    """The expiry closest to 7 days out from 27 Nov 2026 is 4 Dec."""
    listed = [
        date(2026, 11, 28), date(2026, 11, 29), date(2026, 11, 30),
        date(2026, 12, 1), date(2026, 12, 4), date(2026, 12, 11),
        date(2026, 12, 18), date(2026, 12, 25),
    ]
    assert nearest_weekly_expiry(date(2026, 11, 27), listed) == (
        date(2026, 12, 4))


def test_nearest_monthly_expiry():
    """On 27 Nov 2026 the November monthly has expired, so use Dec."""
    listed = [date(2026, 12, 25), date(2027, 1, 29)]
    assert nearest_monthly_expiry(date(2026, 11, 27), listed) == (
        date(2026, 12, 25))
    listed = [date(2026, 11, 27), date(2026, 12, 25)]
    assert nearest_monthly_expiry(date(2026, 11, 20), listed) == (
        date(2026, 11, 27))


def test_dvol_frame_from_rows():
    """Raw DVOL rows become a dated frame of daily closes."""
    rows = [
        [1709424000000, 65.82, 71.84, 65.52, 71.72],
        [1709510400000, 71.72, 72.0, 70.0, 71.0],
        [1709424000000, 65.82, 71.84, 65.52, 71.72],
    ]
    frame = dvol_frame_from_rows(rows)
    assert list(frame.columns) == ["date", "dvol_close"]
    assert len(frame) == 2
    assert frame["date"].iloc[0] == date(2024, 3, 3)
    assert frame["dvol_close"].iloc[0] == 71.72
    assert frame["dvol_close"].iloc[1] == 71.0


def test_chain_row_from_ticker():
    """A ticker result becomes one snapshot row with parsed fields."""
    ticker = {
        "instrument_name": "BTC-25DEC26-90000-C",
        "mark_iv": 61.73,
        "mark_price": 0.0006,
        "best_bid_price": 0.0005,
        "best_ask_price": 0.0006,
        "open_interest": 5731.2,
        "underlying_price": 59917.47,
    }
    row = chain_row_from_ticker(ticker)
    assert row["strike"] == 90000.0
    assert row["expiry"] == date(2026, 12, 25)
    assert row["option_type"] == "C"
    assert row["mark_iv"] == 61.73
    assert row["underlying_price"] == 59917.47


def test_fetch_dvol_history_pages_backwards(monkeypatch):
    """Two canned pages stitch into one ascending history.

    The first call returns the newest 1000 rows with a continuation
    timestamp, the second call ends at that timestamp and returns
    no continuation, and the fetch stops there.
    """
    first_page = {
        "result": {
            "data": [[1709424000000, 65.82, 71.84, 65.52, 71.72]],
            "continuation": 1709337600000,
        }
    }
    second_page = {
        "result": {
            "data": [[1709251200000, 64.0, 66.0, 63.0, 65.0]],
            "continuation": None,
        }
    }
    calls = []

    def fake_get(method, params):
        calls.append(params["end_timestamp"])
        if len(calls) == 1:
            return first_page
        return second_page

    monkeypatch.setattr(deribit, "_get", fake_get)
    frame = fetch_dvol_history("BTC")
    assert len(calls) == 2
    assert calls[1] == 1709337600000
    assert len(frame) == 2
    assert frame["date"].iloc[0] == date(2024, 3, 1)
    assert frame["date"].iloc[1] == date(2024, 3, 3)
