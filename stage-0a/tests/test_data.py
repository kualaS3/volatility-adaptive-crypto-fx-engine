"""Tests for the data layer: the incomplete-bar rule and Parquet storage."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from data import COLUMNS, drop_incomplete_bar, load_parquet, save_parquet


def _frame(dates):
    return pd.DataFrame(
        {
            "open_time": dates,
            "open": [100.0] * len(dates),
            "high": [101.0] * len(dates),
            "low": [99.0] * len(dates),
            "close": [100.5] * len(dates),
            "volume": [10.0] * len(dates),
        }
    )


def test_drop_incomplete_bar_removes_the_current_bar_only():
    # A frame ending on the still-forming bar loses that one row, and
    # a frame ending on a completed bar is returned unchanged.
    now = datetime.now(timezone.utc)
    boundary = now.replace(second=0, microsecond=0, minute=(now.minute // 5) * 5)
    before = boundary - timedelta(minutes=5)
    frame = _frame([before, boundary])
    dropped = drop_incomplete_bar(frame, now=now)
    assert len(dropped) == 1
    assert dropped["open_time"].tolist() == [before]
    assert len(drop_incomplete_bar(dropped, now=now)) == 1


def test_parquet_round_trip_preserves_values_and_order(tmp_path):
    frame = _frame(
        [
            datetime(2024, 1, 2, tzinfo=timezone.utc),
            datetime(2024, 1, 1, tzinfo=timezone.utc),
        ]
    )
    path = tmp_path / "round_trip.parquet"
    save_parquet(frame, str(path))
    loaded = load_parquet(str(path))
    assert list(loaded.columns) == COLUMNS
    assert loaded["close"].tolist() == [100.5, 100.5]
    assert loaded["open_time"].is_monotonic_increasing
    assert loaded["open_time"].iloc[0] == pd.Timestamp(
        "2024-01-01", tz="UTC"
    )
