"""Daily returns and realized variance from 5-minute bars.

The 5-minute log returns are computed on the continuous bar series,
in percent, and then grouped by UTC calendar day. The daily realized
variance (RV) for a day is the sum of the squared 5-minute log
returns inside that day, in percent squared. The first bar of a day
contributes the return from the previous day last close, so the
overnight move is part of that day realized variance, which is the
standard construction when the market trades around the clock. The
daily log return comes from the daily closes instead: the last close
of the day against the last close of the previous day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def five_minute_returns(bars: pd.DataFrame) -> pd.Series:
    """Percent log returns between consecutive 5-minute closes.

    Returns are dated by the bar they land on, so a return belongs to
    the UTC day of its closing bar. The first bar has no previous
    close and is dropped.
    """
    closes = bars.set_index("open_time")["close"]
    returns = 100.0 * np.log(closes).diff()
    return returns.dropna()


def daily_frame(bars: pd.DataFrame) -> pd.DataFrame:
    """One row per UTC day: close, daily log return, realized variance.

    Columns are close (last 5-minute close of the day), ret (percent
    log return from the previous day last close), rv (sum of squared
    5-minute percent log returns inside the day, in percent squared),
    and n_bars (how many 5-minute bars the day held). Days are in
    ascending date order. A day with missing bars still gets an RV
    from the bars it has, and n_bars shows the gap.
    """
    bars = bars.sort_values("open_time").reset_index(drop=True)
    closes = bars.set_index("open_time")["close"]
    returns = five_minute_returns(bars)

    daily_close = closes.resample("1D").last()
    daily_ret = 100.0 * np.log(daily_close).diff()
    daily_rv = (returns ** 2).resample("1D").sum()
    daily_n = closes.resample("1D").count()

    frame = pd.DataFrame(
        {
            "close": daily_close,
            "ret": daily_ret,
            "rv": daily_rv,
            "n_bars": daily_n,
        }
    )
    frame.index = frame.index.date
    frame.index.name = "date"
    return frame.dropna(subset=["ret", "rv"])
