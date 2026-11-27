"""Price data: fetch 5-minute bars from public APIs and store them as Parquet.

Two sources are used, both public and keyless. Four of the five crypto
assets (BTC, ETH, XRP, SOL) come from the Coinbase Exchange candles
API, which returns at most 300 candles per call, so a multi-year pull
pages through the range in 25-hour chunks. BNB is the exception:
Coinbase lists BNB-USD only since October 2025, about 13 months of
history, so the 2+ year BNB series comes from Binance US (BNBUSDT
klines, up to 1000 candles per call). EUR/USD comes from the Kraken
public OHLC endpoint, which returns up to 720 candles per call and
pages with a `since` timestamp. Dukascopy was tried first for FX and
is not usable from this machine: its datafeed answers the first few
requests, then returns HTTP 429 Too Many Requests, then stops
answering at all. histdata.com does not answer either. Kraken EUR/USD
is spot FX traded on Kraken, so it is a real FX price series, and the
venue is named in the file and the summary rather than hidden.

Both fetchers return the same table shape with columns open_time
(UTC timestamp), open, high, low, close, volume. The last bar is
still forming at fetch time, so it is dropped before saving: the
committed files end on a completed bar, and re-running the fetch
reproduces the same rows plus the newer completed bars.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests

COINBASE_CANDLES_URL = (
    "https://api.exchange.coinbase.com/products/{product}/candles"
)
BINANCE_US_KLINES_URL = "https://api.binance.us/api/v3/klines"
KRAKEN_OHLC_URL = "https://api.kraken.com/0/public/OHLC"

COLUMNS = ["open_time", "open", "high", "low", "close", "volume"]

COINBASE_CHUNK = timedelta(hours=25)
COINBASE_SLEEP = 0.4
BINANCE_US_SLEEP = 0.2
KRAKEN_SLEEP = 1.0
MAX_RETRIES = 5


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _current_bar_start(now: datetime) -> datetime:
    """Start of the 5-minute bar that is still forming at `now`."""
    return now.replace(second=0, microsecond=0, minute=(now.minute // 5) * 5)


def drop_incomplete_bar(frame: pd.DataFrame,
                        now: datetime | None = None) -> pd.DataFrame:
    """Drop bars from the still-forming 5-minute bar onwards.

    A bar whose open time is the current 5-minute boundary has a close
    that keeps moving until the bar ends, so keeping it would make two
    runs on the same day fit on different data. Completed bars are
    untouched. A frame that already ends on a completed bar comes back
    unchanged.
    """
    if frame.empty:
        return frame
    boundary = _current_bar_start(now or _utcnow())
    keep = frame["open_time"] < boundary
    return frame.loc[keep].reset_index(drop=True)


def _get_json(url: str, params: dict, sleep: float) -> object:
    """One GET with retries on rate limits and server errors."""
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 429:
                raise RuntimeError(f"HTTP 429 rate limit for {url}")
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, RuntimeError) as error:
            last_error = error
            time.sleep(sleep * (2 ** attempt))
    raise RuntimeError(f"GET {url} failed after {MAX_RETRIES} tries: "
                       f"{last_error}")


def fetch_coinbase_5m(start: datetime, end: datetime,
                      product: str = "BTC-USD") -> pd.DataFrame:
    """5-minute bars for one Coinbase product over [start, end).

    Coinbase returns at most 300 candles per call and answers newest
    first, so the range is walked in 25-hour chunks (300 bars of 5
    minutes) and the chunks are stitched in ascending time order.
    Each candle is [time, low, high, open, close, volume].
    """
    rows: list[dict] = []
    cursor = start
    chunk = 0
    while cursor < end:
        chunk_end = min(cursor + COINBASE_CHUNK, end)
        payload = _get_json(
            COINBASE_CANDLES_URL.format(product=product),
            {
                "granularity": 300,
                "start": cursor.isoformat(),
                "end": chunk_end.isoformat(),
            },
            COINBASE_SLEEP,
        )
        if not isinstance(payload, list):
            raise RuntimeError(f"Coinbase refused the request: {payload}")
        for candle in payload:
            timestamp, low, high, open_, close, volume = candle
            rows.append(
                {
                    "open_time": datetime.fromtimestamp(timestamp,
                                                        tz=timezone.utc),
                    "open": float(open_),
                    "high": float(high),
                    "low": float(low),
                    "close": float(close),
                    "volume": float(volume),
                }
            )
        chunk += 1
        if chunk % 100 == 0:
            print(f"  Coinbase: {chunk} chunks, reached {chunk_end.date()}")
        cursor = chunk_end
        time.sleep(COINBASE_SLEEP)
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame = frame.drop_duplicates(subset="open_time")
    return frame.sort_values("open_time").reset_index(drop=True)


def fetch_binance_us_5m(start: datetime, end: datetime,
                        symbol: str = "BNBUSDT") -> pd.DataFrame:
    """5-minute bars for one Binance US symbol over [start, end).

    Binance US returns at most 1000 klines per call, so the range is
    walked forward in 1000-candle pages (about 3.5 days each), with
    each page starting just after the previous page's last candle.
    Each kline is [open_time_ms, open, high, low, close, volume, ...].
    """
    rows: list[dict] = []
    cursor_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    calls = 0
    while cursor_ms < end_ms:
        page_end_ms = min(cursor_ms + 1000 * 5 * 60 * 1000, end_ms)
        payload = _get_json(
            BINANCE_US_KLINES_URL,
            {
                "symbol": symbol,
                "interval": "5m",
                "limit": 1000,
                "startTime": cursor_ms,
                "endTime": page_end_ms,
            },
            BINANCE_US_SLEEP,
        )
        if not isinstance(payload, list):
            raise RuntimeError(f"Binance US refused the request: {payload}")
        if not payload:
            break
        for kline in payload:
            rows.append(
                {
                    "open_time": datetime.fromtimestamp(
                        int(kline[0]) / 1000, tz=timezone.utc
                    ),
                    "open": float(kline[1]),
                    "high": float(kline[2]),
                    "low": float(kline[3]),
                    "close": float(kline[4]),
                    "volume": float(kline[5]),
                }
            )
        calls += 1
        if calls % 50 == 0:
            print(f"  Binance US: {calls} calls, reached "
                  f"{datetime.fromtimestamp(int(payload[-1][0]) / 1000, tz=timezone.utc).date()}")
        cursor_ms = int(payload[-1][0]) + 5 * 60 * 1000
        time.sleep(BINANCE_US_SLEEP)
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame = frame.drop_duplicates(subset="open_time")
    return frame.sort_values("open_time").reset_index(drop=True)


def fetch_kraken_5m(start: datetime, end: datetime,
                    pair: str = "EURUSD") -> pd.DataFrame:
    """5-minute bars for one Kraken pair over [start, end).

    Kraken returns up to 720 candles per call. Each call takes a
    `since` timestamp and answers with a `last` value to use as the
    next call's `since`, so the range is walked forward until the
    returned candles reach `end`. Each candle is [time, open, high,
    low, close, vwap, volume, count].
    """
    rows: list[dict] = []
    since = int(start.timestamp())
    end_ts = int(end.timestamp())
    calls = 0
    while True:
        payload = _get_json(
            KRAKEN_OHLC_URL,
            {"pair": pair, "interval": 5, "since": since},
            KRAKEN_SLEEP,
        )
        if not isinstance(payload, dict) or payload.get("error"):
            raise RuntimeError(f"Kraken refused the request: {payload}")
        result = payload["result"]
        key = next(k for k in result if k != "last")
        candles = result[key]
        if not candles:
            break
        for candle in candles:
            timestamp = int(candle[0])
            if timestamp >= end_ts:
                continue
            rows.append(
                {
                    "open_time": datetime.fromtimestamp(timestamp,
                                                        tz=timezone.utc),
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "volume": float(candle[6]),
                }
            )
        calls += 1
        if calls % 50 == 0:
            print(f"  Kraken: {calls} calls, reached "
                  f"{datetime.fromtimestamp(candles[-1][0], tz=timezone.utc).date()}")
        next_since = int(result["last"])
        if next_since <= since:
            break
        since = next_since
        if candles[-1][0] >= end_ts:
            break
        time.sleep(KRAKEN_SLEEP)
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame = frame.drop_duplicates(subset="open_time")
    return frame.sort_values("open_time").reset_index(drop=True)


def save_parquet(frame: pd.DataFrame, path: str) -> None:
    """Write a bars frame to Parquet with a UTC timestamp column."""
    frame = frame.copy()
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
    frame.to_parquet(path, index=False)


def load_parquet(path: str) -> pd.DataFrame:
    """Read a bars Parquet file back, with open_time as UTC datetimes."""
    frame = pd.read_parquet(path)
    frame["open_time"] = pd.to_datetime(frame["open_time"], utc=True)
    return frame.sort_values("open_time").reset_index(drop=True)
