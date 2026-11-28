"""Keyless Deribit fetchers for Stage -1.

Two things are pulled from the Deribit public API, with no key:

1. The DVOL volatility index history for BTC and ETH, from
   get_volatility_index_data. The call takes resolution "1D" and a
   start and end timestamp in milliseconds, and returns at most 1000
   rows per call as [timestamp_ms, open, high, low, close] in
   percent, with a continuation timestamp to page backwards for
   earlier data. The close is the daily DVOL value used as the
   market implied vol.
2. A live option chain snapshot: get_instruments lists the live
   options, and the ticker endpoint returns mark_iv, mark_price,
   best bid and ask, open interest and the underlying price per
   instrument. The snapshot covers the nearest weekly expiry and
   the nearest monthly expiry for BTC and ETH.

The ticker endpoint is public/ticker. The name public/get_ticker
does not exist and answers Method not found.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://www.deribit.com/api/v2/public"
MAX_RETRIES = 5
RETRY_SLEEP = 0.5
DVOL_START = datetime(2024, 1, 1, tzinfo=timezone.utc)
CURRENCIES = ("BTC", "ETH")
TICKER_WORKERS = 8


def _get(method: str, params: dict) -> dict:
    """One GET against the Deribit public API, with retries.

    Retries cover rate limits and server errors. A 400 from bad
    parameters is not retried: it means the request itself is wrong.
    """
    last_error: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(f"{BASE_URL}/{method}", params=params,
                                    timeout=30)
            if response.status_code == 400:
                raise ValueError(f"Deribit refused the request: "
                                 f"{response.text[:200]}")
            if response.status_code == 429:
                raise RuntimeError(f"HTTP 429 rate limit for {method}")
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, RuntimeError) as error:
            last_error = error
            time.sleep(RETRY_SLEEP * (2 ** attempt))
    raise RuntimeError(f"GET {method} failed after {MAX_RETRIES} tries: "
                       f"{last_error}")


def dvol_frame_from_rows(rows: list) -> pd.DataFrame:
    """Turn raw DVOL rows into a dated frame of daily closes.

    Each row is [timestamp_ms, open, high, low, close] in percent.
    Duplicate dates are dropped and the frame is sorted ascending.
    Only the date and the close are kept: the close is the daily
    DVOL value used as the market implied vol.
    """
    frame = pd.DataFrame(
        rows, columns=["timestamp_ms", "open", "high", "low", "close"]
    )
    frame["date"] = pd.to_datetime(frame["timestamp_ms"], unit="ms",
                                   utc=True).dt.date
    frame = frame.rename(columns={"close": "dvol_close"})
    frame = frame.drop_duplicates(subset="date")
    frame = frame.sort_values("date").reset_index(drop=True)
    return frame[["date", "dvol_close"]].astype(
        {"dvol_close": float}
    )


def fetch_dvol_history(currency: str,
                       start: datetime = DVOL_START,
                       end: datetime | None = None) -> pd.DataFrame:
    """Full daily DVOL history for one currency, oldest first.

    Pages backwards with the continuation timestamp until the API
    returns no continuation. The first call returns the newest 1000
    rows, so a full pull needs a second call for the earlier rows.
    """
    end = end or datetime.now(timezone.utc)
    end_ms = int(end.timestamp() * 1000)
    start_ms = int(start.timestamp() * 1000)
    rows: list = []
    while True:
        payload = _get("get_volatility_index_data", {
            "currency": currency,
            "resolution": "1D",
            "start_timestamp": start_ms,
            "end_timestamp": end_ms,
        })
        result = payload["result"]
        rows.extend(result["data"])
        continuation = result.get("continuation")
        if not continuation or int(continuation) >= end_ms:
            break
        end_ms = int(continuation)
    return dvol_frame_from_rows(rows)


def parse_expiry(instrument_name: str) -> date:
    """Expiry date from an instrument name like BTC-25DEC26-90000-C."""
    return datetime.strptime(instrument_name.split("-")[1],
                             "%d%b%y").date()


def parse_strike(instrument_name: str) -> float:
    """Strike from an instrument name like BTC-25DEC26-90000-C."""
    return float(instrument_name.split("-")[2])


def parse_option_type(instrument_name: str) -> str:
    """Option type, C or P, from an instrument name."""
    return instrument_name.split("-")[3]


def last_friday_of_month(year: int, month: int) -> date:
    """The last Friday of a calendar month."""
    if month == 12:
        first_next = date(year + 1, 1, 1)
    else:
        first_next = date(year, month + 1, 1)
    last_day = first_next - timedelta(days=1)
    while last_day.weekday() != 4:
        last_day -= timedelta(days=1)
    return last_day


def nearest_monthly_expiry(as_of: date, listed: list) -> date:
    """The monthly expiry for the month containing as_of.

    Monthly options expire on the last Friday of the month. When
    that Friday is today or already past, the next month last Friday
    is used instead. If the computed Friday is not among the listed
    expiries, the listed expiry closest to it is used.
    """
    candidate = last_friday_of_month(as_of.year, as_of.month)
    if candidate <= as_of:
        if as_of.month == 12:
            candidate = last_friday_of_month(as_of.year + 1, 1)
        else:
            candidate = last_friday_of_month(as_of.year, as_of.month + 1)
    if candidate in listed:
        return candidate
    return min(listed, key=lambda expiry: abs((expiry - candidate).days))


def nearest_weekly_expiry(as_of: date, listed: list) -> date:
    """The listed expiry closest to 7 days out from as_of.

    The weekly check matches Deribit weekly expiries, so the expiry
    nearest to a week away is the one the 7-day comparison uses. Its
    actual days to expiry are reported alongside it, since the
    nearest listed expiry is rarely exactly 7 days out.
    """
    future = [expiry for expiry in listed if expiry > as_of]
    return min(future,
               key=lambda expiry: abs((expiry - as_of).days - 7))


def fetch_instruments(currency: str, expired: bool = False) -> list:
    """Live or expired option instruments for one currency."""
    payload = _get("get_instruments", {
        "currency": currency,
        "kind": "option",
        "expired": "true" if expired else "false",
    })
    return payload["result"]


def fetch_ticker(instrument_name: str) -> dict:
    """One ticker: mark IV and price, bid and ask, open interest."""
    payload = _get("ticker", {"instrument_name": instrument_name})
    return payload["result"]


def chain_row_from_ticker(ticker: dict) -> dict:
    """One snapshot row from a ticker result dict."""
    name = ticker["instrument_name"]
    return {
        "instrument_name": name,
        "strike": parse_strike(name),
        "expiry": parse_expiry(name),
        "option_type": parse_option_type(name),
        "mark_iv": ticker.get("mark_iv"),
        "mark_price": ticker.get("mark_price"),
        "best_bid_price": ticker.get("best_bid_price"),
        "best_ask_price": ticker.get("best_ask_price"),
        "open_interest": ticker.get("open_interest"),
        "underlying_price": ticker.get("underlying_price"),
    }


def fetch_option_chain(currency: str, expiries: list,
                       max_workers: int = TICKER_WORKERS) -> pd.DataFrame:
    """Tickers for every live instrument at the given expiries.

    Tickers are fetched in parallel, since a monthly expiry alone
    holds over a hundred instruments. Instruments are taken from the
    live list, so no strike is guessed: a strike that is not listed
    for an expiry simply is not in the snapshot.
    """
    instruments = fetch_instruments(currency)
    wanted = set(expiries)
    names = [
        item["instrument_name"] for item in instruments
        if parse_expiry(item["instrument_name"]) in wanted
    ]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        tickers = list(pool.map(fetch_ticker, names))
    rows = [chain_row_from_ticker(ticker) for ticker in tickers]
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["expiry", "strike", "option_type"])
    return frame.reset_index(drop=True)


def save_dvol(frame: pd.DataFrame, path: str | Path) -> None:
    """Write a DVOL frame to Parquet."""
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame.to_parquet(path, index=False)


def load_dvol(path: str | Path) -> pd.DataFrame:
    """Read a DVOL Parquet file back, with date as a date object."""
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    return frame.sort_values("date").reset_index(drop=True)


def fetch_all_data(data_dir: str | Path,
                   as_of: date | None = None) -> dict:
    """Fetch DVOL histories and the live chain snapshot to data_dir.

    DVOL histories go to dvol_btc.parquet and dvol_eth.parquet. The
    chain snapshot goes to option_chain_snapshot_YYYY-MM-DD.csv, with
    the fetch date in the filename. Returns the paths written and
    the row counts, so the caller can report exactly what was saved.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    as_of = as_of or datetime.now(timezone.utc).date()
    written = {"as_of": str(as_of), "dvol": {}, "snapshot": None,
               "snapshot_rows": 0, "weekly_expiry": {}, "monthly_expiry": {}}
    frames = []
    for currency in CURRENCIES:
        frame = fetch_dvol_history(currency)
        path = data_dir / f"dvol_{currency.lower()}.parquet"
        save_dvol(frame, path)
        written["dvol"][currency] = {
            "path": str(path),
            "rows": len(frame),
            "first_date": str(frame["date"].iloc[0]),
            "last_date": str(frame["date"].iloc[-1]),
        }
        instruments = fetch_instruments(currency)
        listed = sorted({parse_expiry(item["instrument_name"])
                         for item in instruments})
        weekly = nearest_weekly_expiry(as_of, listed)
        monthly = nearest_monthly_expiry(as_of, listed)
        written["weekly_expiry"][currency] = str(weekly)
        written["monthly_expiry"][currency] = str(monthly)
        chain = fetch_option_chain(currency, [weekly, monthly])
        chain["currency"] = currency
        frames.append(chain)
    snapshot = pd.concat(frames, ignore_index=True)
    snapshot_path = (data_dir
                     / f"option_chain_snapshot_{as_of.isoformat()}.csv")
    snapshot.to_csv(snapshot_path, index=False)
    written["snapshot"] = str(snapshot_path)
    written["snapshot_rows"] = len(snapshot)
    return written
