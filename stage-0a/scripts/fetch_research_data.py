"""Fetch the 5-minute research data and write Parquet files.

The top five cryptocurrencies by market capitalisation, excluding
stablecoins, are fetched as follows: BTC-USD, ETH-USD, XRP-USD and
SOL-USD from the Coinbase Exchange candles API, and BNB from Binance
US (BNBUSDT klines), because Coinbase lists BNB-USD only since
October 2025, about 13 months of history, which is short of the 2+
year window. EUR/USD is attempted from the Kraken public OHLC
endpoint (EURUSD spot). All calls are keyless. The window starts
2024-01-01 and runs to now, with the still-forming current bar
dropped before saving, so the committed files end on a completed bar
and re-running this script refreshes them in place. Expect each
crypto fetch to take about 20 minutes: Coinbase pages in 300-candle
chunks. A file is only saved when it covers at least a year, so a
short pull can never silently pass as the dataset.

Run from the repo root: python scripts/fetch_research_data.py
fetches everything. Pass asset names to fetch only those, for
example: python scripts/fetch_research_data.py ETH SOL
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from data import (
    drop_incomplete_bar,
    fetch_binance_us_5m,
    fetch_coinbase_5m,
    fetch_kraken_5m,
    save_parquet,
)

START = datetime(2024, 1, 1, tzinfo=timezone.utc)
MIN_SPAN_DAYS = 365

# The top five cryptocurrencies by market capitalisation, excluding
# stablecoins, in ranking order (CoinGecko, 2026-11-24). Four come
# from Coinbase. BNB comes from Binance US (see fetch_bnb).
COINBASE_ASSETS = [
    ("BTC", "BTC-USD", "btc_usd_5m.parquet"),
    ("ETH", "ETH-USD", "eth_usd_5m.parquet"),
    ("XRP", "XRP-USD", "xrp_usd_5m.parquet"),
    ("SOL", "SOL-USD", "sol_usd_5m.parquet"),
]
BNB_FILENAME = "bnb_usd_5m.parquet"


def _save_if_long_enough(frame, path: Path, label: str) -> bool:
    """Save the frame only when it covers at least a year.

    Returns True when the file was written. A short pull means the
    source could not walk back far enough, and saving it would
    silently misrepresent the coverage.
    """
    span_days = (
        (frame["open_time"].iloc[-1] - frame["open_time"].iloc[0]).days
        if len(frame)
        else 0
    )
    if span_days < MIN_SPAN_DAYS:
        print(f"WARNING: {label} returned only {len(frame)} rows over "
              f"{span_days} days, not the requested 2+ years. The file "
              f"is NOT saved.")
        return False
    save_parquet(frame, str(path))
    print(f"{path.name}: {len(frame)} rows, "
          f"{frame['open_time'].iloc[0]} to {frame['open_time'].iloc[-1]}")
    return True


def fetch_crypto(name: str, product: str, filename: str,
                 data_dir: Path, end: datetime) -> bool:
    """Fetch one crypto asset from Coinbase and save its Parquet file."""
    print(f"Fetching {product} 5-minute bars from Coinbase...")
    frame = drop_incomplete_bar(
        fetch_coinbase_5m(START, end, product=product)
    )
    return _save_if_long_enough(frame, data_dir / filename, name)


def fetch_bnb(data_dir: Path, end: datetime) -> bool:
    """Fetch BNB from Binance US, falling back to Coinbase if short.

    Coinbase lists BNB-USD only since October 2025, about 13 months
    of history, which is short of the 2+ year window. Binance US
    lists BNBUSDT back to 2019, so it is the primary source. The
    Coinbase file is kept as a fallback only if Binance US also comes
    back under a year.
    """
    print("Fetching BNBUSDT 5-minute bars from Binance US...")
    frame = drop_incomplete_bar(
        fetch_binance_us_5m(START, end, symbol="BNBUSDT")
    )
    if _save_if_long_enough(
        frame, data_dir / BNB_FILENAME, "BNB (Binance US)"
    ):
        return True
    print("Binance US BNB series too short, falling back to Coinbase "
          "BNB-USD...")
    return fetch_crypto("BNB", "BNB-USD", BNB_FILENAME, data_dir, end)


def fetch_fx(data_dir: Path, end: datetime) -> bool:
    """Attempt EUR/USD from Kraken and save it only if it is long enough."""
    print("Fetching EUR/USD 5-minute bars from Kraken...")
    eurusd = drop_incomplete_bar(
        fetch_kraken_5m(START, end, pair="EURUSD")
    )
    return _save_if_long_enough(
        eurusd, data_dir / "eurusd_5m.parquet", "EUR/USD"
    )


def main(argv: list[str]) -> None:
    data_dir = ROOT / "research" / "data"
    data_dir.mkdir(exist_ok=True)
    end = datetime.now(timezone.utc)

    wanted = {name.upper() for name in argv} or None
    if wanted is not None:
        unknown = wanted - {name for name, _, _ in COINBASE_ASSETS} - {"BNB", "FX"}
        if unknown:
            raise SystemExit(f"unknown assets: {sorted(unknown)}")
    crypto = COINBASE_ASSETS
    if wanted is not None:
        crypto = [asset for asset in COINBASE_ASSETS if asset[0] in wanted]

    for name, product, filename in crypto:
        fetch_crypto(name, product, filename, data_dir, end)

    if wanted is None or "BNB" in wanted:
        fetch_bnb(data_dir, end)

    if wanted is None or "FX" in wanted:
        fetch_fx(data_dir, end)


if __name__ == "__main__":
    main(sys.argv[1:])
