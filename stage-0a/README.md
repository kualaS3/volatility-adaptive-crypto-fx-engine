# Stage 0A: EWMA against GARCH(1,1) on daily realised variance

## What it is

Stage 0A is the baseline stage of the offline model validation. It fits GARCH(1,1) on daily log returns and scores its one-day-ahead variance forecasts against a constant-variance baseline and an EWMA baseline, on the top five cryptocurrencies by market capitalisation, excluding stablecoins.

## The method

Five-minute bars from 2024-01-02, stored as Parquet, come from the Coinbase Exchange candles API for BTC, ETH, XRP and SOL, and from Binance US for BNB. Each day has two numbers: the daily log return from close to close, and the daily realised variance, the sum of the squared five-minute log returns inside that day. The five-minute bars are the measuring tape, not the model. The models run at daily frequency, and the intraday data only gives a better reading of what actually happened, much less noisy than a single squared daily return. Train is 2024-01-02 to 2026-04-25 (845 days) and test is 2026-04-26 to 2026-11-23 (212 days), split chronologically. GARCH refits every 20 test days on an expanding window, and between refits the forecast still moves daily by stepping the variance recursion. There is no lookahead anywhere. QLIKE is the headline metric because the true variance is never observed, only proxied, and QLIKE still ranks models correctly under a noisy proxy while punishing under-prediction harder than over-prediction. Mincer-Zarnowitz checks calibration and the Diebold-Mariano test asks whether the QLIKE gap between two models is bigger than noise.

## What it found

| Asset | QLIKE GARCH | QLIKE EWMA | QLIKE Constant | DM p-value |
|---|---|---|---|---|
| BTC | 0.2796 | 0.2838 | 0.4415 | 0.871 |
| ETH | 0.3777 | 0.3355 | 0.4727 | 0.522 |
| BNB | 0.2541 | 0.3121 | 0.4468 | 0.231 |
| XRP | 0.2431 | 0.3131 | 0.5754 | 0.281 |
| SOL | 0.2845 | 0.2684 | 0.4641 | 0.774 |

GARCH beats EWMA on QLIKE for BTC (+1.5%), BNB (+18.6%) and XRP (+22.4%), and loses on ETH (-12.6%) and SOL (-6.0%), but no Diebold-Mariano test is significant, so none of the gaps is bigger than noise on 212 test days. The regime split is the same on all five assets: EWMA wins the calm days, GARCH wins the hot days, and mid days are a tie. At a 5-day horizon GARCH beats EWMA on QLIKE on all five assets by 11% to 57%, though the Diebold-Mariano test is still not significant on 42 blocks. The fitted persistence varies a lot by asset, from 0.61 on ETH (half-life 1.4 days) to 0.96 on XRP (half-life 16.5 days), and ETH's low persistence reproduces the daily GARCH project, so it is an ETH property rather than a window artifact. The full per-asset numbers live in research/stage0a_summary.md.

## Limitations

The test window is a single 7-month stretch, much calmer than the training period on every asset, and there is no walk-forward check yet. Five assets on one test set is five chances for a lucky win. Tuning on train already bit us: the tuned EWMA decay for ETH scored much worse on test. The realised variance proxy is much better than a squared daily return but still not perfect, with microstructure noise and days that have 180 bars instead of 288. EUR/USD has no data yet, since no free 2-year five-minute source is reachable, so Stage 0A runs crypto-only. Normal GARCH understates tails, which is what Stage 0B is for.

## How to run it

Run from stage-0a. The data files are committed, so the analysis reproduces with no API call.

```
pip install -e .
python scripts/fetch_research_data.py   # downloads the 5-minute bars
python -m research.stage0a               # fits, forecasts, scores, writes the summary
python -m pytest tests/                 # 22 tests
```
