# Stage -1: implied vol against the GARCH forecast

The question is whether a GARCH vol forecast beats the market implied vol by enough to make buying volatility profitable after costs. The horse race scores the GARCH forecast and the DVOL implied vol as forecasters of subsequently realised vol over 30 days. The repricing values an at-the-money straddle and a 25-delta strangle at forecast vol against the market price. The edge test buys both on days where the forecast exceeds implied vol and holds them to expiry, unhedged.

## Data

Five-minute bars are the Stage 0A files, daily frame 2024-01-02 to 2026-11-23, 1,057 days. DVOL history comes from the Deribit public API, no key, resolution 1D. BTC DVOL covers 2024-01-01 to 2026-11-27, 1062 daily rows. ETH DVOL covers 2024-01-01 to 2026-11-27, 1062 daily rows. The live option chain snapshot is from 2026-11-27. The GARCH forecast applied to the live snapshot is fitted on data through 2026-11-23, a few days before the snapshot, which is a staleness to keep in mind when reading the live comparison.

## Limits, stated plainly

- The 7-day horse race runs on the live snapshot only. A long historical 7-day implied vol series would need stitching across roughly 150 weekly expiries and is not built here. The 7-day check below is one day, not a history.
- The 30-day horizons overlap, so consecutive forecast windows share 29 days of realised variance. The Diebold-Mariano p-values below are optimistic about that serial correlation, since the test lag does not cover a 29-day overlap.
- For the historical 30-day check the market price is proxied by the Black-Scholes price at DVOL implied vol. That is not a real traded price. Real traded prices only appear in the live snapshot, as mark prices.
- The edge test is unhedged. Daily delta hedging is the follow-up, not something this test does. An unhedged straddle is a directional bet as well as a vol bet, so its payoff swings with the underlying move.
- Black-Scholes uses a zero risk-free rate, since Deribit options are European and cash-settled and the rate is small next to crypto vol.

## BTC

### Fitted GARCH(1,1) on the training set

| Parameter | Value |
|---|---|
| mu | 0.1058 |
| omega | 0.4848 |
| alpha | 0.1057 |
| beta | 0.8221 |
| persistence (alpha + beta) | 0.9279 |
| half-life (days) | 9.3 |
| annualised vol (%) | 49.5 |

### Horse race at 30 days

Test days with a full 30-day window: 182. Signal days, where the forecast exceeded implied vol: 180.

| Forecaster | QLIKE | Mean vol (%) |
|---|---|---|
| GARCH forecast | 0.1154 | 46.0 |
| DVOL implied vol | 0.0608 | 38.6 |
| Realised vol | | 38.0 |

Mean gap, forecast minus implied: 7.42 vol points. Diebold-Mariano GARCH against implied vol: statistic 2.84, p-value 0.0045.

### Repricing at 30 days, historical

Market price proxied by the Black-Scholes price at DVOL implied vol, not a real traded price. Edge is model value at forecast vol minus that proxy market price.

| Structure | Mean edge per trade, all days ($) | Mean edge per trade, signal days ($) |
|---|---|---|
| ATM straddle | 1251 | 1267 |
| 25-delta strangle | 1041 | 1054 |

### Edge test at 30 days, held to expiry, unhedged

Costs are 1 percent of premium, since no live spread exists for history, plus a 0.03 percent of underlying taker fee per leg. The underlying at expiry is the daily close from the Stage 0A data.

Directional check: the mean 30-day log return was -1.54 percent, the mean absolute 30-day move was 11.45 percent, against 8.69 percent expected from the realised variance under a normal move. When the actual absolute move exceeds the normal-implied one, an unhedged straddle pays more than its Black-Scholes price even when variance is fairly priced, so this gap separates a vol edge from a fat-move payoff.

| Structure | Mean net payoff, all days ($) | Mean net payoff, signal days ($) | Win rate, signal days |
|---|---|---|---|
| ATM straddle | 1756 | 1798 | 0.49 |
| 25-delta strangle | 1706 | 1752 | 0.42 |

### Live snapshot repricing

Snapshot from 2026-11-27. Forecast vol is fitted on data through 2026-11-23. Market prices are live mark prices. Edge is model value at forecast vol minus market price.

Expiry 2026-12-04 (7 days out), spot 59875, forecast vol 49.3 percent, at-the-money call IV 49.7 percent, at-the-money put IV 49.7 percent. Most recent completed 7-day realised vol 74.2 percent, for context: the matching 7-day realised vol for this expiry has not happened yet.

| Structure | Market price ($) | Model value at forecast vol ($) | Edge ($) | Half spread ($) |
|---|---|---|---|---|
| ATM straddle, strike 60000 | 3209 | 3263 | 54 | 45 |
| 25-delta strangle, call strike 63000 (delta 0.23), put strike 57000 (delta -0.25) | 1150 | 1102 | -48 | 45 |

Expiry 2026-12-25 (28 days out), spot 59973, forecast vol 48.5 percent, at-the-money call IV 48.6 percent, at-the-money put IV 48.6 percent. Most recent completed 7-day realised vol 74.2 percent, for context: the matching 7-day realised vol for this expiry has not happened yet.

| Structure | Market price ($) | Model value at forecast vol ($) | Edge ($) | Half spread ($) |
|---|---|---|---|---|
| ATM straddle, strike 60000 | 6405 | 6419 | 14 | 60 |
| 25-delta strangle, call strike 66000 (delta 0.24), put strike 55000 (delta -0.26) | 2471 | 2384 | -87 | 45 |

## ETH

### Fitted GARCH(1,1) on the training set

| Parameter | Value |
|---|---|
| mu | 0.0940 |
| omega | 5.2131 |
| alpha | 0.1014 |
| beta | 0.5112 |
| persistence (alpha + beta) | 0.6127 |
| half-life (days) | 1.4 |
| annualised vol (%) | 70.1 |

### Horse race at 30 days

Test days with a full 30-day window: 182. Signal days, where the forecast exceeded implied vol: 182.

| Forecaster | QLIKE | Mean vol (%) |
|---|---|---|
| GARCH forecast | 0.1787 | 68.3 |
| DVOL implied vol | 0.0572 | 52.9 |
| Realised vol | | 50.9 |

Mean gap, forecast minus implied: 15.41 vol points. Diebold-Mariano GARCH against implied vol: statistic 5.62, p-value 0.0000.

### Repricing at 30 days, historical

Market price proxied by the Black-Scholes price at DVOL implied vol, not a real traded price. Edge is model value at forecast vol minus that proxy market price.

| Structure | Mean edge per trade, all days ($) | Mean edge per trade, signal days ($) |
|---|---|---|
| ATM straddle | 76 | 76 |
| 25-delta strangle | 64 | 64 |

### Edge test at 30 days, held to expiry, unhedged

Costs are 1 percent of premium, since no live spread exists for history, plus a 0.03 percent of underlying taker fee per leg. The underlying at expiry is the daily close from the Stage 0A data.

Directional check: the mean 30-day log return was -1.49 percent, the mean absolute 30-day move was 16.38 percent, against 11.65 percent expected from the realised variance under a normal move. When the actual absolute move exceeds the normal-implied one, an unhedged straddle pays more than its Black-Scholes price even when variance is fairly priced, so this gap separates a vol edge from a fat-move payoff.

| Structure | Mean net payoff, all days ($) | Mean net payoff, signal days ($) | Win rate, signal days |
|---|---|---|---|
| ATM straddle | 82 | 82 | 0.57 |
| 25-delta strangle | 65 | 65 | 0.51 |

### Live snapshot repricing

Snapshot from 2026-11-27. Forecast vol is fitted on data through 2026-11-23. Market prices are live mark prices. Edge is model value at forecast vol minus market price.

Expiry 2026-12-04 (7 days out), spot 1662, forecast vol 66.8 percent, at-the-money call IV 65.2 percent, at-the-money put IV 65.2 percent. Most recent completed 7-day realised vol 87.5 percent, for context: the matching 7-day realised vol for this expiry has not happened yet.

| Structure | Market price ($) | Model value at forecast vol ($) | Edge ($) | Half spread ($) |
|---|---|---|---|---|
| ATM straddle, strike 1650 | 117 | 123 | 6 | 2 |
| 25-delta strangle, call strike 1750 (delta 0.29), put strike 1550 (delta -0.23) | 46 | 48 | 1 | 1 |

Expiry 2026-12-25 (28 days out), spot 1664, forecast vol 67.0 percent, at-the-money call IV 63.0 percent, at-the-money put IV 63.0 percent. Most recent completed 7-day realised vol 87.5 percent, for context: the matching 7-day realised vol for this expiry has not happened yet.

| Structure | Market price ($) | Model value at forecast vol ($) | Edge ($) | Half spread ($) |
|---|---|---|---|---|
| ATM straddle, strike 1670 | 231 | 247 | 16 | 2 |
| 25-delta strangle, call strike 1900 (delta 0.24), put strike 1500 (delta -0.26) | 90 | 99 | 9 | 2 |

## Verdict

On BTC, the GARCH forecast did not beat implied vol as a forecaster of realised vol, with QLIKE 0.1154 against 0.0608 for implied vol, and the Diebold-Mariano p-value of 0.005 is below 0.05, so the gap is bigger than noise, though the overlapping windows make the p-value optimistic. The forecast sat on average 7.4 vol points above implied vol, so it overstates vol rather than facing a vol risk premium headwind, and implied vol priced realised variance about fairly, at 38.6 percent against 38.0 percent realised. Buying volatility on signal days made money on average, 1798 dollars per straddle and 1752 dollars per strangle after costs. The mean absolute 30-day move was 11.5 percent against 8.7 percent expected from the realised variance under a normal move, so the unhedged payoff is fattened by large directional moves rather than by vol mispricing. The signal fired on 180 of 182 days, so it barely filters at all: the signal-day average is nearly the unconditional average. On ETH, the GARCH forecast did not beat implied vol as a forecaster of realised vol, with QLIKE 0.1787 against 0.0572 for implied vol, and the Diebold-Mariano p-value of 0.000 is below 0.05, so the gap is bigger than noise, though the overlapping windows make the p-value optimistic. The forecast sat on average 15.4 vol points above implied vol, so it overstates vol rather than facing a vol risk premium headwind, and implied vol priced realised variance about fairly, at 52.9 percent against 50.9 percent realised. Buying volatility on signal days made money on average, 82 dollars per straddle and 65 dollars per strangle after costs. The mean absolute 30-day move was 16.4 percent against 11.7 percent expected from the realised variance under a normal move, so the unhedged payoff is fattened by large directional moves rather than by vol mispricing. The signal fired on 182 of 182 days, so it barely filters at all: the signal-day average is nearly the unconditional average. Taken together, there is no vol edge here. Implied vol is the better forecaster and prices realised variance about fairly, the GARCH forecast overstates vol, so the repricing edge is mechanical rather than real, and the positive unhedged payoff comes from 30-day moves running fatter than a normal distribution implies, which a delta-hedged test would strip out. Buying volatility on this signal is a bet on fat moves, not a bet on mispriced vol.
