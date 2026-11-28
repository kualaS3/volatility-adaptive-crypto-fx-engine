# Stage -1: Implied vol against the GARCH forecast

## The question

Option prices already contain the market forecast of volatility as implied volatility, and a GARCH model gives a different forecast of the same thing. If the GARCH forecast is higher than the market forecast by enough to cover costs, then buying volatility through a long straddle or a long strangle should have a positive edge. If it is not higher, then there is no edge. Stage -1 tests that question before any execution engine is built, because if the edge is not there on paper, the rest of the project is a faster way to lose money.

## The method

Implied vol comes from Deribit, the main crypto options exchange, with no API key. The DVOL index history gives a daily 30-day implied vol for BTC and ETH, and the live option chain snapshot gives mark IV, mark prices, bid and ask, and open interest for the nearest weekly and monthly expiries. The GARCH forecast is extended to the same horizons with the analytic multi-step formula, refitted every 20 days on an expanding window with no lookahead, the same discipline Stage 0A uses for one day. The check runs daily at 7 days and 30 days to match Deribit expiries. The five-minute bars measure realised vol over those horizons, they do not forecast every five minutes. The BTC data holds 1,051 complete 7-day windows and 1,028 complete 30-day windows. One-day realised vol swings from 8 percent to 217 percent annualised, while 30-day runs from about 25 percent to 75 percent, so the longer window is a much cleaner target.

The comparison is a horse race. GARCH forecast vol and implied vol are scored as forecasters of subsequently realised vol with QLIKE and a Diebold-Mariano test, and the average gap between forecast and implied vol is the vol risk premium headwind. A long at-the-money straddle and a 25-delta strangle are then repriced with Black-Scholes at forecast vol and compared with the market price, which for history is proxied by the Black-Scholes price at DVOL implied vol and for the live snapshot is the real mark price. Finally the edge test buys both structures on days where the forecast exceeds implied vol and holds them to expiry, unhedged, after costs.

## What it found

There is no vol edge here. On the 30-day horse race over 182 test days, implied vol beat the GARCH forecast decisively on QLIKE, 0.061 against 0.115 on BTC and 0.057 against 0.179 on ETH, and implied vol priced realised variance about fairly, at 38.6 percent against 38.0 percent realised on BTC and 52.9 percent against 50.9 percent on ETH. The GARCH forecast sat 7.4 vol points above implied vol on BTC and 15.4 on ETH, so it overstates vol. Buying volatility on signal days still made money on average after costs, 1,798 dollars per BTC straddle and 82 dollars per ETH straddle, but the signal fired on nearly every day, 180 of 182 on BTC and 182 of 182 on ETH, so it barely filters, and the payoff came from 30-day moves running fatter than a normal distribution implies rather than from mispriced vol. The live snapshot on 27 November 2026 tells the same story. Forecast vol sits within a few vol points of at-the-money IV on both expiries, and the straddle edges are tiny, 54 dollars on BTC and 6 dollars on ETH, barely above the half spread of 45 and 2 dollars. A delta-hedged test is the follow-up that would strip the direction out.

## Limitations

Volatility is not direction, so a vol forecast says how big the moves will be, not which way, and an unhedged straddle is a directional bet as well as a vol bet. The market already prices vol, and here implied vol prices realised variance about fairly, so the headwind is not the binding constraint, the forecast overstating vol is. Costs eat small edges, and the historical market price is a Black-Scholes proxy at DVOL implied vol rather than a traded price. The edge test is unhedged, so daily delta hedging is the follow-up, not something this test does. The test window is one quiet 7-month stretch, and the 30-day Diebold-Mariano p-values are optimistic because the windows overlap and share 29 days of realised variance. The 7-day check runs on the live snapshot only, since a long historical 7-day implied vol series would need stitching across roughly 150 weekly expiries and is not built here. The forecast applied to the live snapshot is fitted on data through 23 November, a few days before the 27 November snapshot.

## How to run it

Run from stage-minus-1. The pipeline reads the saved Deribit files under data, so repeated runs on the same saved data produce an identical summary.

```
python -m stage_minus_1            # runs on the saved Deribit data
python -m stage_minus_1 --fetch    # pulls fresh Deribit data first
python -m pytest tests/            # 25 tests
```
