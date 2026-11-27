# Stage 0A: EWMA vs GARCH(1,1) on daily realized variance

Out-of-sample variance forecasts from a GARCH(1,1) fitted on percent daily log returns, scored against a constant-variance baseline and an EWMA baseline with lambda 0.94. The realised variance each day is the sum of the squared 5-minute log returns inside that day, in percent squared. QLIKE is the headline metric and MSE is reported alongside it. A Diebold-Mariano test on the per-day QLIKE losses asks whether the GARCH edge over the best baseline is bigger than noise.

## BTC

Data: 304445 5-minute bars, 2024-01-02 to 2026-11-23, 1057 days, 845 in the training set and 212 in the test set. Bars per day: median 288, minimum 202.

### Fitted GARCH(1,1)

| Parameter | Value |
|---|---|
| mu | 0.1058 |
| omega | 0.4848 |
| alpha | 0.1057 |
| beta | 0.8221 |
| persistence (alpha + beta) | 0.9279 |
| half-life (days) | 9.3 |
| unconditional daily vol (%) | 2.59 |
| annualised vol (%) | 49.5 |

### Diagnostics on the in-sample fit

Ljung-Box on squared standardised residuals, lag 10: statistic 12.52, p-value 0.2515. ARCH-LM on standardised residuals, lag 10: statistic 11.76, p-value 0.3015.

### Out-of-sample metrics

| Model | QLIKE | MSE | MZ R-squared | MZ slope | MZ intercept |
|---|---|---|---|---|---|
| GARCH(1,1) | 0.2796 | 14.48 | 0.274 | 1.195 | -1.557 |
| Constant | 0.4415 | 23.96 | 0.000 | n/a | n/a |
| EWMA (0.94) | 0.2838 | 15.64 | 0.213 | 1.325 | -0.805 |

Diebold-Mariano GARCH vs EWMA (0.94): statistic -0.16, p-value 0.8707.

On BTC, GARCH beat the best baseline (EWMA (0.94)) by 1.5% on QLIKE. The constant-variance baseline scored 0.441 on QLIKE and EWMA scored 0.284, against 0.280 for GARCH. The Diebold-Mariano p-value of 0.871 is above 0.05, so the gap is within noise and the edge is not statistically significant on this test set. The Mincer-Zarnowitz slope of 1.20 is near 1, so the GARCH forecasts were roughly calibrated, and the R-squared of 0.274 reflects how much of the day-to-day realised variance a variance forecast can explain.

## ETH

Data: 304437 5-minute bars, 2024-01-02 to 2026-11-23, 1057 days, 845 in the training set and 212 in the test set. Bars per day: median 288, minimum 202.

### Fitted GARCH(1,1)

| Parameter | Value |
|---|---|
| mu | 0.0940 |
| omega | 5.2131 |
| alpha | 0.1014 |
| beta | 0.5112 |
| persistence (alpha + beta) | 0.6127 |
| half-life (days) | 1.4 |
| unconditional daily vol (%) | 3.67 |
| annualised vol (%) | 70.1 |

### Diagnostics on the in-sample fit

Ljung-Box on squared standardised residuals, lag 10: statistic 4.12, p-value 0.9418. ARCH-LM on standardised residuals, lag 10: statistic 3.96, p-value 0.9491.

### Out-of-sample metrics

| Model | QLIKE | MSE | MZ R-squared | MZ slope | MZ intercept |
|---|---|---|---|---|---|
| GARCH(1,1) | 0.3777 | 57.63 | 0.155 | 0.924 | -3.151 |
| Constant | 0.4727 | 83.36 | 0.000 | n/a | n/a |
| EWMA (0.94) | 0.3355 | 44.89 | 0.115 | 0.647 | 2.833 |

Diebold-Mariano GARCH vs EWMA (0.94): statistic 0.64, p-value 0.5223.

On ETH, GARCH did not beat the best baseline (EWMA (0.94)). It was 12.6% worse on QLIKE. The constant-variance baseline scored 0.473 on QLIKE and EWMA scored 0.335, against 0.378 for GARCH. The Diebold-Mariano p-value of 0.522 is above 0.05, so the gap is within noise and the edge is not statistically significant on this test set. The Mincer-Zarnowitz slope of 0.92 is near 1, so the GARCH forecasts were roughly calibrated, and the R-squared of 0.155 reflects how much of the day-to-day realised variance a variance forecast can explain.

## BNB

Data: 304514 5-minute bars, 2024-01-02 to 2026-11-23, 1057 days, 845 in the training set and 212 in the test set. Bars per day: median 288, minimum 180.

### Fitted GARCH(1,1)

| Parameter | Value |
|---|---|
| mu | 0.1061 |
| omega | 0.6120 |
| alpha | 0.1256 |
| beta | 0.8004 |
| persistence (alpha + beta) | 0.9260 |
| half-life (days) | 9.0 |
| unconditional daily vol (%) | 2.88 |
| annualised vol (%) | 54.9 |

### Diagnostics on the in-sample fit

Ljung-Box on squared standardised residuals, lag 10: statistic 15.40, p-value 0.1183. ARCH-LM on standardised residuals, lag 10: statistic 14.19, p-value 0.1643.

### Out-of-sample metrics

| Model | QLIKE | MSE | MZ R-squared | MZ slope | MZ intercept |
|---|---|---|---|---|---|
| GARCH(1,1) | 0.2541 | 20.12 | 0.374 | 1.171 | -1.609 |
| Constant | 0.4468 | 38.99 | 0.000 | n/a | n/a |
| EWMA (0.94) | 0.3121 | 24.49 | 0.243 | 1.139 | 0.285 |

Diebold-Mariano GARCH vs EWMA (0.94): statistic -1.20, p-value 0.2314.

On BNB, GARCH beat the best baseline (EWMA (0.94)) by 18.6% on QLIKE. The constant-variance baseline scored 0.447 on QLIKE and EWMA scored 0.312, against 0.254 for GARCH. The Diebold-Mariano p-value of 0.231 is above 0.05, so the gap is within noise and the edge is not statistically significant on this test set. The Mincer-Zarnowitz slope of 1.17 is near 1, so the GARCH forecasts were roughly calibrated, and the R-squared of 0.374 reflects how much of the day-to-day realised variance a variance forecast can explain.

## XRP

Data: 304426 5-minute bars, 2024-01-02 to 2026-11-23, 1057 days, 845 in the training set and 212 in the test set. Bars per day: median 288, minimum 202.

### Fitted GARCH(1,1)

| Parameter | Value |
|---|---|
| mu | 0.0133 |
| omega | 1.1922 |
| alpha | 0.2009 |
| beta | 0.7580 |
| persistence (alpha + beta) | 0.9588 |
| half-life (days) | 16.5 |
| unconditional daily vol (%) | 5.38 |
| annualised vol (%) | 102.8 |

### Diagnostics on the in-sample fit

Ljung-Box on squared standardised residuals, lag 10: statistic 8.59, p-value 0.5710. ARCH-LM on standardised residuals, lag 10: statistic 8.58, p-value 0.5728.

### Out-of-sample metrics

| Model | QLIKE | MSE | MZ R-squared | MZ slope | MZ intercept |
|---|---|---|---|---|---|
| GARCH(1,1) | 0.2431 | 141.06 | 0.560 | 1.310 | -4.919 |
| Constant | 0.5754 | 346.78 | 0.000 | n/a | n/a |
| EWMA (0.94) | 0.3131 | 238.44 | 0.232 | 1.409 | -1.359 |

Diebold-Mariano GARCH vs EWMA (0.94): statistic -1.08, p-value 0.2810.

On XRP, GARCH beat the best baseline (EWMA (0.94)) by 22.4% on QLIKE. The constant-variance baseline scored 0.575 on QLIKE and EWMA scored 0.313, against 0.243 for GARCH. The Diebold-Mariano p-value of 0.281 is above 0.05, so the gap is within noise and the edge is not statistically significant on this test set. The Mincer-Zarnowitz slope of 1.31 is above 1, so the GARCH forecasts under-reacted: they moved less than the realised variance did, and the R-squared of 0.560 reflects how much of the day-to-day realised variance a variance forecast can explain.

## SOL

Data: 304426 5-minute bars, 2024-01-02 to 2026-11-23, 1057 days, 845 in the training set and 212 in the test set. Bars per day: median 288, minimum 202.

### Fitted GARCH(1,1)

| Parameter | Value |
|---|---|
| mu | 0.0454 |
| omega | 3.0756 |
| alpha | 0.1026 |
| beta | 0.7310 |
| persistence (alpha + beta) | 0.8336 |
| half-life (days) | 3.8 |
| unconditional daily vol (%) | 4.30 |
| annualised vol (%) | 82.1 |

### Diagnostics on the in-sample fit

Ljung-Box on squared standardised residuals, lag 10: statistic 18.14, p-value 0.0526. ARCH-LM on standardised residuals, lag 10: statistic 18.14, p-value 0.0527.

### Out-of-sample metrics

| Model | QLIKE | MSE | MZ R-squared | MZ slope | MZ intercept |
|---|---|---|---|---|---|
| GARCH(1,1) | 0.2845 | 75.01 | 0.261 | 1.495 | -9.626 |
| Constant | 0.4641 | 157.44 | 0.000 | n/a | n/a |
| EWMA (0.94) | 0.2684 | 76.68 | 0.154 | 1.126 | 0.914 |

Diebold-Mariano GARCH vs EWMA (0.94): statistic 0.29, p-value 0.7739.

On SOL, GARCH did not beat the best baseline (EWMA (0.94)). It was 6.0% worse on QLIKE. The constant-variance baseline scored 0.464 on QLIKE and EWMA scored 0.268, against 0.284 for GARCH. The Diebold-Mariano p-value of 0.774 is above 0.05, so the gap is within noise and the edge is not statistically significant on this test set. The Mincer-Zarnowitz slope of 1.50 is above 1, so the GARCH forecasts under-reacted: they moved less than the realised variance did, and the R-squared of 0.261 reflects how much of the day-to-day realised variance a variance forecast can explain.

## Skipped assets

The following listed assets had no data file and were skipped: EUR/USD.

EUR/USD is not in the table above: no free 2-year 5-minute source is reachable from this machine. Dukascopy datafeed answers a few requests, then returns HTTP 429, then stops answering. histdata.com does not answer. Kraken EURUSD OHLC answers, but its `since` pagination is erratic: repeated calls return inconsistent, overlapping windows that never reach before the last few days, so no trustworthy 2-year series can be built from it. Yahoo Finance intraday bars only go back about 60 days. The pipeline skips a missing data file with a warning, so EUR/USD slots in the moment a source is found.
