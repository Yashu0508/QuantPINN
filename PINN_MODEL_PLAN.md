# PINN option-pricing model plan

## What the model will predict

This project should be framed as **option valuation**, not unrestricted stock-price forecasting. Given a market scenario

```text
spot S, strike K, time to expiry T, volatility sigma,
risk-free rate r, option type CE/PE
```

the PINN will predict a theoretical option value `V(S, K, T, sigma, r, type)`. Automatic differentiation can also produce Greeks such as delta and gamma. The observed NSE option `Close` is the market benchmark used to measure pricing error; it should not be silently used as the target of the pure-PDE training loss. A separate supervised or residual-correction experiment may later predict market deviations, but that must be labelled separately.

## Work completed

`preprocess_datasets.py` preserves the supplied raw files and creates compressed Parquet outputs in `processed/`:

- `underlying_daily.parquet`: normalized NIFTY data, log returns, and backward-looking 20-, 60-, and 252-day annualized volatility;
- `option_eod_ce.parquet` and `option_eod_pe.parquet`: canonical option records with numeric fields, maturity state, moneyness, quality flags, joined spot and volatility;
- `risk_free_monthly.parquet`: normalized RBI observations with source values and source-unit metadata;
- `market_snapshot_ce.parquet` and `market_snapshot_pe.parquet`: compact pricing-scenario records.
- `validate_data.py` and `reports/phase1_validation.json`: global duplicate-key, range, maturity-state, moneyness, and NIFTY/option-underlying validation.
- `src/quant/black_scholes.py` and `src/quant/monte_carlo.py`: tested classical pricing benchmarks.
- `src/pinn/fixed_parameter.py`: controlled CE/PE PINN baseline, with checkpoints under `models/`.
- `src/pinn/parametric.py`: reusable parameterized PINN architecture for synthetic calibration scenarios.
- `src/validation/metrics.py` and `src/hedging/delta.py`: validation metrics and a daily EOD delta-hedging simulator that requires an explicitly validated decimal rate.
- `evaluate_market.py`: chronological Black–Scholes and sampled Monte Carlo evaluation against eligible NSE observations.

The raw option records remain available in the root unchanged. The preprocessing keeps expired and maturity-day rows but classifies them. It uses the previous calendar month's RBI observation as a conservative as-of join and applies the documented percentage-to-decimal rate policy while retaining the source label.

## Required next phases

1. **Data validation gate:** add global duplicate-key checks for `(valuation_date, expiry, strike, option_type)`, cross-check option `underlying_value` against the NIFTY close, and independently verify date, range, and contract-state rules.
2. **RBI calibration gate:** establish whether the RBI values are percentage yields, document the conversion, and decide the official day-count and prior-month/as-of convention. Until this passes, market snapshots are not eligible for final pricing claims.
3. **Classical benchmark:** implement Black–Scholes calls, puts, and Greeks; implement Monte Carlo GBM with deterministic seeds and convergence tests.
4. **Fixed-parameter PINN:** train a fully connected network on the Black–Scholes PDE residual, terminal payoff, and boundary conditions for controlled CE and PE cases. Verify price and Greeks against Black–Scholes before using market observations.
5. **Parameterized PINN:** train one reusable network on normalized `S/K`, normalized maturity, volatility, validated rate, and option type, with normalized output `V/K`. Keep training and validation chronological.
6. **Market evaluation:** compare PINN, Black–Scholes, and Monte Carlo against NSE `Close`, reporting MAE, RMSE, relative error, PDE residual, and slices by option type, moneyness, maturity, date, volatility, and liquidity.
7. **Hedging experiment:** compare PINN and Black–Scholes delta in a daily EOD historical hedge simulation. Do not claim intraday execution performance.
8. **Application layer:** only after the numerical gates pass, expose inference, Greeks, surfaces, comparisons, and hedge results through the planned API/frontend.

## Current blockers and boundaries

- The RBI export's `Indian rupee` unit label is inconsistent metadata. `configs/rate_policy.json` records the evidence-backed interpretation as an annualized percentage yield with a `0.01` conversion; the raw value and source label remain preserved.
- The supplied RBI series ends in June 2026 while NIFTY and options continue through 5 October 2026. With prior-month alignment, only observations with a supported prior month are eligible; later observations remain uncalibrated.
- The first classical market evaluation shows large errors, so volatility-window selection, liquidity filtering, and possibly carry/dividend treatment must be researched before using market errors as a PINN success measure.
- CE/PE parity diagnostics are now implemented; their large residuals make carry/dividend and quote-quality analysis a required calibration step rather than an optional cosmetic metric.
- Option OHLC and liquidity fields are sparse. `Close` is complete and is retained as the primary market observation; missing LTP, settlement, or activity fields are not imputed into it.
- The supplied data is daily EOD data. The first hedging version must therefore be daily-rebalanced.
- The fixed checkpoints now pass the stated analytical accuracy gate after explicit no-arbitrage projection. The parameterized checkpoint is implemented but remains below its synthetic-scenario accuracy gate and must be improved before market deployment.
- The best current market-parameterized candidate was trained on Kaggle GPU run v2, not locally. Its chronological holdout covers 24 September 2025 through 31 July 2026; normalized PINN-vs-Black–Scholes MAE is 0.7529%, RMSE 1.6016%, and maximum error 10.24% after output projection. The later maturity-balanced experiment was rejected because it was worse; the candidate still needs an independent acceptance gate.
- The fixed research acceptance policy now passes on the final chronological quarter (MAE 0.5460%, RMSE 1.2542%, maximum 7.7428%) and records the model as research-accepted only. It still requires a separate production gate before API deployment.
- Daily EOD hedging is now executed on a real CE contract path; the first diagnostic shows PINN delta MAE 0.0685 versus Black–Scholes and higher replication error, so the hedge component is implemented but not a performance claim.
- A research-only `/health` and `/price` API now loads the Kaggle checkpoint without retraining per request; deployment remains gated by production acceptance.
