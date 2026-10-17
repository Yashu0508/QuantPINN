# Phase 1 preprocessing report

The raw files in the project root were preserved unchanged. Canonical Parquet outputs were generated under `processed/`.

## Outputs

- `underlying_daily.parquet`: normalized NIFTY history, log returns, and 20/60/252-day backward-looking annualized volatility.
- `option_eod_ce.parquet` and `option_eod_pe.parquet`: normalized option records with maturity state, moneyness, quality flags, and joined spot/volatility fields.
- `risk_free_monthly.parquet`: normalized RBI records with original source values and units.
- `market_snapshot_ce.parquet` and `market_snapshot_pe.parquet`: compact pricing-scenario tables.

## Safety decisions

- `Close` remains the market benchmark; missing LTP or settlement values are not substituted into it.
- The RBI file is parsed as pipe-delimited; its source value and `Indian rupee` label are preserved, while the approved policy converts percentage yields to decimal rates.
- RBI alignment uses the prior calendar month to avoid current-month look-ahead. Observations without a supported prior month remain uncalibrated.
- `eligible_for_pure_pde_pricing` requires active maturity, spot, realized volatility, and a supported prior-month rate.
- Expired and maturity-day contracts remain in canonical data and are classified rather than silently deleted.

## Next plan

1. Verify RBI yield units and define the official rate conversion.
2. Add duplicate-key and cross-source consistency tests.
3. Build the analytical Black–Scholes and Monte Carlo benchmarks.
4. Train and verify a fixed-parameter PINN against Black–Scholes before using market data for evaluation.
5. Build a parameterized PINN over normalized spot/strike, maturity, volatility, rate, and option type.
6. Run chronological validation against CE/PE `Close`, then compute Greeks and daily delta-hedging results.
7. Only after those gates pass, expose inference through an API/frontend.
