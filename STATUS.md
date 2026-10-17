# NSE-calibrated NIFTY PINN engine: implementation status

Updated: 2026-10-14

## What is implemented

- Immutable raw CSV ingestion and canonical Parquet preprocessing.
- NIFTY underlying normalization with backward-looking realized-volatility fields.
- CE/PE option canonicalization, contract-state classification, maturity and moneyness fields.
- RBI rate policy with explicit percentage-to-decimal conversion and prior-calendar-month alignment.
- Data validation for duplicates, ranges, maturity state, moneyness, and option underlying consistency.
- Black–Scholes European CE/PE prices and Greeks.
- Antithetic Monte Carlo GBM benchmark with paired standard-error calculation.
- Implied volatility, put-call parity diagnostics, fixed PINN, parameterized PINN, and daily EOD delta hedging.
- FastAPI research inference endpoint loading a checkpoint once without training per request.
- Explicit API input rejection for non-finite values, unsupported PINN spatial domain, missing checkpoints, invalid horizons, and odd Monte Carlo path counts.
- Fail-closed solver gate requiring every independent analytical check to be present and exactly boolean true.
- Solver report checkpoint SHA-256 binding so an accepted report cannot be reused for a different binary.
- Git ignore rules preventing raw data, Parquet outputs, model binaries, logs, caches, and secrets from entering the private repository.
- Reproducible dependency manifest and pytest configuration.

## Verified working

- `python -m pytest -q`: 27 passed.
- `python validate_data.py`: data_quality_gate_passed=true and pricing_calibration_gate_passed=true.
- `python evaluate_solver_validation.py`: runs reproducibly and writes `reports/solver_validation.json`, including the checkpoint SHA.

## Known failure and limitation

The current Kaggle v2 checkpoint is not an accepted analytical PDE solver. The independent solver report fails all five checks:

- normalized price MAE: 0.1288801478 (threshold 0.001)
- normalized maximum price error: 0.4179729055 (threshold 0.01)
- delta MAE: 0.1317828755 (threshold 0.02)
- strike-scaled gamma MAE: 0.4904005383 (threshold 0.1)
- equivalent-horizon maximum price error: 0.4172341655 (threshold 0.001)

The API therefore reports `research_only`. Price projection is used only as an explicitly labelled no-arbitrage presentation layer; it does not turn the checkpoint into a validated solver. The project does not claim production pricing accuracy, live execution, intraday hedging, or independent market acceptance.

## Next required research milestone

Train a fresh synthetic-only PINN and select checkpoints only against the independent analytical protocol. Keep the existing Kaggle market holdout separate for descriptive market diagnostics. Do not change thresholds to make the current checkpoint pass.
