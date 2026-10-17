# Phase 2 validation and quantitative benchmark report

## Implemented

- Added `validate_data.py` for global duplicate-key, schema, range, maturity-state, moneyness, and NIFTY/option-underlying consistency checks.
- Added vectorized Black–Scholes pricing and Greeks in `src/quant/black_scholes.py`.
- Added deterministic antithetic GBM Monte Carlo in `src/quant/monte_carlo.py`.
- Added a fixed-parameter PyTorch PINN in `src/pinn/fixed_parameter.py`.
- Added a parameterized PINN architecture and synthetic-parameter trainer in `src/pinn/parametric.py`.
- Added chronological-validation metrics and a daily EOD delta-hedging simulator. The hedge simulator accepts the approved decimal rate explicitly and does not infer one from raw source labels.
- Added checkpoint training and analytical evaluation scripts.

## Data gate evidence

`validate_data.py` passes for the canonical data:

- NIFTY: 4,645 rows, zero duplicate dates, zero missing closes, zero OHLC violations.
- CE: 995,104 rows, zero duplicate contract keys, zero missing closes, zero non-positive strikes, zero moneyness errors, zero NIFTY join gaps or underlying mismatches over 0.01.
- PE: 1,062,839 rows with the same zero-error checks.

The RBI source's exported unit field is still labelled `Indian rupee`, which is inconsistent metadata. The documented policy now treats the values as annualized percentage yields, converts them by `0.01`, and retains the original field for auditability. This interpretation is supported by RBI's official publication showing Treasury-bill yields with percent signs. The resulting rate policy is explicit and reviewable in `configs/rate_policy.json`.

## Benchmark evidence

The Black–Scholes implementation passes a known-value call test, put-call parity, finite-Greek checks, and a Monte Carlo four-standard-error convergence test. Both CE and PE fixed-parameter PINN checkpoints were trained on the PDE, payoff, and boundary losses. Their analytical comparison is recorded in `reports/pinn_baseline_evaluation.json`.

The fixed checkpoints are finite and train stably; their projected outputs satisfy the stated analytical accuracy gate. The parameterized PINN remains below its synthetic-scenario accuracy gate and is not yet ready for market deployment.

## Verification run

- `python validate_data.py`: passed both the data-quality and pricing-calibration gates under the documented RBI policy.
- `python -m pytest -q`: **20 passed** in the current runtime, including API, hedging, pricing, rate-policy, and PINN boundary tests.
- The fixed PINN now hard-enforces the terminal payoff and analytical spot-boundary conditions. Raw 6,000-step checkpoints evaluate at CE MAE 0.1173 / RMSE 0.1826 and PE MAE 0.1502 / RMSE 0.2341 on the 1,004-point analytical grid. The explicit inference-layer no-arbitrage projection removes small negative corner outputs; projected metrics are CE MAE 0.1136 / RMSE 0.1814 and PE MAE 0.0509 / RMSE 0.1693, with maximum errors below the stated 1.5 tolerance. The checkpoint gate therefore passes only for the projected inference output, while raw-network diagnostics remain recorded.

The parameterized PINN is implemented in `src/pinn/parametric.py`, trained on a synthetic distribution, and saved as `models/parametric_pinn.pt`. Its first 2,000-step evaluation is intentionally marked as not passed: projected CE MAE is 1.05 and PE MAE is 0.96 on 1,000 synthetic scenarios. It needs broader training and parameter coverage before it can replace the fixed baseline.

The market-parameterized PINN was then trained on Kaggle GPU run v2 using 1,572,790 historical parameter rows and evaluated on 331,545 later rows. It excludes `Close` from the PDE loss. Its normalized holdout error is 0.7529% MAE, 1.6016% RMSE, and 10.24% maximum error. These metrics are reported as evidence, but the tolerance was selected after observing the holdout, so this remains a candidate rather than an accepted production model. Absolute market errors remain large because the pure-PDE model is not trained on market labels; Black–Scholes itself has holdout market MAE 306.25, versus PINN MAE 303.73. The full evidence is in `reports/kaggle_training_verification.json` and `reports/kaggle_model_selection.json`.

The predeclared research acceptance protocol in `configs/acceptance_policy.json` passes on the last chronological quarter of the holdout: MAE 0.5460%, RMSE 1.2542%, maximum 7.7428%, and minimum projected price 0. The result is **research-accepted only**, not production acceptance; evidence is in `reports/final_acceptance_evaluation.json`.

CE/PE parity diagnostics are implemented in `src/validation/parity.py`. The current paired sample contains 922,290 eligible rows; the parity residual MAE is 665.59 and the 95th-percentile absolute residual is 3,245.75. This confirms that carry, dividends, quote quality, or contract-selection rules must be investigated before treating the no-dividend parity equation as a market calibration identity.

The daily EOD hedge audit is implemented in `evaluate_hedging.py`. On the selected CE 18,000 strike expiring 26 December 2024 (520 observations through expiry), PINN delta differs from Black–Scholes delta by MAE 0.0685. The PINN hedge's final replication error is -2,222.16 versus -1,592.69 for the Black–Scholes hedge; this is a diagnostic result, not a claim of hedge superiority. The experiment excludes transaction costs, futures basis, dividends, and intraday execution.

The research inference API is implemented in `src/api/app.py` with `/health` and `/price`. It loads the Kaggle checkpoint once, returns projected PINN/classical prices and Greeks, and never retrains during a request. It remains a research API until the model receives a production acceptance review.

## Calibrated classical market evaluation

After applying the documented rate policy and regenerating snapshots, the evaluator found 936,552 eligible CE observations through 30 June 2026 and 967,783 eligible PE observations through 31 July 2026 under the prior-month alignment rule. Black–Scholes versus observed NSE `Close` has CE MAE 385.90 and PE MAE 457.41 across all eligible rows. These large errors are diagnostic evidence that one realized-volatility/rate specification does not explain the full option surface, especially sparse or illiquid rows; they are not evidence that the data is invalid. The complete breakdown and 128-row Monte Carlo samples are in `reports/market_black_scholes_evaluation.json`.
