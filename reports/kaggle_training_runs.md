# Kaggle training runs

## Run 1

The notebook launched on Kaggle GPU but failed immediately because it assumed a fixed input directory that Kaggle did not mount under that name. No model artifact from this run was accepted.

## Run 2

The notebook was corrected to discover attached Parquet files recursively and to normalize volatility, rate, and maturity inputs. It completed on CUDA for 12,000 steps.

- training rows: 1,572,790
- chronological cutoff: 2025-09-23
- holdout rows: 331,545
- holdout: 2025-09-24 through 2026-07-31
- option `Close` in PDE training loss: false
- normalized (V/K) error versus Black–Scholes: MAE 0.7529%, RMSE 1.6016%, maximum 10.24%
- market `Close` MAE: 303.73

Run 2 is the best current candidate and is stored as `models/market_parametric_pinn_kaggle.pt`. The normalized tolerance is reported as a diagnostic, not an independent acceptance gate.

## Run 3

A maturity-balanced sampler was tested on Kaggle GPU. It completed but was worse: normalized MAE 1.3278%, RMSE 2.7540%, maximum 16.63%, and market MAE 398.69. It was rejected and is not the selected model.
