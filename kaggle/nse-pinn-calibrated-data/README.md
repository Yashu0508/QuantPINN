# NIFTY calibrated PINN training data

This private Kaggle dataset contains canonical Parquet snapshots generated from the
raw NIFTY, CE, PE, and RBI files. Training code must use only calibrated parameter
columns (`spot`, `strike`, `time_to_maturity`, realized volatility, and validated
risk-free rate). `close` is retained only for post-training market evaluation and
must not enter the PDE training loss.
