# NSE-calibrated NIFTY PINN engine

This repository builds a daily end-of-day NIFTY CE/PE option-pricing research
engine. Raw CSV files remain immutable. `preprocess_datasets.py` creates the
canonical Parquet data, `validate_data.py` checks the data gates, and
`src/quant/` contains Black–Scholes, Monte Carlo, Greeks, and implied-volatility
benchmarks.

The model is a Physics-Informed Neural Network: it learns the Black–Scholes PDE,
payoff, and boundary conditions using calibrated spot, maturity, volatility, and
rate parameters. Market option `Close` is evaluation-only. Final parameterized
training is performed on Kaggle GPU through
`kaggle/nse-pinn-market-training/`; local `.pt` files from the earlier prototype
are exploratory artifacts.

Useful checks:

```powershell
python preprocess_datasets.py
python validate_data.py
python -m pytest -q
python evaluate_acceptance.py
python evaluate_hedging.py
```

The research API is started with:

```powershell
uvicorn src.api.app:app --reload
```

Production deployment remains gated by an independently approved model
acceptance protocol. See `PINN_MODEL_PLAN.md` and `reports/` for evidence and
limitations.
 
