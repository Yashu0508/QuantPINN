# Inference API status

`src/api/app.py` exposes `/health` and `/price`. It loads the Kaggle-trained
research checkpoint once, never retrains during a request, and returns the
projected PINN price, raw PINN price, Black–Scholes price, Monte Carlo estimate,
and Black–Scholes Greeks. It intentionally does not claim production readiness
or fabricate a market `Close` value.
