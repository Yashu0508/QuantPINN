"""Evaluate the parameterized PINN against analytical Black–Scholes scenarios."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.pinn.parametric import ParametricPINN, predict_parametric_price
from src.pinn.fixed_parameter import no_arbitrage_projection
from src.quant.black_scholes import black_scholes_price
from src.validation.metrics import regression_metrics


def evaluate_type(model, option_type: str, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    n = 1_000
    strike = rng.uniform(80, 140, n)
    spot = strike * rng.uniform(0.6, 1.4, n)
    maturity = rng.uniform(0.05, 1.0, n)
    tau = maturity * rng.uniform(0.02, 1.0, n)
    sigma = rng.uniform(0.12, 0.48, n)
    rate = rng.uniform(0.01, 0.09, n)
    raw = predict_parametric_price(model, spot, strike, tau, maturity, sigma, rate, option_type)
    projected = no_arbitrage_projection(raw, spot, strike, tau, rate, option_type)
    exact = black_scholes_price(spot, strike, tau, rate, sigma, option_type)
    return {
        "option_type": option_type,
        "rows": n,
        "raw": regression_metrics(exact, raw),
        "projected": regression_metrics(exact, projected),
        "raw_min": float(np.min(raw)),
        "projected_min": float(np.min(projected)),
    }


def main() -> None:
    payload = torch.load("models/parametric_pinn.pt", map_location="cpu", weights_only=False)
    model = ParametricPINN()
    model.load_state_dict(payload["model_state_dict"])
    result = {"CE": evaluate_type(model, "CE"), "PE": evaluate_type(model, "PE"), "exit_criterion_passed": False}
    result["reason"] = "Parameterized PINN is implemented and evaluated on synthetic scenarios; market calibration is a separate gate."
    Path("reports/parametric_pinn_evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
