"""Evaluate saved fixed-parameter PINN checkpoints against Black–Scholes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.pinn.fixed_parameter import FixedParameterPINN, no_arbitrage_projection, predict_price
from src.quant.black_scholes import black_scholes_price


def evaluate(option_type: str, checkpoint: Path) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = FixedParameterPINN(
        option_type=option_type,
        maturity=float(payload["maturity"]),
        rate=float(payload["rate"]),
    )
    model.load_state_dict(payload["model_state_dict"])
    spots = np.linspace(50.0, 300.0, 251)
    taus = np.array([0.01, 0.05, 0.10, 0.25])
    spot_grid, tau_grid = np.meshgrid(spots, taus)
    pinn = predict_price(model, spot_grid, 100.0, tau_grid, 0.25)
    projected = no_arbitrage_projection(pinn, spot_grid, 100.0, tau_grid, 0.05, option_type)
    exact = black_scholes_price(spot_grid, 100.0, tau_grid, 0.05, 0.20, option_type)
    error = projected - exact
    raw_error = pinn - exact
    metrics = {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_abs_error": float(np.max(np.abs(error))),
    }
    return {
        "option_type": option_type,
        "checkpoint": str(checkpoint),
        "grid_points": int(error.size),
        "raw_mae": float(np.mean(np.abs(raw_error))),
        "raw_rmse": float(np.sqrt(np.mean(raw_error**2))),
        "raw_min_pinn_price": float(np.min(pinn)),
        **metrics,
        "min_pinn_price": float(np.min(projected)),
        "pinn_at_spot_100": float(no_arbitrage_projection(predict_price(model, 100.0, 100.0, 0.25, 0.25), 100.0, 100.0, 0.25, 0.05, option_type)),
        "black_scholes_at_spot_100": float(black_scholes_price(100.0, 100.0, 0.25, 0.05, 0.20, option_type)),
    }


def main() -> None:
    result = {
        "CE": evaluate("CE", Path("models/fixed_parameter_ce_baseline.pt")),
        "PE": evaluate("PE", Path("models/fixed_parameter_pe_baseline.pt")),
        "tolerance": {"mae": 0.25, "rmse": 0.50, "max_abs_error": 1.50},
    }
    result["exit_criterion_passed"] = all(
        result[typ]["mae"] <= result["tolerance"]["mae"]
        and result[typ]["rmse"] <= result["tolerance"]["rmse"]
        and result[typ]["max_abs_error"] <= result["tolerance"]["max_abs_error"]
        and result[typ]["min_pinn_price"] >= -1e-6
        for typ in ("CE", "PE")
    )
    result["reason"] = "Both fixed-parameter checkpoints meet the stated analytical grid tolerance after the explicit no-arbitrage inference projection." if result["exit_criterion_passed"] else "At least one checkpoint misses the stated analytical grid tolerance after projection."
    Path("reports/pinn_baseline_evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
