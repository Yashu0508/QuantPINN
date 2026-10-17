"""Run the independent analytical solver-validation benchmark.

This benchmark is deliberately separate from the chronological NSE market
holdout. It samples a fixed, recorded parameter distribution, compares the
raw Kaggle PINN against Black--Scholes prices and Greeks, and applies the
thresholds in configs/solver_validation.json without changing them after
seeing the result.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.pinn.market import DEFAULT_CHECKPOINT, load_kaggle_market_checkpoint
from src.validation.gates import checkpoint_sha256
from src.pinn.parametric import predict_parametric_greeks, predict_parametric_price
from src.quant.black_scholes import black_scholes_greeks, black_scholes_price


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "solver_validation.json"
REPORT_PATH = ROOT / "reports" / "solver_validation.json"


def _mae(values: np.ndarray) -> float:
    return float(np.mean(np.abs(values)))


def _max_abs(values: np.ndarray) -> float:
    return float(np.max(np.abs(values)))


def _sample_scenarios(config: dict) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(int(config["benchmark_seed"]))
    n = int(config["scenario_count"])
    x = rng.uniform(*config["spot_over_strike"], size=n)
    tau = rng.uniform(*config["time_remaining_years"], size=n)
    sigma = rng.uniform(*config["volatility"], size=n)
    rate = rng.uniform(*config["rate"], size=n)
    # Keep tau <= T and keep the horizon inside the configured five-year
    # training domain. The equivalent-horizon check below separately uses T=tau.
    maturity = tau + (config["time_remaining_years"][1] - tau) * rng.random(n)
    code = np.where(rng.random(n) >= 0.5, 1.0, -1.0)
    return {
        "spot": x,
        "strike": np.ones(n, dtype=float),
        "tau": tau,
        "maturity": maturity,
        "sigma": sigma,
        "rate": rate,
        "code": code,
    }


def _black_scholes_arrays(scenarios: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    option_types = scenarios["code"]
    price = np.empty_like(option_types)
    greek_arrays = {name: np.empty_like(option_types) for name in ("delta", "gamma", "theta", "vega", "rho")}
    for code in (1.0, -1.0):
        mask = option_types == code
        kind = "CE" if code > 0 else "PE"
        price[mask] = black_scholes_price(
            scenarios["spot"][mask], scenarios["strike"][mask], scenarios["tau"][mask],
            scenarios["rate"][mask], scenarios["sigma"][mask], kind,
        )
        greeks = black_scholes_greeks(
            scenarios["spot"][mask], scenarios["strike"][mask], scenarios["tau"][mask],
            scenarios["rate"][mask], scenarios["sigma"][mask], kind,
        )
        for name, values in greeks.items():
            greek_arrays[name][mask] = values
    return price, greek_arrays


def _model_arrays(model, scenarios: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    price = predict_parametric_price(
        model, scenarios["spot"], scenarios["strike"], scenarios["tau"], scenarios["maturity"],
        scenarios["sigma"], scenarios["rate"], scenarios["code"],
    )
    greeks = predict_parametric_greeks(
        model, scenarios["spot"], scenarios["strike"], scenarios["tau"], scenarios["maturity"],
        scenarios["sigma"], scenarios["rate"], scenarios["code"],
    )
    return price, greeks


def main() -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    thresholds = config["thresholds"]
    scenarios = _sample_scenarios(config)
    model, metadata = load_kaggle_market_checkpoint()
    model_price, model_greeks = _model_arrays(model, scenarios)
    reference_price, reference_greeks = _black_scholes_arrays(scenarios)

    raw_price_error = model_price - reference_price
    raw_normalized_error = raw_price_error / scenarios["strike"]
    price_metrics = {
        "mae": _mae(raw_normalized_error),
        "max_abs_error": _max_abs(raw_normalized_error),
        "min_raw_price": float(np.min(model_price)),
    }
    greek_metrics = {}
    for name in ("delta", "gamma"):
        error = model_greeks[name] - reference_greeks[name]
        if name == "gamma":
            error = error * scenarios["strike"]
            key = "strike_scaled_gamma_mae"
        else:
            key = "delta_mae"
        greek_metrics[key] = _mae(error)

    # The same scenarios with T=tau test that the extra maturity coordinate
    # does not change an equivalent pricing horizon.
    equivalent_price = predict_parametric_price(
        model, scenarios["spot"], scenarios["strike"], scenarios["tau"], scenarios["tau"],
        scenarios["sigma"], scenarios["rate"], scenarios["code"],
    )
    equivalent_error = (equivalent_price - reference_price) / scenarios["strike"]
    equivalent_metrics = {
        "max_abs_error": _max_abs(equivalent_error),
        "mae": _mae(equivalent_error),
    }

    checks = {
        "price_over_strike_mae": price_metrics["mae"] <= thresholds["price_over_strike_mae"],
        "price_over_strike_max_error": price_metrics["max_abs_error"] <= thresholds["price_over_strike_max_error"],
        "delta_mae": greek_metrics["delta_mae"] <= thresholds["delta_mae"],
        "strike_scaled_gamma_mae": greek_metrics["strike_scaled_gamma_mae"] <= thresholds["strike_scaled_gamma_mae"],
        "equivalent_horizon_price_over_strike_max_error": equivalent_metrics["max_abs_error"] <= thresholds["equivalent_horizon_price_over_strike_max_error"],
    }
    report = {
        "status": "accepted" if all(checks.values()) else "failed",
        "protocol": config,
        "checkpoint_metadata": metadata,
        "checkpoint_sha256": checkpoint_sha256(DEFAULT_CHECKPOINT),
        "scenario_summary": {
            "count": int(len(scenarios["spot"])),
            "option_counts": {
                "CE": int(np.sum(scenarios["code"] > 0)),
                "PE": int(np.sum(scenarios["code"] < 0)),
            },
            "spot_over_strike": [float(np.min(scenarios["spot"])), float(np.max(scenarios["spot"]))],
            "tau_years": [float(np.min(scenarios["tau"])), float(np.max(scenarios["tau"]))],
            "maturity_years": [float(np.min(scenarios["maturity"])), float(np.max(scenarios["maturity"]))],
        },
        "raw_model_metrics": {
            "price_over_strike": price_metrics,
            "greeks": greek_metrics,
            "equivalent_horizon": equivalent_metrics,
        },
        "checks": checks,
        "interpretation": (
            "The checkpoint satisfies the independent analytical solver gate."
            if all(checks.values())
            else "The checkpoint does not satisfy the independent analytical solver gate; thresholds were not relaxed."
        ),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
