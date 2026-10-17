"""Chronological Black–Scholes evaluation against eligible NSE observations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.quant.black_scholes import black_scholes_price
from src.quant.monte_carlo import monte_carlo_price
from src.validation.metrics import regression_metrics


ROOT = Path(__file__).resolve().parent


def evaluate_type(option_type: str, monte_carlo_rows: int = 128) -> dict:
    path = ROOT / "processed" / f"market_snapshot_{option_type.lower()}.parquet"
    columns = [
        "valuation_date", "expiry", "option_type", "strike", "spot", "time_to_maturity",
        "close", "realized_vol_20d", "risk_free_rate_decimal", "eligible_for_pure_pde_pricing",
        "contract_state", "moneyness", "price_quality",
    ]
    frame = pd.read_parquet(path, columns=columns)
    eligible = frame[frame["eligible_for_pure_pde_pricing"] & frame["close"].notna()].copy()
    eligible = eligible[eligible["time_to_maturity"] > 0]
    model = black_scholes_price(
        eligible["spot"].to_numpy(), eligible["strike"].to_numpy(), eligible["time_to_maturity"].to_numpy(),
        eligible["risk_free_rate_decimal"].to_numpy(), eligible["realized_vol_20d"].to_numpy(), option_type,
    )
    actual = eligible["close"].to_numpy()
    result = {
        "option_type": option_type,
        "eligible_rows": int(len(eligible)),
        "date_min": str(pd.to_datetime(eligible["valuation_date"]).min().date()),
        "date_max": str(pd.to_datetime(eligible["valuation_date"]).max().date()),
        "metrics_all": regression_metrics(actual, model),
        "metrics_by_price_quality": {},
    }
    for quality, group in eligible.assign(model=model).groupby("price_quality"):
        result["metrics_by_price_quality"][quality] = regression_metrics(group["close"], group["model"])

    # Monte Carlo is intentionally sampled because a full 1M-row simulation is not a useful benchmark.
    sample = eligible.sample(n=min(monte_carlo_rows, len(eligible)), random_state=7)
    mc_errors = []
    mc_standard_errors = []
    for row in sample.itertuples(index=False):
        mc = monte_carlo_price(
            float(row.spot), float(row.strike), float(row.time_to_maturity),
            float(row.risk_free_rate_decimal), float(row.realized_vol_20d), option_type,
            paths=20_000, seed=7,
        )
        mc_errors.append(mc["price"] - float(row.close))
        mc_standard_errors.append(mc["standard_error"])
    result["monte_carlo_sample"] = {
        "rows": len(mc_errors),
        "mae_vs_market_close": float(np.mean(np.abs(mc_errors))),
        "mean_standard_error": float(np.mean(mc_standard_errors)),
    }
    return result


def main() -> None:
    result = {"CE": evaluate_type("CE"), "PE": evaluate_type("PE")}
    Path("reports/market_black_scholes_evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
