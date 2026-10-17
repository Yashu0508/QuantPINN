"""Describe previously examined predictions; this is not independent acceptance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    error = (frame["pinn_price"] - frame["black_scholes_price"]) / frame["strike"]
    return {
        "rows": int(len(frame)),
        "normalized_mae": float(np.mean(np.abs(error))),
        "normalized_rmse": float(np.sqrt(np.mean(error**2))),
        "normalized_max_abs": float(np.max(np.abs(error))),
        "minimum_price": float(frame["pinn_price"].min()),
    }


def main() -> None:
    policy = json.loads((ROOT / "configs" / "acceptance_policy.json").read_text(encoding="utf-8"))
    frame = pd.read_parquet(ROOT / "reports" / "kaggle_market_predictions.parquet")
    frame["valuation_date"] = pd.to_datetime(frame["valuation_date"])
    unique_dates = np.sort(frame["valuation_date"].dt.date.unique())
    start = unique_dates[int(len(unique_dates) * 0.75)]
    audit = frame[frame["valuation_date"].dt.date >= start].copy()
    result = {
        "checkpoint": "models/market_parametric_pinn_kaggle.pt",
        "training_source": "Kaggle GPU run v2",
        "audit_start": str(start),
        "audit_end": str(frame["valuation_date"].max().date()),
        "policy": policy,
        "all_holdout": metrics(frame),
        "audit_slice": metrics(audit),
        "market_close_diagnostic_mae": float(np.mean(np.abs(audit["pinn_price"] - audit["close"]))),
    }
    thresholds = policy["thresholds"]
    audit_metrics = result["audit_slice"]
    result["descriptive_thresholds_met"] = bool(
        audit_metrics["normalized_mae"] <= thresholds["normalized_mae_max"]
        and audit_metrics["normalized_rmse"] <= thresholds["normalized_rmse_max"]
        and audit_metrics["normalized_max_abs"] <= thresholds["normalized_max_abs_max"]
        and audit_metrics["minimum_price"] >= thresholds["minimum_price"]
    )
    result["passed"] = False
    result["status"] = "candidate_only"
    result["reason"] = "This quarter is part of the holdout already used to compare v2 and v3; new thresholds cannot make it independent."
    (ROOT / "reports" / "final_acceptance_evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit("Acceptance not established: descriptive evaluation of reused data only")


if __name__ == "__main__":
    main()
