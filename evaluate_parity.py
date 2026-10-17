"""Evaluate CE/PE parity diagnostics on paired eligible snapshots."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.validation.parity import put_call_parity_residual


def main() -> None:
    cols = [
        "valuation_date", "expiry", "strike", "spot", "time_to_maturity", "close",
        "risk_free_rate_decimal", "eligible_for_pure_pde_pricing", "price_quality",
    ]
    ce = pd.read_parquet("processed/market_snapshot_ce.parquet", columns=cols).rename(columns={"close": "call_close", "price_quality": "call_quality"})
    pe = pd.read_parquet("processed/market_snapshot_pe.parquet", columns=cols).rename(columns={"close": "put_close", "price_quality": "put_quality"})
    key = ["valuation_date", "expiry", "strike"]
    frame = ce.merge(pe, on=key, suffixes=("_ce", "_pe"))
    frame = frame[
        frame["eligible_for_pure_pde_pricing_ce"]
        & frame["eligible_for_pure_pde_pricing_pe"]
        & frame["call_close"].notna()
        & frame["put_close"].notna()
        & frame["time_to_maturity_ce"].gt(0)
    ].copy()
    residual = put_call_parity_residual(
        frame["call_close"], frame["put_close"], frame["spot_ce"], frame["strike"],
        frame["time_to_maturity_ce"], frame["risk_free_rate_decimal_ce"],
    )
    result = {
        "paired_eligible_rows": int(len(frame)),
        "residual_mean": float(np.mean(residual)),
        "residual_median": float(np.median(residual)),
        "residual_mae": float(np.mean(np.abs(residual))),
        "residual_rmse": float(np.sqrt(np.mean(residual**2))),
        "residual_p95_abs": float(np.quantile(np.abs(residual), 0.95)),
        "interpretation": "Diagnostic only; large residuals may reflect carry/dividends, stale or illiquid quotes, and contract conventions.",
    }
    Path("reports/put_call_parity_evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
