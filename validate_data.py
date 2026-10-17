"""Validate canonical datasets and write a machine-readable phase gate report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.calibration.rates import load_rate_policy


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "processed"
REPORTS = ROOT / "reports"


def validate_option(path: Path, expected_type: str, underlying: pd.DataFrame) -> dict:
    cols = [
        "valuation_date", "expiry", "option_type", "strike", "spot", "close",
        "underlying_value", "time_to_expiry_days", "time_to_maturity", "contract_state",
        "moneyness",
    ]
    df = pd.read_parquet(path, columns=cols)
    key = ["valuation_date", "expiry", "strike", "option_type"]
    dates = pd.to_datetime(df["valuation_date"])
    exp = pd.to_datetime(df["expiry"])
    checks = {
        "row_count": int(len(df)),
        "option_type_only_expected": bool(df["option_type"].eq(expected_type).all()),
        "duplicate_contract_keys": int(df.duplicated(key).sum()),
        "missing_close": int(df["close"].isna().sum()),
        "nonpositive_close": int((df["close"] <= 0).sum()),
        "nonpositive_strike": int((df["strike"] <= 0).sum()),
        "missing_spot": int(df["spot"].isna().sum()),
        "missing_underlying_value": int(df["underlying_value"].isna().sum()),
        "date_after_expiry_state_errors": int(
            ((df["time_to_expiry_days"] < 0) & (df["contract_state"] != "expired")).sum()
            + ((df["time_to_expiry_days"] == 0) & (df["contract_state"] != "maturity_day")).sum()
            + ((df["time_to_expiry_days"] > 0) & (df["contract_state"] != "active")).sum()
        ),
        "moneyness_errors": int((~np.isclose(df["moneyness"], df["spot"] / df["strike"], equal_nan=False)).sum()),
    }
    joined = df[["valuation_date", "spot", "underlying_value"]].copy()
    joined["date"] = pd.to_datetime(joined["valuation_date"])
    base = underlying.copy()
    base["date"] = pd.to_datetime(base["valuation_date"])
    joined = joined.merge(base[["date", "close"]], on="date", how="left", suffixes=("_option", "_nifty"))
    diff = (joined["underlying_value"] - joined["close"]).abs()
    checks["nifty_close_join_missing"] = int(joined["close"].isna().sum())
    checks["underlying_nifty_mismatch_over_0_01"] = int((diff > 0.01).sum())
    checks["date_min"] = str(dates.min().date())
    checks["date_max"] = str(dates.max().date())
    checks["expiry_min"] = str(exp.min().date())
    checks["expiry_max"] = str(exp.max().date())
    checks["passed"] = (
        checks["option_type_only_expected"]
        and checks["duplicate_contract_keys"] == 0
        and checks["missing_close"] == 0
        and checks["nonpositive_close"] == 0
        and checks["nonpositive_strike"] == 0
        and checks["missing_spot"] == 0
        and checks["missing_underlying_value"] == 0
        and checks["date_after_expiry_state_errors"] == 0
        and checks["moneyness_errors"] == 0
        and checks["nifty_close_join_missing"] == 0
        and checks["underlying_nifty_mismatch_over_0_01"] == 0
    )
    return checks


def main() -> None:
    underlying = pd.read_parquet(PROCESSED / "underlying_daily.parquet")
    policy = load_rate_policy(ROOT / "configs" / "rate_policy.json")
    result = {
        "underlying": {
            "rows": int(len(underlying)),
            "duplicate_dates": int(underlying["valuation_date"].duplicated().sum()),
            "missing_close": int(underlying["close"].isna().sum()),
            "ohlc_invalid": int((~underlying["ohlc_valid"]).sum()),
        },
        "CE": validate_option(PROCESSED / "option_eod_ce.parquet", "CE", underlying),
        "PE": validate_option(PROCESSED / "option_eod_pe.parquet", "PE", underlying),
        "rbi": {
            "rows": int(len(pd.read_parquet(PROCESSED / "risk_free_monthly.parquet"))),
            "rate_unit_verified": bool(policy["approved"]),
            "rate_policy": policy,
        },
    }
    result["data_quality_gate_passed"] = (
        result["underlying"]["duplicate_dates"] == 0
        and result["underlying"]["missing_close"] == 0
        and result["underlying"]["ohlc_invalid"] == 0
        and result["CE"]["passed"]
        and result["PE"]["passed"]
    )
    result["pricing_calibration_gate_passed"] = bool(
        result["rbi"]["rate_unit_verified"]
        and result["CE"]["passed"]
        and result["PE"]["passed"]
    )
    (REPORTS / "phase1_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["data_quality_gate_passed"]:
        raise SystemExit("data quality validation failed")


if __name__ == "__main__":
    main()
