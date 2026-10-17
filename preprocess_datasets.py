"""Build canonical, leakage-aware datasets from the supplied raw files.

Raw files are never modified. Outputs are written below processed/ and reports/.
The RBI source value and source-unit label are retained for auditability. The
approved rate policy converts the values from annualized percentage points to
decimal rates while preserving the source metadata defect.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.calibration.rates import load_rate_policy, source_value_to_decimal


ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "processed"
REPORTS = ROOT / "reports"
PROCESSED.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)


def clean_name(value: str) -> str:
    value = value.strip().lower().replace("₹", "rupee")
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return value


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, compression="zstd")


def read_underlying() -> tuple[pd.DataFrame, dict]:
    source = ROOT / "nifty50_historical.csv"
    df = pd.read_csv(source)
    df.columns = [clean_name(c) for c in df.columns]
    df = df.rename(columns={
        "date": "valuation_date",
        "turnover_rupee_cr": "turnover_cr",
    })
    df["valuation_date"] = pd.to_datetime(
        df["valuation_date"], format="%d-%b-%Y", errors="coerce"
    )
    numeric = ["open", "high", "low", "close", "shares_traded", "turnover_cr"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("valuation_date").reset_index(drop=True)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    for window in (20, 60, 252):
        df[f"realized_vol_{window}d"] = (
            df["log_return"].rolling(window, min_periods=window).std(ddof=1)
            * math.sqrt(252.0)
        )
    df["ohlc_valid"] = (
        df[["open", "high", "low", "close"]].notna().all(axis=1)
        & (df["low"] <= df[["open", "close"]].min(axis=1))
        & (df["high"] >= df[["open", "close"]].max(axis=1))
        & (df["low"] <= df["high"])
    )
    df["valuation_date"] = df["valuation_date"].dt.date
    write_parquet(df, PROCESSED / "underlying_daily.parquet")
    summary = {
        "rows": int(len(df)),
        "date_min": str(df["valuation_date"].min()),
        "date_max": str(df["valuation_date"].max()),
        "duplicate_dates": int(df["valuation_date"].duplicated().sum()),
        "missing_by_column": {k: int(v) for k, v in df.isna().sum().items()},
        "ohlc_invalid_complete_rows": int((~df["ohlc_valid"]).sum()),
    }
    return df, summary


def read_rbi() -> tuple[pd.DataFrame, dict]:
    source = ROOT / "rbi_tbill_yield_15_91d_monthly.csv"
    # The first line is a parser directive, not a data row.
    df = pd.read_csv(source, sep="|", skiprows=1)
    df.columns = [clean_name(c) for c in df.columns]
    df = df.rename(columns={
        "element": "element",
        "term_to_maturity_in_days": "term_to_maturity_days",
        "time": "observation_month",
        "unit": "source_unit",
        "value_in_actuals": "source_value",
    })
    df["observation_month"] = pd.PeriodIndex(
        pd.to_datetime(df["observation_month"], format="%m/%Y", errors="coerce"),
        freq="M",
    )
    df["source_value"] = pd.to_numeric(df["source_value"], errors="coerce")
    policy = load_rate_policy(ROOT / "configs" / "rate_policy.json")
    df["rate_status"] = "verified_percentage_yield" if policy["approved"] else "unit_unverified"
    df["risk_free_rate_decimal"] = df["source_value"].map(
        lambda value: source_value_to_decimal(value, policy) if pd.notna(value) else np.nan
    )
    df = df.sort_values("observation_month").reset_index(drop=True)
    # Parquet does not preserve pandas Period reliably across all consumers.
    df["observation_month"] = df["observation_month"].astype(str)
    write_parquet(df, PROCESSED / "risk_free_monthly.parquet")
    summary = {
        "rows": int(len(df)),
        "month_min": str(df["observation_month"].min()),
        "month_max": str(df["observation_month"].max()),
        "source_units": sorted(df["source_unit"].dropna().unique().tolist()),
        "source_value_min": float(df["source_value"].min()),
        "source_value_max": float(df["source_value"].max()),
        "rate_status": df["rate_status"].iloc[0],
        "decimal_scale": float(policy["decimal_scale"]),
        "policy_approved": bool(policy["approved"]),
    }
    return df, summary


def canonical_option_columns(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk.columns = [clean_name(c) for c in chunk.columns]
    chunk = chunk.rename(columns={
        "date": "valuation_date",
        "option_type": "option_type",
        "strike_price": "strike",
        "settle_price": "settle_price",
        "no_of_contracts": "contracts",
        "turnover_in_rupee_lakhs": "turnover_lakhs",
        "premium_turnover_in_rupee_lakhs": "premium_turnover_lakhs",
        "open_int": "open_interest",
        "underlying_value": "underlying_value",
    })
    chunk["valuation_date"] = pd.to_datetime(chunk["valuation_date"], errors="coerce")
    chunk["expiry"] = pd.to_datetime(chunk["expiry"], errors="coerce")
    numeric = [
        "strike", "open", "high", "low", "close", "ltp", "settle_price",
        "contracts", "turnover_lakhs", "premium_turnover_lakhs", "open_interest",
        "change_in_oi", "underlying_value",
    ]
    for col in numeric:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

    chunk["time_to_expiry_days"] = (chunk["expiry"] - chunk["valuation_date"]).dt.days
    chunk["time_to_maturity"] = chunk["time_to_expiry_days"] / 365.0
    chunk["contract_state"] = np.select(
        [chunk["time_to_expiry_days"] < 0, chunk["time_to_expiry_days"] == 0],
        ["expired", "maturity_day"],
        default="active",
    )
    chunk["moneyness"] = chunk["underlying_value"] / chunk["strike"]
    chunk["log_moneyness"] = np.log(chunk["moneyness"].where(chunk["moneyness"] > 0))
    for col, flag in [
        ("ltp", "has_ltp"),
        ("settle_price", "has_settlement"),
        ("contracts", "has_volume"),
        ("open_interest", "has_oi"),
        ("change_in_oi", "has_change_oi"),
    ]:
        chunk[flag] = chunk[col].notna()
    complete = chunk[["open", "high", "low", "close"]].notna().all(axis=1)
    chunk["ohlc_complete"] = complete
    chunk["ohlc_valid"] = (
        complete
        & (chunk["low"] <= chunk[["open", "close"]].min(axis=1))
        & (chunk["high"] >= chunk[["open", "close"]].max(axis=1))
        & (chunk["low"] <= chunk["high"])
    )
    chunk["price_quality"] = np.select(
        [chunk["close"].isna() | (chunk["close"] <= 0), ~chunk["ohlc_complete"], ~chunk["ohlc_valid"]],
        ["invalid_close", "close_only", "ohlc_inconsistent"],
        default="complete_ohlc",
    )
    chunk["valuation_date"] = chunk["valuation_date"].dt.date
    chunk["expiry"] = chunk["expiry"].dt.date
    return chunk


def process_options(underlying: pd.DataFrame, rbi: pd.DataFrame) -> tuple[dict, dict]:
    underlying_lookup = underlying.set_index("valuation_date")
    rbi_indexed = rbi.set_index("observation_month")
    rbi_lookup = rbi_indexed["risk_free_rate_decimal"]
    rbi_source_lookup = rbi_indexed["source_value"]
    summaries = {}
    for option_type, filename in [("CE", "nifty_ce_options_model_ready.csv"), ("PE", "nifty_pe_options_model_ready.csv")]:
        source = ROOT / filename
        canonical_path = PROCESSED / f"option_eod_{option_type.lower()}.parquet"
        snapshot_path = PROCESSED / f"market_snapshot_{option_type.lower()}.parquet"
        canonical_writer = None
        snapshot_writer = None
        totals = {"rows": 0, "missing": {}, "states": {}, "price_quality": {}}
        for raw in pd.read_csv(source, chunksize=100_000, low_memory=False):
            chunk = canonical_option_columns(raw)
            dates = pd.Index(chunk["valuation_date"])
            for col in ["close", "close", "underlying_value", "realized_vol_20d", "realized_vol_60d", "realized_vol_252d"]:
                pass
            for col in ["close", "turnover_lakhs", "premium_turnover_lakhs", "open_interest"]:
                totals["missing"][col] = totals["missing"].get(col, 0) + int(chunk[col].isna().sum())
            totals["rows"] += len(chunk)
            for state, count in chunk["contract_state"].value_counts().items():
                totals["states"][state] = totals["states"].get(state, 0) + int(count)
            for quality, count in chunk["price_quality"].value_counts().items():
                totals["price_quality"][quality] = totals["price_quality"].get(quality, 0) + int(count)

            # Join canonical NIFTY values by valuation date.
            join = underlying_lookup.reindex(dates)
            chunk["spot"] = join["close"].to_numpy()
            for window in (20, 60, 252):
                chunk[f"realized_vol_{window}d"] = join[f"realized_vol_{window}d"].to_numpy()

            # Conservative prior-month as-of alignment avoids using a current-month aggregate.
            month = pd.to_datetime(chunk["valuation_date"]).dt.to_period("M")
            prior_month = (month - 1).astype(str)
            chunk["rbi_observation_month_asof"] = prior_month.to_numpy()
            chunk["rbi_source_value_asof"] = [rbi_source_lookup.get(m, np.nan) for m in prior_month]
            chunk["risk_free_rate_decimal"] = [rbi_lookup.get(m, np.nan) for m in prior_month]
            chunk["rate_status"] = np.where(
                pd.notna(chunk["risk_free_rate_decimal"]), "verified_percentage_yield", "missing_prior_rbi_observation"
            )
            chunk["calibration_status"] = np.where(
                chunk["spot"].notna() & chunk["realized_vol_20d"].notna() & chunk["risk_free_rate_decimal"].notna(),
                "ready",
                "missing_calibration_input",
            )
            chunk["eligible_for_pure_pde_pricing"] = (
                chunk["contract_state"].eq("active")
                & chunk["spot"].notna()
                & chunk["realized_vol_20d"].notna()
                & chunk["risk_free_rate_decimal"].notna()
            )

            snapshot_cols = [
                "valuation_date", "expiry", "option_type", "strike", "spot", "time_to_expiry_days",
                "time_to_maturity", "contract_state", "close", "underlying_value", "moneyness",
                "log_moneyness", "realized_vol_20d", "realized_vol_60d", "realized_vol_252d",
                "rbi_observation_month_asof", "rbi_source_value_asof", "risk_free_rate_decimal",
                "rate_status", "calibration_status", "eligible_for_pure_pde_pricing",
                "price_quality", "has_ltp", "has_settlement", "has_volume", "has_oi", "has_change_oi",
            ]
            snapshot = chunk[snapshot_cols].copy()
            for df, path, writer_name in [
                (chunk, canonical_path, "canonical_writer"),
                (snapshot, snapshot_path, "snapshot_writer"),
            ]:
                table = pa.Table.from_pandas(df, preserve_index=False)
                if writer_name == "canonical_writer":
                    if canonical_writer is None:
                        canonical_writer = pq.ParquetWriter(path, table.schema, compression="zstd")
                    canonical_writer.write_table(table)
                else:
                    if snapshot_writer is None:
                        snapshot_writer = pq.ParquetWriter(path, table.schema, compression="zstd")
                    snapshot_writer.write_table(table)
        if canonical_writer is not None:
            canonical_writer.close()
        if snapshot_writer is not None:
            snapshot_writer.close()
        summaries[option_type] = totals
    return summaries, {
        "design": "market_snapshot outputs retain all rows; pricing eligibility requires a supported prior-month rate and realized-volatility window",
        "rate_alignment": "prior calendar month",
    }


def write_report(underlying_summary: dict, rbi_summary: dict, option_summary: dict, design: dict) -> None:
    report = {
        "underlying": underlying_summary,
        "rbi": rbi_summary,
        "options": option_summary,
        "design": design,
    }
    (REPORTS / "phase1_preprocessing_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    text = f"""# Phase 1 preprocessing report

The raw files in the project root were preserved unchanged. Canonical Parquet outputs were generated under `processed/`.

## Outputs

- `underlying_daily.parquet`: normalized NIFTY history, log returns, and 20/60/252-day backward-looking annualized volatility.
- `option_eod_ce.parquet` and `option_eod_pe.parquet`: normalized option records with maturity state, moneyness, quality flags, and joined spot/volatility fields.
- `risk_free_monthly.parquet`: normalized RBI records with original source values and units.
- `market_snapshot_ce.parquet` and `market_snapshot_pe.parquet`: compact pricing-scenario tables.

## Safety decisions

- `Close` remains the market benchmark; missing LTP or settlement values are not substituted into it.
- The RBI file is parsed as pipe-delimited; its source value and `Indian rupee` label are preserved, while the approved policy converts percentage yields to decimal rates.
- RBI alignment uses the prior calendar month to avoid current-month look-ahead. Observations without a supported prior month remain uncalibrated.
- `eligible_for_pure_pde_pricing` requires active maturity, spot, realized volatility, and a supported prior-month rate.
- Expired and maturity-day contracts remain in canonical data and are classified rather than silently deleted.

## Next plan

1. Run the parameterized PINN on Kaggle GPU using the private calibrated dataset.
2. Evaluate the saved model on a chronological holdout against Black–Scholes and NSE `Close`.
3. Add volatility-window, liquidity, and carry/dividend diagnostics before selecting a market-facing model.
4. Compute PINN Greeks and daily delta-hedging results.
5. Only after those gates pass, expose inference through an API/frontend.
"""
    (REPORTS / "phase1_preprocessing_report.md").write_text(text, encoding="utf-8")


def main() -> None:
    underlying, underlying_summary = read_underlying()
    rbi, rbi_summary = read_rbi()
    option_summary, design = process_options(underlying, rbi)
    write_report(underlying_summary, rbi_summary, option_summary, design)
    print(json.dumps({"underlying": underlying_summary, "rbi": rbi_summary, "options": option_summary}, indent=2))


if __name__ == "__main__":
    main()
