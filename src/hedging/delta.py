"""Daily EOD delta-hedging simulator for one historical option contract."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.quant.black_scholes import black_scholes_greeks


def daily_delta_hedge(
    observations: pd.DataFrame,
    *,
    option_type: str,
    strike: float,
    expiry: str | pd.Timestamp,
    rate: float | None = None,
    rate_column: str | None = None,
    volatility_column: str = "volatility",
    delta_column: str | None = None,
) -> pd.DataFrame:
    """Simulate a long-option, short-delta, daily EOD hedge."""
    if rate is None and rate_column is None:
        raise ValueError("provide either a scalar rate or rate_column")
    if rate is not None and rate_column is not None:
        raise ValueError("provide only one rate source")
    required = {"valuation_date", "spot", "market_price", volatility_column}
    if rate_column is not None:
        required.add(rate_column)
    if delta_column is not None:
        required.add(delta_column)
    missing = required.difference(observations.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if option_type not in {"CE", "PE"} or strike <= 0 or (rate is not None and not np.isfinite(rate)):
        raise ValueError("invalid contract or financing rate")
    expiry = pd.Timestamp(expiry)
    frame = observations.copy()
    frame["valuation_date"] = pd.to_datetime(frame["valuation_date"])
    frame = frame.sort_values("valuation_date").reset_index(drop=True)
    frame["tau"] = (expiry - frame["valuation_date"]).dt.days / 365.0
    if frame.empty or len(frame) < 2:
        raise ValueError("at least two observations are required")
    if frame["valuation_date"].duplicated().any() or frame["valuation_date"].isna().any():
        raise ValueError("valuation dates must be unique and valid")
    if (frame["tau"] < 0).any():
        raise ValueError("observations after expiry are not supported")

    rate_values = np.full(len(frame), float(rate)) if rate is not None else pd.to_numeric(frame[rate_column], errors="coerce").to_numpy(float)
    numeric_columns = ["spot", "market_price", volatility_column]
    if rate_column is not None:
        numeric_columns.append(rate_column)
    if not np.isfinite(frame[numeric_columns].to_numpy(float)).all() or not frame["spot"].gt(0).all() or not frame[volatility_column].gt(0).all():
        raise ValueError("hedging inputs must be finite with positive spot and volatility")
    if not np.isfinite(rate_values).all():
        raise ValueError("hedge rates must be finite")
    frame["hedge_rate"] = rate_values

    deltas = []
    for i, row in enumerate(frame.itertuples(index=False)):
        if row.tau == 0:
            delta = 0.0
        elif delta_column is not None:
            delta = float(frame.loc[i, delta_column])
            if not np.isfinite(delta):
                raise ValueError("external deltas must be finite")
        else:
            delta = black_scholes_greeks(row.spot, strike, row.tau, rate_values[i], getattr(row, volatility_column), option_type)["delta"]
        deltas.append(delta)
    frame["model_delta"] = deltas

    shares = -float(frame.loc[0, "model_delta"])
    cash = -float(frame.loc[0, "market_price"]) - shares * float(frame.loc[0, "spot"])
    values = [float(frame.loc[0, "market_price"]) + shares * float(frame.loc[0, "spot"]) + cash]
    hedge_error = [values[0]]
    cash_values = [cash]
    share_values = [shares]
    for i in range(1, len(frame)):
        days = (frame.loc[i, "valuation_date"] - frame.loc[i - 1, "valuation_date"]).days
        cash *= float(np.exp(rate_values[i - 1] * max(days, 0) / 365.0))
        spot = float(frame.loc[i, "spot"])
        new_shares = -float(frame.loc[i, "model_delta"])
        cash -= (new_shares - shares) * spot
        shares = new_shares
        option_value = float(frame.loc[i, "market_price"])
        if frame.loc[i, "tau"] == 0:
            option_value = max(spot - strike, 0.0) if option_type == "CE" else max(strike - spot, 0.0)
        value = option_value + shares * spot + cash
        values.append(value)
        hedge_error.append(value)
        cash_values.append(cash)
        share_values.append(shares)
    frame["hedge_portfolio_value"] = values
    frame["hedge_error"] = hedge_error
    frame["cash_balance"] = cash_values
    frame["hedge_shares"] = share_values
    frame.attrs["completed_to_expiry"] = bool(frame.loc[len(frame) - 1, "tau"] == 0)
    frame.attrs["limitations"] = "Index units are idealized tradable shares; excludes costs, futures basis, and dividends."
    return frame
