"""Run a daily EOD PINN-delta versus Black–Scholes-delta hedge audit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.hedging.delta import daily_delta_hedge
from src.pinn.market import load_kaggle_market_checkpoint
from src.pinn.parametric import predict_parametric_delta
from src.quant.black_scholes import black_scholes_greeks


def main() -> None:
    option_type, expiry, strike = "CE", pd.Timestamp("2024-12-26").date(), 18000.0
    columns = ["valuation_date", "expiry", "strike", "spot", "close", "realized_vol_20d", "risk_free_rate_decimal"]
    frame = pd.read_parquet("processed/option_eod_ce.parquet", columns=columns)
    frame = frame[(frame["expiry"] == expiry) & (frame["strike"] == strike)].copy()
    frame = frame.dropna(subset=["spot", "close", "realized_vol_20d", "risk_free_rate_decimal"])
    frame = frame.rename(columns={"close": "market_price", "realized_vol_20d": "volatility"})
    if len(frame) < 100:
        raise RuntimeError("selected contract does not have enough observations")
    model, _ = load_kaggle_market_checkpoint()
    tau = (pd.Timestamp(expiry) - pd.to_datetime(frame["valuation_date"])).dt.days.to_numpy() / 365.0
    frame["pinn_delta"] = 0.0
    interior = tau > 0
    frame.loc[interior, "pinn_delta"] = predict_parametric_delta(
        model, frame.loc[interior, "spot"].to_numpy(), strike, tau[interior], tau[interior],
        frame.loc[interior, "volatility"].to_numpy(), frame.loc[interior, "risk_free_rate_decimal"].to_numpy(), option_type,
    )
    frame["bs_delta"] = [
        black_scholes_greeks(s, strike, max(t, 1 / 365), r, v, option_type)["delta"]
        for s, t, r, v in zip(frame["spot"], tau, frame["risk_free_rate_decimal"], frame["volatility"])
    ]
    pinn = daily_delta_hedge(frame, option_type=option_type, strike=strike, expiry=expiry, rate_column="risk_free_rate_decimal", volatility_column="volatility", delta_column="pinn_delta")
    bs = daily_delta_hedge(frame, option_type=option_type, strike=strike, expiry=expiry, rate_column="risk_free_rate_decimal", volatility_column="volatility", delta_column="bs_delta")
    result = {
        "contract": {"option_type": option_type, "expiry": str(expiry), "strike": strike},
        "rows": len(frame),
        "date_min": str(pd.to_datetime(frame["valuation_date"]).min().date()),
        "date_max": str(pd.to_datetime(frame["valuation_date"]).max().date()),
        "pinn_delta_vs_bs_mae": float(np.mean(np.abs(frame["pinn_delta"] - frame["bs_delta"]))),
        "pinn_hedge": {"final_error": float(pinn["hedge_error"].iloc[-1]), "mean_abs_error": float(np.mean(np.abs(pinn["hedge_error"]))), "error_std": float(pinn["hedge_error"].std(ddof=1))},
        "bs_hedge": {"final_error": float(bs["hedge_error"].iloc[-1]), "mean_abs_error": float(np.mean(np.abs(bs["hedge_error"]))), "error_std": float(bs["hedge_error"].std(ddof=1))},
        "completed_to_expiry": bool(pinn.attrs["completed_to_expiry"]),
        "limitations": pinn.attrs["limitations"],
    }
    Path("reports/delta_hedging_evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pinn.to_parquet("reports/pinn_delta_hedge_path.parquet", index=False)
    bs.to_parquet("reports/bs_delta_hedge_path.parquet", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
