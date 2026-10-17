"""Train a parameterized PINN from calibrated market parameters only.

Observed option prices are intentionally excluded from this training path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.pinn.parametric import train_parametric_pinn_from_arrays


def load_parameters() -> dict[str, np.ndarray]:
    frames = []
    cols = ["valuation_date", "time_to_maturity", "realized_vol_20d", "risk_free_rate_decimal", "option_type", "eligible_for_pure_pde_pricing"]
    for option_type in ("ce", "pe"):
        frame = pd.read_parquet(f"processed/market_snapshot_{option_type}.parquet", columns=cols)
        frame = frame[frame["eligible_for_pure_pde_pricing"]].copy()
        frame["option_code"] = np.where(frame["option_type"].eq("CE"), 1.0, -1.0)
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True).sort_values("valuation_date")
    # A chronological training slice prevents future parameter regimes from leaking backwards.
    cutoff = frame["valuation_date"].quantile(0.8)
    frame = frame[frame["valuation_date"] <= cutoff]
    return {
        "maturity": frame["time_to_maturity"].to_numpy(float),
        "sigma": frame["realized_vol_20d"].to_numpy(float),
        "rate": frame["risk_free_rate_decimal"].to_numpy(float),
        "option_code": frame["option_code"].to_numpy(float),
    }, {"rows": int(len(frame)), "train_cutoff": str(cutoff)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--collocation", type=int, default=512)
    parser.add_argument("--boundary-points", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("models/market_parametric_pinn.pt"))
    args = parser.parse_args()
    parameters, metadata = load_parameters()
    result = train_parametric_pinn_from_arrays(parameters, steps=args.steps, collocation=args.collocation, boundary_points=args.boundary_points)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": result.model.state_dict(), "history": result.history, "metadata": metadata}, args.output)
    print(json.dumps({"checkpoint": str(args.output), "metadata": metadata, "final": result.history[-1]}, indent=2))


if __name__ == "__main__":
    main()
