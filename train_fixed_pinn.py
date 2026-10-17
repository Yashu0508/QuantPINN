"""Train a controlled fixed-parameter PINN and save its checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.pinn.fixed_parameter import train_fixed_parameter_pinn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--option-type", choices=["CE", "PE"], default="CE")
    parser.add_argument("--strike", type=float, default=100.0)
    parser.add_argument("--maturity", type=float, default=0.25)
    parser.add_argument("--rate", type=float, default=0.05)
    parser.add_argument("--volatility", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--collocation", type=int, default=2_048)
    parser.add_argument("--boundary-points", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("models/fixed_parameter_pinn.pt"))
    args = parser.parse_args()
    result = train_fixed_parameter_pinn(
        option_type=args.option_type,
        strike=args.strike,
        maturity=args.maturity,
        rate=args.rate,
        volatility=args.volatility,
        steps=args.steps,
        collocation=args.collocation,
        boundary_points=args.boundary_points,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": result.model.state_dict(),
            "option_type": args.option_type,
            "strike": args.strike,
            "maturity": args.maturity,
            "rate": args.rate,
            "volatility": args.volatility,
            "history": result.history,
        },
        args.output,
    )
    print(json.dumps({"checkpoint": str(args.output), "final": result.history[-1]}, indent=2))


if __name__ == "__main__":
    main()
