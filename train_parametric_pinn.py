"""Train the parameterized PINN on a synthetic parameter distribution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.pinn.parametric import train_parametric_pinn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2_000)
    parser.add_argument("--collocation", type=int, default=2_048)
    parser.add_argument("--boundary-points", type=int, default=256)
    parser.add_argument("--output", type=Path, default=Path("models/parametric_pinn.pt"))
    args = parser.parse_args()
    result = train_parametric_pinn(steps=args.steps, collocation=args.collocation, boundary_points=args.boundary_points)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": result.model.state_dict(), "history": result.history}, args.output)
    print(json.dumps({"checkpoint": str(args.output), "final": result.history[-1]}, indent=2))


if __name__ == "__main__":
    main()
