"""Loader and inference helpers for the Kaggle-trained market PINN."""

from __future__ import annotations

from pathlib import Path

import torch

from .parametric import ParametricPINN, predict_parametric_price


DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[2] / "models" / "market_parametric_pinn_kaggle.pt"


def load_kaggle_market_checkpoint(path: str | Path = DEFAULT_CHECKPOINT):
    """Load the Kaggle v2 architecture (128 hidden units, depth 5)."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = ParametricPINN(hidden=128, depth=5)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload.get("metadata", {})
