"""Kaggle GPU training for the market-parameterized NIFTY PINN.

The option Close field is loaded only after training for evaluation. It is never
passed into the PDE, payoff, or boundary losses.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import ndtr
from torch import nn


INPUT_ROOT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
SEED = 7
STEPS = int(os.environ.get("PINN_STEPS", "12000"))
COLLOCATION = int(os.environ.get("PINN_COLLOCATION", "4096"))
BOUNDARY = int(os.environ.get("PINN_BOUNDARY", "1024"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ParametricPINN(nn.Module):
    def __init__(self, hidden: int = 128, depth: int = 5, s_max_multiple: float = 4.0, sigma_scale: float = 1.0, rate_scale: float = 0.1, maturity_scale: float = 5.0):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(6, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        layers.append(nn.Linear(hidden, 1))
        self.network = nn.Sequential(*layers)
        self.s_max_multiple = s_max_multiple
        self.sigma_scale = sigma_scale
        self.rate_scale = rate_scale
        self.maturity_scale = maturity_scale

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raw = self.network(inputs)
        x, y, _sigma, rate_norm, maturity_norm, code = [inputs[:, i:i + 1] for i in range(6)]
        rate = rate_norm * self.rate_scale
        maturity = maturity_norm * self.maturity_scale
        terminal = torch.where(code > 0, torch.relu(x - 1.0), torch.relu(1.0 - x))
        discount = torch.exp(-rate * maturity * y)
        boundary = torch.where(
            code > 0,
            (x / self.s_max_multiple) * (self.s_max_multiple - discount),
            (1.0 - x / self.s_max_multiple) * discount,
        )
        interior = x * (self.s_max_multiple - x) * raw
        return terminal + y * (boundary - terminal + interior)


def load_rows() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    candidates = list(INPUT_ROOT.rglob("market_snapshot_ce.parquet"))
    if not candidates:
        raise FileNotFoundError("Kaggle dataset attachment is missing market_snapshot_ce.parquet")
    input_dir = candidates[0].parent
    frames = []
    columns = [
        "valuation_date", "expiry", "strike", "spot", "time_to_maturity", "realized_vol_20d",
        "risk_free_rate_decimal", "option_type", "close", "eligible_for_pure_pde_pricing",
    ]
    for filename in ("market_snapshot_ce.parquet", "market_snapshot_pe.parquet"):
        frame = pd.read_parquet(input_dir / filename, columns=columns)
        frame = frame[frame["eligible_for_pure_pde_pricing"]].copy()
        frame["option_code"] = np.where(frame["option_type"].eq("CE"), 1.0, -1.0)
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    frame["valuation_date"] = pd.to_datetime(frame["valuation_date"])
    dates = np.sort(frame["valuation_date"].dt.date.unique())
    cutoff = str(dates[int(len(dates) * 0.8) - 1])
    train = frame[frame["valuation_date"].dt.date <= pd.Timestamp(cutoff).date()].copy()
    test = frame[frame["valuation_date"].dt.date > pd.Timestamp(cutoff).date()].copy()
    return train, test, cutoff


def pde_residual(model, x, y, sigma, rate, maturity, code):
    inputs = torch.cat([x, y, sigma / model.sigma_scale, rate / model.rate_scale, maturity / model.maturity_scale, code], dim=1).requires_grad_(True)
    value = model(inputs)
    gradient = torch.autograd.grad(value, inputs, torch.ones_like(value), create_graph=True)[0]
    ux, uy = gradient[:, :1], gradient[:, 1:2]
    second = torch.autograd.grad(ux, inputs, torch.ones_like(ux), create_graph=True)[0]
    uxx = second[:, :1]
    return uy / maturity - 0.5 * sigma**2 * x**2 * uxx - rate * x * ux + rate * value


def train(model: ParametricPINN, train_frame: pd.DataFrame) -> list[dict[str, float]]:
    rng = np.random.default_rng(SEED)
    market_arrays = {
        "sigma": train_frame["realized_vol_20d"].to_numpy(float),
        "rate": train_frame["risk_free_rate_decimal"].to_numpy(float),
        "maturity": train_frame["time_to_maturity"].to_numpy(float),
        "code": train_frame["option_code"].to_numpy(float),
    }
    n_market = len(train_frame)
    # Market rows calibrate the parameter distribution, while the synthetic
    # half of every batch prevents extrapolation outside the observed NSE
    # regime from being mistaken for a solved Black--Scholes PDE.
    synthetic_fraction = 0.5
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    for step in range(STEPS):
        n_synth = int(COLLOCATION * synthetic_fraction)
        n_market_batch = COLLOCATION - n_synth
        idx = rng.integers(0, n_market, size=n_market_batch)
        sigma_np = np.concatenate([
            market_arrays["sigma"][idx],
            rng.uniform(0.055, 0.9, size=n_synth),
        ])
        rate_np = np.concatenate([
            market_arrays["rate"][idx],
            rng.uniform(0.03, 0.07, size=n_synth),
        ])
        maturity_np = np.concatenate([
            market_arrays["maturity"][idx],
            rng.uniform(0.01, 5.0, size=n_synth),
        ])
        code_np = np.concatenate([
            market_arrays["code"][idx],
            np.where(rng.random(n_synth) >= 0.5, 1.0, -1.0),
        ])
        sigma = torch.tensor(sigma_np[:, None], dtype=torch.float32, device=DEVICE)
        rate = torch.tensor(rate_np[:, None], dtype=torch.float32, device=DEVICE)
        maturity = torch.tensor(maturity_np[:, None], dtype=torch.float32, device=DEVICE)
        code = torch.tensor(code_np[:, None], dtype=torch.float32, device=DEVICE)
        x = torch.rand(COLLOCATION, 1, device=DEVICE) * 4.0
        y = torch.rand(COLLOCATION, 1, device=DEVICE)
        pde_loss = torch.mean(pde_residual(model, x, y, sigma, rate, maturity, code) ** 2)

        n_synth_b = int(BOUNDARY * synthetic_fraction)
        n_market_b = BOUNDARY - n_synth_b
        bidx = rng.integers(0, n_market, size=n_market_b)
        sigma_b_np = np.concatenate([
            market_arrays["sigma"][bidx],
            rng.uniform(0.055, 0.9, size=n_synth_b),
        ])
        rate_b_np = np.concatenate([
            market_arrays["rate"][bidx],
            rng.uniform(0.03, 0.07, size=n_synth_b),
        ])
        maturity_b_np = np.concatenate([
            market_arrays["maturity"][bidx],
            rng.uniform(0.01, 5.0, size=n_synth_b),
        ])
        code_b_np = np.concatenate([
            market_arrays["code"][bidx],
            np.where(rng.random(n_synth_b) >= 0.5, 1.0, -1.0),
        ])
        sigma_b = torch.tensor(sigma_b_np[:, None], dtype=torch.float32, device=DEVICE)
        rate_b = torch.tensor(rate_b_np[:, None], dtype=torch.float32, device=DEVICE)
        maturity_b = torch.tensor(maturity_b_np[:, None], dtype=torch.float32, device=DEVICE)
        code_b = torch.tensor(code_b_np[:, None], dtype=torch.float32, device=DEVICE)
        xb = torch.rand(BOUNDARY, 1, device=DEVICE) * 4.0
        terminal = model(torch.cat([xb, torch.zeros_like(xb), sigma_b / model.sigma_scale, rate_b / model.rate_scale, maturity_b / model.maturity_scale, code_b], dim=1))
        terminal_target = torch.where(code_b > 0, torch.relu(xb - 1.0), torch.relu(1.0 - xb))
        terminal_loss = torch.mean((terminal - terminal_target) ** 2)
        yb = torch.rand(BOUNDARY, 1, device=DEVICE)
        low = model(torch.cat([torch.zeros_like(yb), yb, sigma_b / model.sigma_scale, rate_b / model.rate_scale, maturity_b / model.maturity_scale, code_b], dim=1))
        high = model(torch.cat([torch.full_like(yb, 4.0), yb, sigma_b / model.sigma_scale, rate_b / model.rate_scale, maturity_b / model.maturity_scale, code_b], dim=1))
        discount = torch.exp(-rate_b * maturity_b * yb)
        boundary_loss = torch.mean((low - torch.where(code_b > 0, torch.zeros_like(yb), discount)) ** 2)
        boundary_loss += torch.mean((high - torch.where(code_b > 0, 4.0 - discount, torch.zeros_like(yb))) ** 2)
        loss = pde_loss + 10.0 * terminal_loss + boundary_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % max(1, STEPS // 12) == 0:
            history.append({"step": step + 1, "loss": float(loss.detach().cpu()), "pde_loss": float(pde_loss.detach().cpu()), "terminal_loss": float(terminal_loss.detach().cpu()), "boundary_loss": float(boundary_loss.detach().cpu())})
    return history


def model_prices(model: ParametricPINN, frame: pd.DataFrame) -> np.ndarray:
    spot = frame["spot"].to_numpy(float)
    strike = frame["strike"].to_numpy(float)
    maturity = frame["time_to_maturity"].to_numpy(float)
    sigma = frame["realized_vol_20d"].to_numpy(float)
    rate = frame["risk_free_rate_decimal"].to_numpy(float)
    code = frame["option_code"].to_numpy(float)
    inputs = torch.tensor(np.column_stack([spot / strike, np.ones_like(maturity), sigma / model.sigma_scale, rate / model.rate_scale, maturity / model.maturity_scale, code]), dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        raw = strike * model(inputs).squeeze(1).cpu().numpy()
    discount = np.exp(-rate * maturity)
    call = code > 0
    lower = np.where(call, np.maximum(spot - strike * discount, 0), np.maximum(strike * discount - spot, 0))
    upper = np.where(call, spot, strike * discount)
    return np.clip(raw, lower, upper)


def black_scholes(frame: pd.DataFrame) -> np.ndarray:
    spot = frame["spot"].to_numpy(float)
    strike = frame["strike"].to_numpy(float)
    maturity = frame["time_to_maturity"].to_numpy(float)
    sigma = frame["realized_vol_20d"].to_numpy(float)
    rate = frame["risk_free_rate_decimal"].to_numpy(float)
    code = frame["option_code"].to_numpy(float)
    root = np.sqrt(maturity)
    d1 = (np.log(spot / strike) + (rate + 0.5 * sigma**2) * maturity) / (sigma * root)
    d2 = d1 - sigma * root
    call = spot * ndtr(d1) - strike * np.exp(-rate * maturity) * ndtr(d2)
    put = strike * np.exp(-rate * maturity) * ndtr(-d2) - spot * ndtr(-d1)
    return np.where(code > 0, call, put)


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    return {"n": int(len(actual)), "mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error**2))), "max_abs_error": float(np.max(np.abs(error)))}


def normalized_metrics(actual: np.ndarray, predicted: np.ndarray, strike: np.ndarray) -> dict[str, float]:
    error = (predicted - actual) / strike
    return {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error**2))), "max_abs_error": float(np.max(np.abs(error)))}


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    train_frame, test_frame, cutoff = load_rows()
    model = ParametricPINN().to(DEVICE)
    history = train(model, train_frame)
    test_frame = test_frame.copy()
    pinn = model_prices(model, test_frame)
    bs = black_scholes(test_frame)
    test_frame["pinn_price"] = pinn
    test_frame["black_scholes_price"] = bs
    market = test_frame["close"].notna().to_numpy()
    report = {
        "device": DEVICE,
        "train_rows": int(len(train_frame)),
        "test_rows": int(len(test_frame)),
        "train_cutoff": cutoff,
        "steps": STEPS,
        "pde_training_excludes_close": True,
        "training_parameter_distribution": {
            "market_fraction": 0.5,
            "synthetic_sigma": [0.055, 0.9],
            "synthetic_rate": [0.03, 0.07],
            "synthetic_maturity": [0.01, 5.0],
        },
        "pinn_vs_black_scholes": metrics(bs, pinn),
        "pinn_vs_black_scholes_normalized": normalized_metrics(bs, pinn, test_frame["strike"].to_numpy(float)),
        "black_scholes_vs_market_close": metrics(test_frame.loc[market, "close"].to_numpy(float), bs[market]) if market.any() else None,
        "pinn_vs_market_close": metrics(test_frame.loc[market, "close"].to_numpy(float), pinn[market]) if market.any() else None,
        "history": history,
    }
    torch.save({"model_state_dict": model.state_dict(), "metadata": report}, WORK / "market_parametric_pinn.pt")
    test_frame[["valuation_date", "expiry", "strike", "option_type", "spot", "time_to_maturity", "pinn_price", "black_scholes_price", "close"]].to_parquet(WORK / "market_parametric_predictions.parquet", index=False)
    (WORK / "market_parametric_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
