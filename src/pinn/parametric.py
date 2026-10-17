"""Parameterized PINN architecture for many normalized option scenarios."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class ParametricPINN(nn.Module):
    """Map (S/K, tau/T, sigma, r, T, option-code) to normalized V/K."""

    def __init__(self, hidden: int = 96, depth: int = 4, s_max_multiple: float = 4.0, sigma_scale: float = 1.0, rate_scale: float = 0.1, maturity_scale: float = 5.0):
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
        terminal = _payoff(x, code)
        discount = torch.exp(-rate * maturity * y)
        boundary = torch.where(
            code > 0,
            (x / self.s_max_multiple) * (self.s_max_multiple - discount),
            (1.0 - x / self.s_max_multiple) * discount,
        )
        interior = x * (self.s_max_multiple - x) * raw
        return terminal + y * (boundary - terminal + interior)


def option_code(option_type: str) -> float:
    value = option_type.upper()
    if value == "CE":
        return 1.0
    if value == "PE":
        return -1.0
    raise ValueError("option_type must be CE or PE")


def _payoff(x: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
    call = torch.relu(x - 1.0)
    put = torch.relu(1.0 - x)
    return torch.where(code > 0, call, put)


def _features(model, x, y, sigma, rate, maturity, code):
    return torch.cat([
        x, y, sigma / model.sigma_scale, rate / model.rate_scale,
        maturity / model.maturity_scale, code,
    ], dim=1)


def parametric_pde_residual(model, x, y, sigma, rate, maturity, code):
    """Return normalized Black–Scholes PDE residual in time-to-maturity coordinates."""
    inputs = _features(model, x, y, sigma, rate, maturity, code).requires_grad_(True)
    value = model(inputs)
    grad = torch.autograd.grad(value, inputs, torch.ones_like(value), create_graph=True)[0]
    ux, uy = grad[:, :1], grad[:, 1:2]
    second = torch.autograd.grad(ux, inputs, torch.ones_like(ux), create_graph=True)[0]
    uxx = second[:, :1]
    return uy / maturity - 0.5 * sigma**2 * x**2 * uxx - rate * x * ux + rate * value


@dataclass
class ParametricTrainingResult:
    model: ParametricPINN
    history: list[dict[str, float]]


def _validated_scenarios(model, spot, strike, tau, maturity, sigma, rate, option_type):
    spots, strikes, taus, maturities, sigmas, rates = np.broadcast_arrays(
        np.asarray(spot, dtype=float), np.asarray(strike, dtype=float), np.asarray(tau, dtype=float),
        np.asarray(maturity, dtype=float), np.asarray(sigma, dtype=float), np.asarray(rate, dtype=float),
    )
    code_value = option_code(option_type) if isinstance(option_type, str) else np.asarray(option_type, dtype=float)
    codes = np.broadcast_to(code_value, spots.shape)
    if any(not np.isfinite(a).all() for a in (spots, strikes, taus, maturities, sigmas, rates, codes)):
        raise ValueError("PINN inputs must be finite")
    if not np.isin(codes, [-1.0, 1.0]).all():
        raise ValueError("option codes must be +1 for CE or -1 for PE")
    if np.any(strikes <= 0) or np.any(spots < 0) or np.any(taus < 0) or np.any(taus > maturities):
        raise ValueError("invalid parameterized PINN inputs")
    if np.any(maturities <= 0) or np.any(sigmas <= 0):
        raise ValueError("maturity and volatility must be positive")
    if np.any(spots / strikes > model.s_max_multiple):
        raise ValueError("spot/strike exceeds the trained spatial domain")
    return spots, strikes, taus, maturities, sigmas, rates, codes


def predict_parametric_price(model: ParametricPINN, spot, strike, tau, maturity, sigma, rate, option_type):
    """Evaluate a parameterized PINN and return raw currency prices."""
    spots, strikes, taus, maturities, sigmas, rates, codes = _validated_scenarios(
        model, spot, strike, tau, maturity, sigma, rate, option_type
    )
    device = next(model.parameters()).device
    inputs = torch.as_tensor(
        np.column_stack([
            spots.ravel() / strikes.ravel(), taus.ravel() / maturities.ravel(),
            sigmas.ravel() / model.sigma_scale, rates.ravel() / model.rate_scale,
            maturities.ravel() / model.maturity_scale, codes.ravel(),
        ]), dtype=torch.float32, device=device,
    )
    model.eval()
    with torch.no_grad():
        normalized = model(inputs).squeeze(1).cpu().numpy()
    return (strikes.ravel() * normalized).reshape(spots.shape)


def predict_parametric_delta(model: ParametricPINN, spot, strike, tau, maturity, sigma, rate, option_type):
    """Return raw PINN delta dV/dS before any price projection."""
    spots, strikes, taus, maturities, sigmas, rates, codes = _validated_scenarios(
        model, spot, strike, tau, maturity, sigma, rate, option_type
    )
    device = next(model.parameters()).device
    x = torch.tensor((spots / strikes).ravel(), dtype=torch.float32, device=device).reshape(-1, 1).requires_grad_(True)
    inputs = _features(
        model, x, torch.tensor((taus / maturities).ravel(), dtype=torch.float32, device=device).reshape(-1, 1),
        torch.tensor(sigmas.ravel(), dtype=torch.float32, device=device).reshape(-1, 1),
        torch.tensor(rates.ravel(), dtype=torch.float32, device=device).reshape(-1, 1),
        torch.tensor(maturities.ravel(), dtype=torch.float32, device=device).reshape(-1, 1),
        torch.tensor(codes.ravel(), dtype=torch.float32, device=device).reshape(-1, 1),
    )
    value = model(inputs)
    delta = torch.autograd.grad(value, x, torch.ones_like(value), create_graph=False)[0]
    return delta.detach().cpu().numpy().reshape(spots.shape)


def predict_parametric_greeks(model, spot, strike, tau, maturity, sigma, rate, option_type):
    """Differentiate the raw network, holding its horizon normalization fixed.

    These are not derivatives of the clipped price. Vega and rho are per unit
    decimal volatility/rate; theta is per calendar year, with theta=-dV/dtau.
    """
    s, k, t, horizon, vol, r, code = _validated_scenarios(model, spot, strike, tau, maturity, sigma, rate, option_type)
    if (t <= 0).any():
        raise ValueError("Greeks require positive time remaining")
    device = next(model.parameters()).device
    inp = torch.tensor(np.column_stack([
        (s / k).ravel(), (t / horizon).ravel(), (vol / model.sigma_scale).ravel(),
        (r / model.rate_scale).ravel(), (horizon / model.maturity_scale).ravel(), code.ravel(),
    ]), dtype=next(model.parameters()).dtype, device=device, requires_grad=True)
    value = model(inp)
    first = torch.autograd.grad(value, inp, torch.ones_like(value), create_graph=True)[0]
    second = torch.autograd.grad(first[:, :1], inp, torch.ones_like(first[:, :1]))[0][:, 0]
    grad = first.detach().cpu().numpy()
    gamma = second.detach().cpu().numpy() / k.ravel()
    return {
        "delta": grad[:, 0].reshape(s.shape), "gamma": gamma.reshape(s.shape),
        "theta": (-k.ravel() * grad[:, 1] / horizon.ravel()).reshape(s.shape),
        "vega": (k.ravel() * grad[:, 2] / model.sigma_scale).reshape(s.shape),
        "rho": (k.ravel() * grad[:, 3] / model.rate_scale).reshape(s.shape),
    }


def train_parametric_pinn(
    *,
    steps: int = 2_000,
    collocation: int = 2_048,
    boundary_points: int = 256,
    seed: int = 7,
    device: str = "cpu",
) -> ParametricTrainingResult:
    """Train on a synthetic parameter distribution before market calibration."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ParametricPINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history: list[dict[str, float]] = []
    for step in range(steps):
        x = torch.rand(collocation, 1, device=device) * 4.0
        y = torch.rand(collocation, 1, device=device)
        sigma = 0.12 + 0.36 * torch.rand(collocation, 1, device=device)
        rate = 0.01 + 0.08 * torch.rand(collocation, 1, device=device)
        maturity = 0.05 + 1.0 * torch.rand(collocation, 1, device=device)
        code = torch.where(torch.rand(collocation, 1, device=device) > 0.5, 1.0, -1.0)
        residual = parametric_pde_residual(model, x, y, sigma, rate, maturity, code)
        pde_loss = torch.mean(residual**2)

        xb = torch.rand(boundary_points, 1, device=device) * 4.0
        y0 = torch.zeros_like(xb)
        sigma_b = 0.12 + 0.36 * torch.rand(boundary_points, 1, device=device)
        rate_b = 0.01 + 0.08 * torch.rand(boundary_points, 1, device=device)
        maturity_b = 0.05 + torch.rand(boundary_points, 1, device=device)
        code_b = torch.where(torch.rand(boundary_points, 1, device=device) > 0.5, 1.0, -1.0)
        terminal = model(_features(model, xb, y0, sigma_b, rate_b, maturity_b, code_b))
        terminal_loss = torch.mean((terminal - _payoff(xb, code_b)) ** 2)

        yb = torch.rand(boundary_points, 1, device=device)
        sigma_b = 0.12 + 0.36 * torch.rand(boundary_points, 1, device=device)
        rate_b = 0.01 + 0.08 * torch.rand(boundary_points, 1, device=device)
        maturity_b = 0.05 + torch.rand(boundary_points, 1, device=device)
        code_b = torch.where(torch.rand(boundary_points, 1, device=device) > 0.5, 1.0, -1.0)
        low = model(_features(model, torch.zeros_like(yb), yb, sigma_b, rate_b, maturity_b, code_b))
        high = model(_features(model, torch.full_like(yb, 4.0), yb, sigma_b, rate_b, maturity_b, code_b))
        discount = torch.exp(-rate_b * maturity_b * yb)
        low_target = torch.where(code_b > 0, torch.zeros_like(yb), discount)
        high_target = torch.where(code_b > 0, 4.0 - discount, torch.zeros_like(yb))
        boundary_loss = torch.mean((low - low_target) ** 2) + torch.mean((high - high_target) ** 2)

        loss = pde_loss + 10.0 * terminal_loss + boundary_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % max(1, steps // 10) == 0:
            history.append({
                "step": float(step + 1),
                "loss": float(loss.detach().cpu()),
                "pde_loss": float(pde_loss.detach().cpu()),
                "terminal_loss": float(terminal_loss.detach().cpu()),
                "boundary_loss": float(boundary_loss.detach().cpu()),
            })
    return ParametricTrainingResult(model=model, history=history)


def train_parametric_pinn_from_arrays(
    parameters: dict[str, np.ndarray],
    *,
    steps: int = 2_000,
    collocation: int = 2_048,
    boundary_points: int = 256,
    seed: int = 7,
    device: str = "cpu",
) -> ParametricTrainingResult:
    """Train only on calibrated parameter rows; market prices are never read."""
    required = {"maturity", "sigma", "rate", "option_code"}
    if required.difference(parameters):
        raise ValueError(f"missing parameter arrays: {sorted(required.difference(parameters))}")
    arrays = {key: np.asarray(value, dtype=float) for key, value in parameters.items()}
    n = len(arrays["maturity"])
    if n == 0 or any(len(value) != n for value in arrays.values()):
        raise ValueError("parameter arrays must be non-empty and equally sized")
    if np.any(arrays["maturity"] <= 0) or np.any(arrays["sigma"] <= 0):
        raise ValueError("maturity and sigma must be positive")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = ParametricPINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history: list[dict[str, float]] = []
    for step in range(steps):
        indices = rng.integers(0, n, size=collocation)
        sigma = torch.as_tensor(arrays["sigma"][indices, None], dtype=torch.float32, device=device)
        rate = torch.as_tensor(arrays["rate"][indices, None], dtype=torch.float32, device=device)
        maturity = torch.as_tensor(arrays["maturity"][indices, None], dtype=torch.float32, device=device)
        code = torch.as_tensor(arrays["option_code"][indices, None], dtype=torch.float32, device=device)
        x = torch.rand(collocation, 1, device=device) * 4.0
        y = torch.rand(collocation, 1, device=device)
        residual = parametric_pde_residual(model, x, y, sigma, rate, maturity, code)
        pde_loss = torch.mean(residual**2)

        bidx = rng.integers(0, n, size=boundary_points)
        sigma_b = torch.as_tensor(arrays["sigma"][bidx, None], dtype=torch.float32, device=device)
        rate_b = torch.as_tensor(arrays["rate"][bidx, None], dtype=torch.float32, device=device)
        maturity_b = torch.as_tensor(arrays["maturity"][bidx, None], dtype=torch.float32, device=device)
        code_b = torch.as_tensor(arrays["option_code"][bidx, None], dtype=torch.float32, device=device)
        xb = torch.rand(boundary_points, 1, device=device) * 4.0
        terminal = model(_features(model, xb, torch.zeros_like(xb), sigma_b, rate_b, maturity_b, code_b))
        terminal_loss = torch.mean((terminal - _payoff(xb, code_b)) ** 2)
        yb = torch.rand(boundary_points, 1, device=device)
        low = model(_features(model, torch.zeros_like(yb), yb, sigma_b, rate_b, maturity_b, code_b))
        high = model(_features(model, torch.full_like(yb, 4.0), yb, sigma_b, rate_b, maturity_b, code_b))
        discount = torch.exp(-rate_b * maturity_b * yb)
        boundary_loss = torch.mean((low - torch.where(code_b > 0, torch.zeros_like(yb), discount)) ** 2)
        boundary_loss += torch.mean((high - torch.where(code_b > 0, 4.0 - discount, torch.zeros_like(yb))) ** 2)
        loss = pde_loss + 10.0 * terminal_loss + boundary_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % max(1, steps // 10) == 0:
            history.append({"step": float(step + 1), "loss": float(loss.detach().cpu()), "pde_loss": float(pde_loss.detach().cpu()), "terminal_loss": float(terminal_loss.detach().cpu()), "boundary_loss": float(boundary_loss.detach().cpu())})
    return ParametricTrainingResult(model=model, history=history)
