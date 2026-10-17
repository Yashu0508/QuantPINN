"""Fixed-parameter Black–Scholes PINN baseline.

The network uses x=S/K and y=tau/T, where tau is time remaining to expiry.
The terminal/payoff condition is imposed at y=0. For this coordinate choice the
normalized PDE residual is u_y/T - .5*sigma^2*x^2*u_xx - r*x*u_x + r*u.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class FixedParameterPINN(nn.Module):
    def __init__(
        self,
        option_type: str = "CE",
        hidden: int = 64,
        depth: int = 3,
        hard_constraints: bool = True,
        maturity: float | None = None,
        rate: float | None = None,
        s_max_multiple: float = 4.0,
    ):
        super().__init__()
        option_type = option_type.upper()
        if option_type not in {"CE", "PE"}:
            raise ValueError("option_type must be CE or PE")
        self.option_type = option_type
        self.hard_constraints = hard_constraints
        self.maturity = maturity
        self.rate = rate
        self.s_max_multiple = s_max_multiple
        layers: list[nn.Module] = [nn.Linear(2, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        layers.append(nn.Linear(hidden, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.network(x)
        if not self.hard_constraints or self.maturity is None or self.rate is None:
            return raw
        spot_ratio, normalized_tau = x[:, :1], x[:, 1:2]
        terminal = _payoff(spot_ratio, self.option_type)
        discount = torch.exp(-self.rate * self.maturity * normalized_tau)
        if self.option_type == "CE":
            boundary = (spot_ratio / self.s_max_multiple) * (self.s_max_multiple - discount)
        else:
            boundary = (1.0 - spot_ratio / self.s_max_multiple) * discount
        interior = spot_ratio * (self.s_max_multiple - spot_ratio) * raw
        return terminal + normalized_tau * (boundary - terminal + interior)


def _payoff(x: torch.Tensor, option_type: str) -> torch.Tensor:
    return torch.relu(x - 1.0) if option_type == "CE" else torch.relu(1.0 - x)


def _pde_residual(model, x, y, maturity, rate, volatility):
    inputs = torch.cat([x, y], dim=1).requires_grad_(True)
    u = model(inputs)
    grad = torch.autograd.grad(u, inputs, torch.ones_like(u), create_graph=True)[0]
    ux, uy = grad[:, :1], grad[:, 1:2]
    second = torch.autograd.grad(ux, inputs, torch.ones_like(ux), create_graph=True)[0]
    uxx = second[:, :1]
    residual = uy / maturity - 0.5 * volatility**2 * x**2 * uxx - rate * x * ux + rate * u
    return residual


@dataclass
class PINNTrainingResult:
    model: FixedParameterPINN
    history: list[dict[str, float]]


def predict_price(model: FixedParameterPINN, spot, strike: float, tau, maturity: float) -> np.ndarray:
    """Evaluate a trained normalized PINN and return prices in currency units."""
    if strike <= 0 or maturity <= 0:
        raise ValueError("strike and maturity must be positive")
    spots = np.asarray(spot, dtype=float)
    taus = np.asarray(tau, dtype=float)
    spots, taus = np.broadcast_arrays(spots, taus)
    if np.any(spots < 0) or np.any(taus < 0) or np.any(taus > maturity):
        raise ValueError("spot must be nonnegative and tau must lie in [0, maturity]")
    device = next(model.parameters()).device
    inputs = torch.as_tensor(
        np.column_stack([spots.ravel() / strike, taus.ravel() / maturity]),
        dtype=torch.float32,
        device=device,
    )
    model.eval()
    with torch.no_grad():
        values = strike * model(inputs).squeeze(1).cpu().numpy()
    return values.reshape(spots.shape)


def no_arbitrage_projection(price, spot, strike: float, tau, rate: float, option_type: str) -> np.ndarray:
    """Project raw predictions onto elementary European no-arbitrage bounds."""
    value = np.asarray(price, dtype=float)
    spots, strikes, taus, rates = np.broadcast_arrays(
        np.asarray(spot, dtype=float), np.asarray(strike, dtype=float), np.asarray(tau, dtype=float), np.asarray(rate, dtype=float)
    )
    discount = np.exp(-rates * taus)
    if option_type.upper() == "CE":
        lower = np.maximum(spots - strikes * discount, 0.0)
        upper = spots
    elif option_type.upper() == "PE":
        lower = np.maximum(strikes * discount - spots, 0.0)
        upper = strikes * discount
    else:
        raise ValueError("option_type must be CE or PE")
    return np.clip(value, lower, upper)


def train_fixed_parameter_pinn(
    *,
    option_type: str = "CE",
    strike: float = 100.0,
    maturity: float = 0.25,
    rate: float = 0.05,
    volatility: float = 0.2,
    s_max_multiple: float = 4.0,
    collocation: int = 2048,
    boundary_points: int = 256,
    steps: int = 2_000,
    learning_rate: float = 1e-3,
    seed: int = 7,
    device: str = "cpu",
) -> PINNTrainingResult:
    """Train the controlled fixed-parameter PINN and return loss history."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = FixedParameterPINN(
        option_type=option_type,
        maturity=maturity,
        rate=rate,
        s_max_multiple=s_max_multiple,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    for step in range(steps):
        x = torch.rand(collocation, 1, device=device) * s_max_multiple
        y = torch.rand(collocation, 1, device=device)
        residual = _pde_residual(model, x, y, maturity, rate, volatility)
        pde_loss = torch.mean(residual**2)
        interior_pred = model(torch.cat([x, y], dim=1))
        positivity_loss = torch.mean(torch.relu(-interior_pred) ** 2)

        x_terminal = torch.rand(boundary_points, 1, device=device) * s_max_multiple
        y_terminal = torch.zeros_like(x_terminal)
        terminal_pred = model(torch.cat([x_terminal, y_terminal], dim=1))
        terminal_loss = torch.mean((terminal_pred - _payoff(x_terminal, model.option_type)) ** 2)

        x_low = torch.zeros(boundary_points, 1, device=device)
        y_boundary = torch.rand(boundary_points, 1, device=device)
        low_pred = model(torch.cat([x_low, y_boundary], dim=1))
        if model.option_type == "CE":
            low_target = torch.zeros_like(y_boundary)
        else:
            low_target = torch.exp(-rate * maturity * y_boundary)
        low_loss = torch.mean((low_pred - low_target) ** 2)

        x_high = torch.full((boundary_points, 1), s_max_multiple, device=device)
        high_pred = model(torch.cat([x_high, y_boundary], dim=1))
        if model.option_type == "CE":
            high_target = s_max_multiple - torch.exp(-rate * maturity * y_boundary)
        else:
            high_target = torch.zeros_like(y_boundary)
        high_loss = torch.mean((high_pred - high_target) ** 2)

        loss = pde_loss + 10.0 * terminal_loss + low_loss + high_loss + 10.0 * positivity_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % max(1, steps // 10) == 0:
            history.append({
                "step": float(step + 1),
                "loss": float(loss.detach().cpu()),
                "pde_loss": float(pde_loss.detach().cpu()),
                "terminal_loss": float(terminal_loss.detach().cpu()),
                "boundary_loss": float((low_loss + high_loss).detach().cpu()),
                "positivity_loss": float(positivity_loss.detach().cpu()),
            })
    return PINNTrainingResult(model=model, history=history)
