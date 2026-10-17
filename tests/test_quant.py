import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.quant.black_scholes import black_scholes_greeks, black_scholes_price
from src.quant.monte_carlo import monte_carlo_price
from src.quant.engine import PricingScenario, price_scenario
from src.quant.implied_vol import implied_volatility
from src.validation.metrics import regression_metrics

try:
    import torch
    from src.pinn.fixed_parameter import FixedParameterPINN, no_arbitrage_projection
    from src.pinn.parametric import ParametricPINN
except ModuleNotFoundError:
    torch = None
    train_fixed_parameter_pinn = None
    train_parametric_pinn = None


def test_black_scholes_known_call():
    value = black_scholes_price(100, 100, 1, 0.05, 0.2, "CE")
    assert abs(value - 10.4505835722) < 1e-8


def test_put_call_parity():
    call = black_scholes_price(100, 105, 0.5, 0.03, 0.25, "CE")
    put = black_scholes_price(100, 105, 0.5, 0.03, 0.25, "PE")
    assert abs(call - put - (100 - 105 * np.exp(-0.03 * 0.5))) < 1e-10


def test_greeks_are_finite_and_delta_has_expected_sign():
    call = black_scholes_greeks(100, 100, 1, 0.05, 0.2, "CE")
    put = black_scholes_greeks(100, 100, 1, 0.05, 0.2, "PE")
    assert all(np.isfinite(list(call.values())))
    assert 0 < call["delta"] < 1
    assert -1 < put["delta"] < 0
    assert call["gamma"] > 0 and call["vega"] > 0


def test_monte_carlo_is_within_four_standard_errors():
    exact = black_scholes_price(100, 100, 1, 0.05, 0.2, "CE")
    result = monte_carlo_price(100, 100, 1, 0.05, 0.2, "CE", paths=200_000, seed=11)
    assert abs(result["price"] - exact) <= 4 * result["standard_error"]


def test_regression_metrics_are_consistent():
    metrics = regression_metrics([1, 2, 3], [1, 3, 2])
    assert metrics["mae"] == 2 / 3
    assert metrics["n"] == 3


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable in this test runtime")
def test_no_arbitrage_projection_enforces_bounds():
    projected = no_arbitrage_projection([-10.0, 500.0], [100.0, 100.0], 100.0, [0.25, 0.25], 0.05, "CE")
    assert np.all(projected >= 0)
    assert projected[1] <= 100


def test_pricing_engine_returns_all_classical_outputs():
    result = price_scenario(PricingScenario(100, 100, 0.25, 0.05, 0.2, "CE"), paths=10_000)
    assert result["black_scholes_price"] > 0
    assert result["monte_carlo"]["price"] > 0
    assert "delta" in result["greeks"]


def test_implied_volatility_recovers_input():
    price = black_scholes_price(100, 100, 0.5, 0.04, 0.3, "PE")
    recovered = implied_volatility(price, 100, 100, 0.5, 0.04, "PE")
    assert abs(recovered - 0.3) < 1e-6


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable in this test runtime")
def test_fixed_pinn_terminal_payoff_without_training():
    model = FixedParameterPINN("CE", maturity=0.25, rate=0.05)
    inputs = torch.tensor([[0.8, 0.0], [1.2, 0.0]])
    assert torch.allclose(model(inputs), torch.tensor([[0.0], [0.2]]), atol=1e-7)


@pytest.mark.skipif(torch is None, reason="PyTorch is unavailable in this test runtime")
def test_parametric_pinn_terminal_payoff_without_training():
    model = ParametricPINN()
    inputs = torch.tensor([[0.8, 0.0, 0.2, 0.5, 0.05, -1.0], [1.2, 0.0, 0.2, 0.5, 0.05, 1.0]])
    assert torch.allclose(model(inputs), torch.tensor([[0.2], [0.2]]), atol=1e-7)


def test_itm_put_iv_below_undiscounted_intrinsic_is_valid():
    price = black_scholes_price(100, 120, 1, 0.1, 0.1, "PE")
    assert price < 20
    assert abs(implied_volatility(price, 100, 120, 1, 0.1, "PE") - 0.1) < 1e-6


def test_antithetic_standard_error_from_pair_averages():
    n = 2000
    z = np.random.default_rng(3).standard_normal(n // 2)
    terminal = 100 * np.exp((0.05 - 0.5 * 0.2**2) + 0.2 * np.r_[z, -z])
    pay = np.exp(-0.05) * np.maximum(terminal - 100, 0)
    pairs = (pay[:n // 2] + pay[n // 2:]) / 2
    result = monte_carlo_price(100, 100, 1, 0.05, 0.2, "CE", paths=n, seed=3)
    assert result["standard_error"] == pytest.approx(pairs.std(ddof=1) / np.sqrt(n // 2))
