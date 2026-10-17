"""Dependency-free scenario interface for the classical pricing engine."""

from __future__ import annotations

from dataclasses import dataclass

from .black_scholes import black_scholes_greeks, black_scholes_price
from .monte_carlo import monte_carlo_price


@dataclass(frozen=True)
class PricingScenario:
    spot: float
    strike: float
    maturity: float
    rate: float
    volatility: float
    option_type: str


def price_scenario(scenario: PricingScenario, *, paths: int = 100_000, seed: int = 7) -> dict:
    """Return analytical price, Monte Carlo price/error, and Black–Scholes Greeks."""
    bs = black_scholes_price(
        scenario.spot, scenario.strike, scenario.maturity,
        scenario.rate, scenario.volatility, scenario.option_type,
    )
    mc = monte_carlo_price(
        scenario.spot, scenario.strike, scenario.maturity,
        scenario.rate, scenario.volatility, scenario.option_type,
        paths=paths, seed=seed,
    )
    return {
        "black_scholes_price": float(bs),
        "monte_carlo": mc,
        "greeks": black_scholes_greeks(
            scenario.spot, scenario.strike, scenario.maturity,
            scenario.rate, scenario.volatility, scenario.option_type,
        ) if scenario.maturity > 0 else None,
    }
