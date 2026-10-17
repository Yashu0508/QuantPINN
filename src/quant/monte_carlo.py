"""Monte Carlo benchmark for European GBM option prices."""

from __future__ import annotations

import numpy as np

from .black_scholes import _is_call


def monte_carlo_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    option_type: str,
    paths: int = 100_000,
    seed: int = 7,
    antithetic: bool = True,
) -> dict[str, float]:
    """Price a European option by terminal GBM simulation and report standard error."""
    if paths < (4 if antithetic else 2):
        raise ValueError("at least two independent samples (four antithetic paths) are required")
    if antithetic and paths % 2:
        raise ValueError("antithetic paths must be even")
    if not np.isfinite([spot, strike, maturity, rate, volatility]).all():
        raise ValueError("Monte Carlo inputs must be finite")
    if spot <= 0 or strike <= 0 or maturity < 0 or volatility <= 0:
        raise ValueError("invalid Monte Carlo inputs")
    if maturity == 0:
        intrinsic = max(spot - strike, 0.0) if _is_call(option_type) else max(strike - spot, 0.0)
        return {"price": intrinsic, "standard_error": 0.0, "paths": 1.0}
    rng = np.random.default_rng(seed)
    n = paths // 2 if antithetic else paths
    z = rng.standard_normal(n)
    if antithetic:
        z = np.concatenate([z, -z])
    terminal = spot * np.exp((rate - 0.5 * volatility**2) * maturity + volatility * np.sqrt(maturity) * z)
    if _is_call(option_type):
        payoff = np.maximum(terminal - strike, 0.0)
    else:
        payoff = np.maximum(strike - terminal, 0.0)
    discounted = np.exp(-rate * maturity) * payoff
    # Paired payoffs are correlated. Estimate uncertainty from independent pair
    # averages rather than treating the two members of each pair as independent.
    samples = (discounted[:n] + discounted[n:]) / 2 if antithetic else discounted
    return {
        "price": float(discounted.mean()),
        "standard_error": float(samples.std(ddof=1) / np.sqrt(len(samples))),
        "paths": float(len(discounted)),
        "independent_samples": float(len(samples)),
    }
