"""Black–Scholes implied-volatility inversion for market diagnostics."""

from __future__ import annotations

import math

from .black_scholes import _is_call, black_scholes_price


def implied_volatility(
    market_price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    option_type: str,
    *,
    lower: float = 1e-6,
    upper: float = 5.0,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> float:
    """Invert Black–Scholes by bisection; return NaN for impossible quotes."""
    call = _is_call(option_type)
    if not all(math.isfinite(v) for v in [market_price, spot, strike, maturity, rate]):
        return float("nan")
    if market_price < 0 or maturity <= 0 or spot <= 0 or strike <= 0:
        return float("nan")
    discounted_strike = strike * math.exp(-rate * maturity)
    intrinsic = max(spot - discounted_strike, 0.0) if call else max(discounted_strike - spot, 0.0)
    upper_bound = spot if call else discounted_strike
    if market_price < intrinsic - tolerance or market_price > upper_bound + tolerance:
        return float("nan")
    low, high = lower, upper
    f_low = black_scholes_price(spot, strike, maturity, rate, low, option_type) - market_price
    f_high = black_scholes_price(spot, strike, maturity, rate, high, option_type) - market_price
    if f_low * f_high > 0:
        return float("nan")
    for _ in range(max_iterations):
        mid = 0.5 * (low + high)
        f_mid = black_scholes_price(spot, strike, maturity, rate, mid, option_type) - market_price
        if abs(f_mid) <= tolerance:
            return float(mid)
        if f_low * f_mid <= 0:
            high, f_high = mid, f_mid
        else:
            low, f_low = mid, f_mid
    return float(0.5 * (low + high))
