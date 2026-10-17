"""Vectorized Black–Scholes pricing and Greeks for European options."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.special import ndtr

OptionType = Literal["CE", "PE", "call", "put"]


def _is_call(option_type: OptionType) -> bool:
    value = str(option_type).upper()
    if value in {"CE", "CALL"}:
        return True
    if value in {"PE", "PUT"}:
        return False
    raise ValueError(f"unsupported option type: {option_type}")


def _inputs(spot, strike, maturity, rate, volatility):
    s, k, t, r, sigma = np.broadcast_arrays(
        np.asarray(spot, dtype=float),
        np.asarray(strike, dtype=float),
        np.asarray(maturity, dtype=float),
        np.asarray(rate, dtype=float),
        np.asarray(volatility, dtype=float),
    )
    if any(not np.isfinite(a).all() for a in (s, k, t, r, sigma)):
        raise ValueError("pricing inputs must be finite")
    if np.any(s <= 0) or np.any(k <= 0):
        raise ValueError("spot and strike must be positive")
    if np.any(t < 0):
        raise ValueError("maturity cannot be negative")
    if np.any(sigma <= 0):
        raise ValueError("volatility must be positive")
    return s, k, t, r, sigma


def black_scholes_price(spot, strike, maturity, rate, volatility, option_type: OptionType):
    """Return the no-dividend Black–Scholes price.

    Scalars return a Python float; arrays preserve the broadcast shape. At zero
    maturity the intrinsic value is returned exactly.
    """
    s, k, t, r, sigma = _inputs(spot, strike, maturity, rate, volatility)
    call = _is_call(option_type)
    out = np.empty_like(s, dtype=float)
    zero = t == 0
    if np.any(~zero):
        sv, kv, tv, rv, sigv = s[~zero], k[~zero], t[~zero], r[~zero], sigma[~zero]
        root_t = np.sqrt(tv)
        d1 = (np.log(sv / kv) + (rv + 0.5 * sigv**2) * tv) / (sigv * root_t)
        d2 = d1 - sigv * root_t
        if call:
            out[~zero] = sv * ndtr(d1) - kv * np.exp(-rv * tv) * ndtr(d2)
        else:
            out[~zero] = kv * np.exp(-rv * tv) * ndtr(-d2) - sv * ndtr(-d1)
    if np.any(zero):
        out[zero] = np.maximum(s[zero] - k[zero], 0) if call else np.maximum(k[zero] - s[zero], 0)
    return float(out) if out.ndim == 0 else out


def black_scholes_greeks(spot, strike, maturity, rate, volatility, option_type: OptionType):
    """Return delta, gamma, theta, vega, and rho per one unit of underlying."""
    s, k, t, r, sigma = _inputs(spot, strike, maturity, rate, volatility)
    if np.any(t <= 0):
        raise ValueError("Greeks require strictly positive maturity")
    call = _is_call(option_type)
    root_t = np.sqrt(t)
    d1 = (np.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * root_t)
    d2 = d1 - sigma * root_t
    pdf = np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi)
    discount = np.exp(-r * t)
    delta = ndtr(d1) if call else ndtr(d1) - 1
    gamma = pdf / (s * sigma * root_t)
    vega = s * pdf * root_t
    if call:
        theta = -(s * pdf * sigma) / (2 * root_t) - r * k * discount * ndtr(d2)
        rho = k * t * discount * ndtr(d2)
    else:
        theta = -(s * pdf * sigma) / (2 * root_t) + r * k * discount * ndtr(-d2)
        rho = -k * t * discount * ndtr(-d2)
    result = {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega, "rho": rho}
    if all(np.asarray(v).ndim == 0 for v in result.values()):
        return {k_: float(v) for k_, v in result.items()}
    return result
