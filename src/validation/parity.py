"""CE/PE parity diagnostics for market observations."""

from __future__ import annotations

import numpy as np


def put_call_parity_residual(call_price, put_price, spot, strike, maturity, rate):
    """Return C-P-(S-K exp(-rT)) for no-dividend European parity."""
    return np.asarray(call_price, dtype=float) - np.asarray(put_price, dtype=float) - (
        np.asarray(spot, dtype=float) - np.asarray(strike, dtype=float) * np.exp(-np.asarray(rate, dtype=float) * np.asarray(maturity, dtype=float))
    )
