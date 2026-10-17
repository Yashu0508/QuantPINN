import numpy as np
import pandas as pd
import pytest

from src.hedging.delta import daily_delta_hedge


def test_flat_zero_rate_funded_hedge_has_zero_pnl():
    data = pd.DataFrame({"valuation_date": pd.date_range("2020-01-01", periods=3),
                         "spot": [100.0] * 3, "market_price": [5.0] * 3, "volatility": [0.2] * 3})
    out = daily_delta_hedge(data, option_type="CE", strike=100, expiry="2020-02-01", rate=0)
    assert np.allclose(out.hedge_portfolio_value, 0)
    assert np.allclose(out.hedge_error, 0)
    assert np.allclose(out.market_price + out.hedge_shares * out.spot + out.cash_balance, 0)


def test_terminal_hedge_uses_payoff_and_liquidates_shares():
    data = pd.DataFrame({"valuation_date": ["2020-01-01", "2020-01-02"],
                         "spot": [100., 105.], "market_price": [5., 999.], "volatility": [0.2, 0.2],
                         "delta": [0.5, 0.0]})
    out = daily_delta_hedge(data, option_type="CE", strike=100, expiry="2020-01-02", rate=0, delta_column="delta")
    assert out.hedge_error.iloc[-1] == pytest.approx(-2.5)
    assert out.hedge_shares.iloc[-1] == 0
    assert out.attrs["completed_to_expiry"]
