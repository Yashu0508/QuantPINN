import pytest

from src.calibration.rates import load_rate_policy, source_value_to_decimal
from src.validation.parity import put_call_parity_residual


def test_rate_policy_fails_closed():
    policy = {"approved": False, "decimal_scale": 0.01, "as_of_alignment": "prior_calendar_month", "day_count": 365}
    with pytest.raises(RuntimeError):
        source_value_to_decimal(5.0, policy)


def test_approved_rate_policy_conversion():
    policy = {"approved": True, "decimal_scale": 0.01, "as_of_alignment": "prior_calendar_month", "day_count": 365}
    assert source_value_to_decimal(5.0, policy) == 0.05


def test_project_policy_is_approved_and_loadable():
    policy = load_rate_policy()
    assert policy["approved"] is True
    assert policy["decimal_scale"] == 0.01


def test_put_call_parity_residual_zero_for_parity_quote():
    import numpy as np
    spot, strike, maturity, rate = 100.0, 105.0, 0.5, 0.03
    put = 2.0
    call = put + spot - strike * np.exp(-rate * maturity)
    assert abs(put_call_parity_residual(call, put, spot, strike, maturity, rate)) < 1e-12
