import pytest

torch = pytest.importorskip("torch")
from src.api.app import PriceRequest, health, price


def test_health_does_not_train_on_request():
    assert health()["training_on_request"] is False
    assert health()["readiness"] == "research_only"
    assert health()["analytical_solver_gate_passed"] is False


@pytest.mark.skipif(not __import__('pathlib').Path('models/market_parametric_pinn_kaggle.pt').is_file(), reason='Private checkpoint not installed')
def test_price_endpoint_returns_projected_pin_and_classical_outputs():
    result = price(PriceRequest(spot=20000, strike=20000, tau=0.1, maturity=0.1, volatility=0.2, rate=0.05, option_type="CE", monte_carlo_paths=1000))
    assert result["pinn_price"] >= 0
    assert result["black_scholes_price"] > 0
    assert result["monte_carlo"]["price"] > 0


def test_price_endpoint_rejects_invalid_horizon():
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        price(PriceRequest(spot=20000, strike=20000, tau=0.2, maturity=0.1, volatility=0.2, rate=0.05, option_type="CE", monte_carlo_paths=1000))


def test_api_rejects_nonfinite_inputs():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        PriceRequest(spot=float("inf"), strike=100, tau=0.1, maturity=0.1,
                     volatility=0.2, rate=0.05, option_type="CE")

def test_domain_errors_are_client_errors():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        price(PriceRequest(spot=500, strike=100, tau=0.1, maturity=0.1,
                           volatility=0.2, rate=0.05, option_type="CE"))
    assert exc.value.status_code == 422

def test_missing_checkpoint_is_service_unavailable(monkeypatch):
    import src.api.app as module
    from fastapi import HTTPException
    def missing():
        raise FileNotFoundError("missing")
    monkeypatch.setattr(module, "_model", missing)
    with pytest.raises(HTTPException) as exc:
        price(PriceRequest(spot=100, strike=100, tau=0.1, maturity=0.1,
                           volatility=0.2, rate=0.05, option_type="CE"))
    assert exc.value.status_code == 503
