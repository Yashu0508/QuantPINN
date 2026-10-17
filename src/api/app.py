"""FastAPI inference boundary for the Kaggle-trained PINN candidate."""

from __future__ import annotations

from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.pinn.fixed_parameter import no_arbitrage_projection
from src.pinn.market import DEFAULT_CHECKPOINT, load_kaggle_market_checkpoint
from src.pinn.parametric import predict_parametric_delta, predict_parametric_price
from src.quant.black_scholes import black_scholes_greeks, black_scholes_price
from src.quant.monte_carlo import monte_carlo_price
from src.validation.gates import require_solver_gate


class PriceRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    spot: float = Field(gt=0)
    strike: float = Field(gt=0)
    tau: float = Field(ge=0, description="time remaining to expiry in years")
    maturity: float = Field(gt=0, description="scenario maturity scale in years")
    volatility: float = Field(gt=0)
    rate: float
    option_type: Literal["CE", "PE"]
    monte_carlo_paths: int = Field(default=20_000, ge=4, le=1_000_000)


@lru_cache(maxsize=1)
def _model():
    return load_kaggle_market_checkpoint()[0]


app = FastAPI(title="NIFTY PINN Pricing API", version="research-kaggle-v2")


@app.get("/health")
def health() -> dict:
    solver_passed = False
    try:
        require_solver_gate(checkpoint=DEFAULT_CHECKPOINT)
        solver_passed = True
    except (OSError, ValueError, RuntimeError):
        pass
    return {
        "status": "ok", "model": "kaggle-market-parametric-pinn-v2",
        "training_on_request": False,
        "checkpoint_available": DEFAULT_CHECKPOINT.is_file(),
        "readiness": "solver_accepted_candidate" if solver_passed else "research_only",
        "analytical_solver_gate_passed": solver_passed,
    }


@app.post("/price")
def price(request: PriceRequest) -> dict:
    if request.tau > request.maturity:
        raise HTTPException(status_code=422, detail="tau cannot exceed maturity")
    if not math.isfinite(request.rate) or request.monte_carlo_paths % 2:
        raise HTTPException(status_code=422, detail="rate must be finite and Monte Carlo paths must be even")
    if request.spot / request.strike > 4:
        raise HTTPException(status_code=422, detail="spot/strike exceeds PINN spatial domain (maximum 4)")
    try:
        model = _model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Research checkpoint is absent; see models/README.md") from exc
    pinn_raw = float(predict_parametric_price(model, request.spot, request.strike, request.tau, request.maturity, request.volatility, request.rate, request.option_type))
    pinn = float(no_arbitrage_projection(pinn_raw, request.spot, request.strike, request.tau, request.rate, request.option_type))
    bs = float(black_scholes_price(request.spot, request.strike, request.tau, request.rate, request.volatility, request.option_type))
    mc = monte_carlo_price(request.spot, request.strike, request.tau, request.rate, request.volatility, request.option_type, paths=request.monte_carlo_paths)
    greeks = None if request.tau == 0 else black_scholes_greeks(request.spot, request.strike, request.tau, request.rate, request.volatility, request.option_type)
    pinn_delta = None if request.tau == 0 else float(predict_parametric_delta(model, request.spot, request.strike, request.tau, request.maturity, request.volatility, request.rate, request.option_type))
    return {
        "pinn_price": pinn,
        "pinn_raw_price": pinn_raw,
        "black_scholes_price": bs,
        "monte_carlo": mc,
        "black_scholes_greeks": greeks,
        "pinn_delta_raw": pinn_delta,
        "market_close": None,
        "status": "research_inference",
        "limitations": "Unvalidated PINN candidate; projected price and raw delta are different quantities.",
    }
