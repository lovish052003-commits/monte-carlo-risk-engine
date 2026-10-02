from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Optional
import os

from risk_engine import run_monte_carlo_engine

app = FastAPI(title="Monte Carlo Risk Simulator")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


class SimulationRequest(BaseModel):
    ticker: str = Field(
        default="AAPL",
        pattern=r"^[A-Za-z0-9.^=\-]{1,15}$",
        description="Stock ticker symbol"
    )
    portfolio_value: float = Field(default=100000.0, gt=0, description="Total portfolio investment amount")
    time_horizon: int = Field(default=252, ge=1, le=2520, description="Simulation trading days (1-Day Basel to multi-year)")
    currency: Optional[str] = Field(default=None, description="Optional investment currency override")
    random_seed: Optional[int] = Field(default=42, description="Random seed for simulation reproducibility")
    drift_mode: Optional[str] = Field(default="zero", description="Drift assumption: 'zero', 'risk_free', or 'historical'")
    confidence_level: Optional[int] = Field(default=95, description="Confidence level: 90, 95, or 99")
    rf_rate: Optional[float] = Field(default=None, ge=0.0, le=0.5, description="Optional risk-free rate override")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")


# Threadpool-offloaded synchronous endpoint
@app.post("/simulate")
def simulate(payload: SimulationRequest):
    try:
        result = run_monte_carlo_engine(
            ticker=payload.ticker,
            investment_amount=payload.portfolio_value,
            time_horizon=payload.time_horizon,
            simulations=10000,
            requested_currency=payload.currency,
            seed=payload.random_seed,
            drift_mode=payload.drift_mode or "zero",
            confidence_level=payload.confidence_level or 95,
            rf_rate=payload.rf_rate
        )
        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        raise HTTPException(status_code=500, detail=str(re))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation engine error: {str(e)}")
