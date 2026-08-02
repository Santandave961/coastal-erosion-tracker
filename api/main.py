"""
FastAPI serving layer for the Coastal Erosion & Mangrove Loss Tracker.

Endpoints:
  POST /score/zone   -> composite erosion/mangrove-loss risk score
  GET  /health        -> liveness check

Run locally:
  uvicorn api.main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import date

from src.risk_model import score_zone

app = FastAPI(
    title="Coastal Erosion & Mangrove Loss Tracker API",
    description="Satellite-based erosion and mangrove canopy loss scoring for Niger Delta coastal zones.",
    version="0.1.0",
)


class ZoneScoreRequest(BaseModel):
    zone_name: str = Field(..., example="Nembe Coastline")
    latitude: float = Field(..., example=4.5)
    longitude: float = Field(..., example=6.4)
    net_erosion_pct: float = Field(..., description="% shoreline retreat (land->water) between two dates")
    net_canopy_loss_pct: float = Field(..., description="% mangrove canopy lost between two dates")
    years_observed: float = Field(1.0, gt=0, description="Years between the two compared dates")
    assessment_date: date = Field(default_factory=date.today)


class ZoneScoreResponse(BaseModel):
    zone_name: str
    latitude: float
    longitude: float
    risk_score: float
    risk_level: str
    annualized_erosion_pct_per_year: float
    annualized_canopy_loss_pct_per_year: float
    assessment_date: date


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score/zone", response_model=ZoneScoreResponse)
def score_zone_endpoint(req: ZoneScoreRequest):
    try:
        result = score_zone(
            net_erosion_pct=req.net_erosion_pct,
            net_canopy_loss_pct=req.net_canopy_loss_pct,
            years_observed=req.years_observed,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return ZoneScoreResponse(
        zone_name=req.zone_name,
        latitude=req.latitude,
        longitude=req.longitude,
        risk_score=result["risk_score"],
        risk_level=result["risk_level"],
        annualized_erosion_pct_per_year=result["annualized_erosion_pct_per_year"],
        annualized_canopy_loss_pct_per_year=result["annualized_canopy_loss_pct_per_year"],
        assessment_date=req.assessment_date,
    )


@app.post("/score/batch")
def score_batch(requests: list[ZoneScoreRequest]):
    return [score_zone_endpoint(r) for r in requests]