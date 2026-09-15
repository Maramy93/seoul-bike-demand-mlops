from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from pathlib import Path

from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from src.metrics import (
    ACTIVE_REQUESTS,
    PREDICTION_COUNT,
    PREDICTION_LATENCY,
    PREDICTION_VALUE,
    REQUEST_COUNT,
    REQUEST_LATENCY,
)
from src.train import ALL_FEATURES

DRIFT_REPORT_PATH = Path(
    "monitoring/evidently/reports/seoul_bike_drift_report.html"
)

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "bike_demand_pipeline.joblib"
)

model: Any | None = None


def get_model() -> Any:
    global model

    if model is None:
        model = joblib.load(MODEL_PATH)

    return model


app = FastAPI(
    title="Seoul Bike Demand API",
    description="Predicts hourly bicycle rental demand in Seoul.",
    version="1.0.0",
)


class BikeDemandRequest(BaseModel):
    date: date
    hour: int = Field(ge=0, le=23)
    temperature: float
    humidity: float = Field(ge=0, le=100)
    wind_speed: float = Field(ge=0)
    visibility: float = Field(ge=0)
    dew_point_temperature: float
    solar_radiation: float = Field(ge=0)
    rainfall: float = Field(ge=0)
    snowfall: float = Field(ge=0)
    season: Literal["Spring", "Summer", "Autumn", "Winter"]
    holiday: Literal["Holiday", "No Holiday"]
    functioning_day: Literal["Yes", "No"]


class BikeDemandResponse(BaseModel):
    predicted_bike_count: int


@app.middleware("http")
async def prometheus_middleware(
    request: Request,
    call_next,
) -> Response:
    start_time = perf_counter()
    status_code = 500
    ACTIVE_REQUESTS.inc()

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        duration = perf_counter() - start_time

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=str(status_code),
        ).inc()

        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path,
        ).observe(duration)

        ACTIVE_REQUESTS.dec()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/metrics")
def metrics() -> Response:
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )

@app.get("/drift-report", include_in_schema=False)
def drift_report() -> FileResponse:
    """Return the latest Evidently data-drift dashboard."""

    if not DRIFT_REPORT_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="The drift report has not been generated.",
        )

    return FileResponse(
        path=DRIFT_REPORT_PATH,
        media_type="text/html",
    )


@app.post("/predict", response_model=BikeDemandResponse)
def predict(request: BikeDemandRequest) -> BikeDemandResponse:
    values = request.model_dump()
    prediction_date = values.pop("date")

    values["month"] = prediction_date.month
    values["day_of_week"] = prediction_date.weekday()

    features = pd.DataFrame([values], columns=ALL_FEATURES)

    PREDICTION_COUNT.inc()

    start_time = perf_counter()
    prediction = get_model().predict(features)
    prediction_duration = perf_counter() - start_time

    predicted_count = int(round(np.maximum(prediction[0], 0)))

    PREDICTION_LATENCY.observe(prediction_duration)
    PREDICTION_VALUE.set(predicted_count)

    return BikeDemandResponse(
        predicted_bike_count=predicted_count,
    )

