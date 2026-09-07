from datetime import date
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.train import ALL_FEATURES


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "bike_demand_pipeline.joblib"
)

model = joblib.load(MODEL_PATH)

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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/predict", response_model=BikeDemandResponse)
def predict(request: BikeDemandRequest) -> BikeDemandResponse:
    values = request.model_dump()
    prediction_date = values.pop("date")

    values["month"] = prediction_date.month
    values["day_of_week"] = prediction_date.weekday()

    features = pd.DataFrame([values], columns=ALL_FEATURES)

    prediction = model.predict(features)
    predicted_count = int(round(np.maximum(prediction[0], 0)))

    return BikeDemandResponse(
        predicted_bike_count=predicted_count,
    )