import numpy as np
import src.api as api_module

from fastapi.testclient import TestClient

from src.api import app



client = TestClient(app)


VALID_REQUEST = {
    "date": "2018-10-08",
    "hour": 8,
    "temperature": 15.2,
    "humidity": 55,
    "wind_speed": 1.8,
    "visibility": 1800,
    "dew_point_temperature": 6.2,
    "solar_radiation": 0.8,
    "rainfall": 0,
    "snowfall": 0,
    "season": "Autumn",
    "holiday": "No Holiday",
    "functioning_day": "Yes",
}


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_endpoint(monkeypatch) -> None:
    class FakeModel:
        def predict(self, features):
            return np.array([1704.0])

    monkeypatch.setattr(api_module, "model", FakeModel())

    response = client.post("/predict", json=VALID_REQUEST)

    assert response.status_code == 200

    predicted_count = response.json()["predicted_bike_count"]

    assert isinstance(predicted_count, int)
    assert predicted_count == 1704


def test_invalid_hour_is_rejected() -> None:
    invalid_request = VALID_REQUEST.copy()
    invalid_request["hour"] = 30

    response = client.post("/predict", json=invalid_request)

    assert response.status_code == 422