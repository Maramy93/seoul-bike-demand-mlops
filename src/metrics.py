"""Prometheus metrics for the Seoul Bike Demand API."""

from prometheus_client import Counter, Gauge, Histogram, Info


REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10],
)

PREDICTION_COUNT = Counter(
    "predictions_total",
    "Total number of bicycle-demand predictions",
)

PREDICTION_VALUE = Gauge(
    "prediction_value_bicycles",
    "Most recent predicted bicycle count",
)

PREDICTION_LATENCY = Histogram(
    "prediction_duration_seconds",
    "Time required to calculate a prediction",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1],
)

ACTIVE_REQUESTS = Gauge(
    "active_requests",
    "Number of HTTP requests currently being processed",
)

MODEL_INFO = Info(
    "model_info",
    "Information about the deployed model",
)

MODEL_INFO.info(
    {
        "name": "seoul_bike_demand_random_forest",
        "version": "1.0.0",
    }
)
