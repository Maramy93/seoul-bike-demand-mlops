# Seoul Bike Demand MLOps

[![CI](https://github.com/Maramy93/seoul-bike-demand-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/Maramy93/seoul-bike-demand-mlops/actions/workflows/ci.yml)

An end-to-end MLOps project that predicts hourly bicycle rental demand in Seoul using weather, calendar, and operational information.

## Live deployment

The prediction API and Evidently drift dashboard are publicly deployed on a Linux VPS.

- API documentation: https://maram-bike.duckdns.org/docs
- Health check: https://maram-bike.duckdns.org/health
- Evidently drift dashboard: https://maram-bike.duckdns.org/drift-report
- Source code: https://github.com/Maramy93/seoul-bike-demand-mlops

### Deployment architecture

```text
User
  → HTTPS
  → DuckDNS domain
  → Nginx reverse proxy
  → Docker container
  → FastAPI
  → trained scikit-learn model
```

The FastAPI container is bound internally to `127.0.0.1:1084`. Nginx forwards public HTTPS requests to the container, and Certbot provides HTTPS with automatic certificate renewal.

## Dataset

The project uses the [UCI Seoul Bike Sharing Demand dataset](https://archive.ics.uci.edu/dataset/560/seoul+bike+sharing+demand).

The dataset contains 8,760 hourly observations from December 2017 through November 2018.

## Project workflow

1. Explore and validate the dataset.
2. Split the data chronologically into training, validation, and test sets.
3. Establish a mean-prediction baseline.
4. Train and compare regression models.
5. Track experiments and artifacts with MLflow.
6. Orchestrate training with Prefect.
7. Serve predictions through FastAPI.
8. Package the service with Docker.
9. Test changes automatically with pytest and GitHub Actions.
10. Deploy the API to a Linux VPS with Nginx and HTTPS.
11. Monitor API operations with Prometheus and Grafana.
12. Detect input-feature drift with Evidently.

## Model results

Lower MAE is better.

| Experiment | Validation MAE |
|---|---:|
| Mean baseline | 596.76 |
| Linear regression | 390.55 |
| Random forest | 238.96 |
| Tuned random forest | 228.95 |

The selected tuned Random Forest achieved a final test MAE of **199.49 bicycles**.

The test set was preserved until final model selection.

## Project structure

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   └── raw/
├── models/
├── monitoring/
│   ├── evidently/
│   │   ├── drift_report.py
│   │   └── reports/
│   │       ├── seoul_bike_drift_report.html
│   │       └── seoul_bike_drift_report.json
│   ├── grafana/
│   │   ├── dashboards/
│   │   │   ├── dashboards.yml
│   │   │   └── seoul-bike-operational.json
│   │   └── datasources/
│   │       └── prometheus.yml
│   └── prometheus/
│       └── prometheus.yml
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_baseline_model.ipynb
├── reports/
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── metrics.py
│   ├── train.py
│   └── training_flow.py
├── tests/
│   └── test_api.py
├── .dockerignore
├── .env.example
├── .gitignore
├── .python-version
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── README.md
└── uv.lock
```

## Installation

This project uses Python 3.11 and `uv`.

```bash
git clone https://github.com/Maramy93/seoul-bike-demand-mlops.git
cd seoul-bike-demand-mlops
uv sync
```

Download `SeoulBikeData.csv` from UCI and place it at:

```text
data/raw/SeoulBikeData.csv
```

Raw data and trained model artifacts are intentionally excluded from Git.

## Train the model

Run:

```bash
uv run python src/train.py
```

Training creates:

```text
models/bike_demand_pipeline.joblib
```

Training also records parameters, metrics, tags, and the model artifact with MLflow.

## View MLflow experiments

Start the MLflow user interface:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open:

```text
http://127.0.0.1:5000
```

## Run the Prefect workflow

Start the local Prefect server:

```bash
uv run prefect server start
```

In a second terminal, run:

```bash
PREFECT_API_URL=http://127.0.0.1:4200/api uv run python -m src.training_flow
```

Open the Prefect dashboard:

```text
http://127.0.0.1:4200
```

## Run the prediction API locally

The trained model must exist before starting the API.

```bash
uv run uvicorn src.api:app --reload
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

## Try the deployed model online

The easiest way to use the trained model is through the public Swagger interface:

https://maram-bike.duckdns.org/docs

Then:

1. Expand `POST /predict`.
2. Click **Try it out**.
3. Enter the request data.
4. Click **Execute**.

### Example prediction request

```json
{
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
  "functioning_day": "Yes"
}
```

Example response:

```json
{
  "predicted_bike_count": 1704
}
```

The exact prediction depends on the saved model. A successful request returns HTTP status `200`.

### Test with curl

A prediction can also be requested from a terminal:

```bash
curl -X POST "https://maram-bike.duckdns.org/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
    "functioning_day": "Yes"
  }'
```

## Reproduce the project from GitHub

A new user can reproduce the project with these steps:

```bash
git clone https://github.com/Maramy93/seoul-bike-demand-mlops.git
cd seoul-bike-demand-mlops
uv sync
```

Download the dataset and place it at:

```text
data/raw/SeoulBikeData.csv
```

Train and save the model:

```bash
uv run python src/train.py
```

Start the API:

```bash
uv run uvicorn src.api:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

The dataset and trained model are excluded from Git because they are external or generated artifacts. The training code allows the model to be regenerated.

## Run automated tests

Run:

```bash
uv run pytest -v
```

The tests cover:

- The health endpoint.
- A valid prediction request.
- Rejection of invalid input.
- The Prometheus metrics endpoint.
- The Evidently drift-report endpoint.

GitHub Actions runs these tests automatically on pushes and pull requests to `main`.

## Run with Docker locally

Train the model first so the local model artifact exists:

```bash
uv run python src/train.py
```

Build the Docker image:

```bash
docker build -t seoul-bike-api .
```

Run the container:

```bash
docker run --rm --name seoul-bike-api-container -p 8000:8000 seoul-bike-api
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Monitoring with Prometheus and Grafana

The deployed API includes operational monitoring using Prometheus and Grafana.

Prometheus collects metrics from the FastAPI `/metrics` endpoint every 10 seconds. Grafana displays the collected metrics in a provisioned operational dashboard.

The dashboard monitors:

- API availability
- Total number of predictions
- Latest predicted bicycle count
- HTTP request rate by endpoint
- Server error rate
- API response time at the 95th percentile
- Model prediction time at the 95th percentile
- Active HTTP requests

### Monitoring architecture

```text
FastAPI /metrics
       ↓
Prometheus
       ↓
Grafana dashboard
```

### Run the monitoring stack locally

Create a private `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Set a secure Grafana administrator password in `.env`:

```env
GRAFANA_ADMIN_PASSWORD=your-secure-password
```

The real `.env` file must not be committed to Git.

Start the API, Prometheus, and Grafana:

```bash
docker compose up --build -d
```

Check the services:

```bash
docker compose ps
```

Open:

- API documentation: http://127.0.0.1:1084/docs
- Prometheus targets: http://127.0.0.1:1085/targets
- Grafana: http://127.0.0.1:1086

The Grafana username is `admin`.

Stop the stack with:

```bash
docker compose down
```

### Access VPS monitoring securely

Prometheus and Grafana are bound to localhost on the VPS rather than exposed directly to the internet.

Create an SSH tunnel from a local terminal:

```bash
ssh \
  -L 2085:127.0.0.1:1085 \
  -L 2086:127.0.0.1:1086 \
  Maram@35.202.67.240
```

While the tunnel remains open, access:

- Prometheus targets: http://127.0.0.1:2085/targets
- Grafana dashboard: http://127.0.0.1:2086/d/seoul-bike-operational

## Data drift monitoring with Evidently

The project uses Evidently to detect changes in the distributions of model input features.

- Live Evidently dashboard: https://maram-bike.duckdns.org/drift-report

The drift analysis compares two chronologically separated portions of the dataset:

- **Reference data:** the first 70% of the dataset, representing the model-training period.
- **Current data:** the final 15% of the dataset, representing a later observation period.

The following 12 model input features are monitored:

- Hour
- Temperature
- Humidity
- Wind speed
- Visibility
- Dew-point temperature
- Solar radiation
- Rainfall
- Snowfall
- Season
- Holiday
- Functioning day

Evidently compares the reference and current distributions for every feature using statistical distance tests suitable for numerical and categorical data.

Dataset drift is reported when at least 50% of the monitored features are detected as drifted.

### Current drift result

| Metric | Result |
|---|---:|
| Features analyzed | 12 |
| Drifted features | 7 |
| Share of drifted features | 58.3% |
| Drift threshold | 50% |
| Dataset drift detected | Yes |

The result indicates that the later observation period has a significantly different input-feature distribution from the training reference period. Much of this difference is expected because Seoul bicycle demand and weather data are strongly seasonal.

Data drift does not automatically prove that model accuracy has decreased. Instead, it indicates that model performance should be investigated and that retraining may be required when representative newer data becomes available.

### Drift-monitoring process

```text
Training-period features
          ↓
   Reference dataset
          │
          ├── Evidently comparison
          │
    Current dataset
          ↑
 Later-period features
          ↓
 HTML and JSON drift reports
```

### Generate the Evidently report

Run:

```bash
uv run python monitoring/evidently/drift_report.py
```

This creates:

```text
monitoring/evidently/reports/seoul_bike_drift_report.html
monitoring/evidently/reports/seoul_bike_drift_report.json
```

Open the report locally:

```bash
open monitoring/evidently/reports/seoul_bike_drift_report.html
```

Alternatively, start the API and open:

```text
http://127.0.0.1:8000/drift-report
```

The deployed FastAPI service exposes the generated report at:

```text
https://maram-bike.duckdns.org/drift-report
```

### Interpretation and response to drift

When drift is detected:

1. Identify which features changed.
2. Determine whether the changes are expected, such as seasonal weather changes.
3. Evaluate the model using newer labelled observations when they become available.
4. Retrain the model if its predictive performance has degraded.
5. Promote and deploy the validated replacement model.

## VPS deployment

The production deployment uses the following components:

- A Linux VPS hosts the application.
- Docker packages and runs the FastAPI service.
- Prometheus collects operational metrics.
- Grafana provides operational visualisation.
- Evidently provides batch input-feature drift analysis.
- The API container is available internally through port `1084`.
- Prometheus is available internally through port `1085`.
- Grafana is available internally through port `1086`.
- Nginx acts as a reverse proxy for the public API.
- DuckDNS provides the public domain.
- Certbot and Let’s Encrypt provide HTTPS.
- Docker restart policies restart the services after a server reboot.

The internal production port mappings are:

```text
127.0.0.1:1084 → FastAPI container port 8000
127.0.0.1:1085 → Prometheus container port 9090
127.0.0.1:1086 → Grafana container port 3000
```

The public endpoint is:

```text
https://maram-bike.duckdns.org
```

## Technology stack

- Python 3.11
- pandas and NumPy
- scikit-learn
- MLflow
- Prefect
- FastAPI and Uvicorn
- pytest
- Prometheus
- Grafana
- Evidently
- Docker and Docker Compose
- Nginx
- DuckDNS
- Certbot and Let’s Encrypt
- GitHub Actions
- uv