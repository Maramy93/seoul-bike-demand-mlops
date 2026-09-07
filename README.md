# Seoul Bike Demand MLOps

[![CI](https://github.com/Maramy93/seoul-bike-demand-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/Maramy93/seoul-bike-demand-mlops/actions/workflows/ci.yml)

An end-to-end MLOps project that predicts hourly bicycle rental demand in Seoul using weather, calendar, and operational information.

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
├── .github/workflows/ci.yml.yml
├── data/raw/
├── models/
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_baseline_model.ipynb
├── src/
│   ├── api.py
│   ├── train.py
│   └── training_flow.py
├── tests/
│   └── test_api.py
├── Dockerfile
├── pyproject.toml
└── uv.lock
```

## Installation

This project uses Python 3.11 and `uv`.

```bash
git clone https://github.com/Maramy93/seoul-bike-demand-mlops.git
cd seoul-bike-demand-mlops
uv sync
```

Download `SeoulBikeData.csv` from UCI and place it inside:

```text
data/raw/SeoulBikeData.csv
```

Raw data and trained model artifacts are intentionally excluded from Git.

## Train the model

```bash
uv run python src/train.py
```

Training creates:

```text
models/bike_demand_pipeline.joblib
```

It also records parameters, metrics, tags, and the model artifact with MLflow.

## View MLflow experiments

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

## Run the prediction API

The trained model must exist before starting the API.

```bash
uv run uvicorn src.api:app --reload
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

### Example prediction request

Send a `POST` request to `/predict`:

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

The exact prediction depends on the saved model.

## Run automated tests

```bash
uv run pytest -v
```

The tests cover:

- the health endpoint;
- a valid prediction request;
- rejection of invalid input.

GitHub Actions runs these tests automatically on pushes and pull requests to `main`.

## Run with Docker

Train the model first so that the local model artifact exists:

```bash
uv run python src/train.py
```

Build the image:

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

## Technology stack

- Python 3.11
- pandas and NumPy
- scikit-learn
- MLflow
- Prefect
- FastAPI and Uvicorn
- pytest
- Docker
- GitHub Actions
- uv