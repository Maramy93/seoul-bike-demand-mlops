from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "SeoulBikeData.csv"


COLUMN_NAMES = {
    "Date": "date",
    "Rented Bike Count": "rented_bike_count",
    "Hour": "hour",
    "Temperature(°C)": "temperature",
    "Humidity(%)": "humidity",
    "Wind speed (m/s)": "wind_speed",
    "Visibility (10m)": "visibility",
    "Dew point temperature(°C)": "dew_point_temperature",
    "Solar Radiation (MJ/m2)": "solar_radiation",
    "Rainfall(mm)": "rainfall",
    "Snowfall (cm)": "snowfall",
    "Seasons": "season",
    "Holiday": "holiday",
    "Functioning Day": "functioning_day",
}

NUMERIC_FEATURES = [
    "hour",
    "temperature",
    "humidity",
    "wind_speed",
    "visibility",
    "dew_point_temperature",
    "solar_radiation",
    "rainfall",
    "snowfall",
    "month",
    "day_of_week",
]

CATEGORICAL_FEATURES = [
    "season",
    "holiday",
    "functioning_day",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "rented_bike_count"

MODEL_PARAMS = {
    "n_estimators": 400,
    "max_depth": 25,
    "min_samples_leaf": 1,
    "max_features": 0.8,
    "random_state": 42,
    "n_jobs": -1,
}


def load_data(data_path: Path) -> pd.DataFrame:
    """Load the raw Seoul Bike dataset."""
    df = pd.read_csv(data_path, encoding="ISO-8859-1")
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns and create time-related features."""
    df = df.rename(columns=COLUMN_NAMES).copy()

    df["date"] = pd.to_datetime(
        df["date"],
        format="%d/%m/%Y",
    )

    df["timestamp"] = (
        df["date"]
        + pd.to_timedelta(df["hour"], unit="h")
    )

    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data chronologically into train, validation, and test."""
    train_end = int(len(df) * 0.70)
    validation_end = int(len(df) * 0.85)

    train_df = df.iloc[:train_end].copy()
    validation_df = df.iloc[
        train_end:validation_end
    ].copy()
    test_df = df.iloc[validation_end:].copy()

    return train_df, validation_df, test_df


def build_model() -> Pipeline:
    """Create the preprocessing and Random Forest pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model = RandomForestRegressor(**MODEL_PARAMS)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def evaluate_model(
    model: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[float, np.ndarray]:
    """Predict bike demand and calculate MAE."""
    predictions = model.predict(features)
    predictions = np.maximum(predictions, 0)

    mae = mean_absolute_error(target, predictions)

    return mae, predictions

def main() -> None:
    raw_df = load_data(DATA_PATH)
    prepared_df = prepare_data(raw_df)

    train_df, validation_df, test_df = split_data(prepared_df)

    X_train = train_df[ALL_FEATURES]
    y_train = train_df[TARGET]

    X_validation = validation_df[ALL_FEATURES]
    y_validation = validation_df[TARGET]

    # Calculate the mean baseline.
    training_mean = y_train.mean()
    baseline_predictions = np.full(
        len(y_validation),
        training_mean,
    )
    baseline_mae = mean_absolute_error(
        y_validation,
        baseline_predictions,
    )

    # Train on the training set and evaluate on validation.
    validation_model = build_model()
    validation_model.fit(X_train, y_train)

    validation_mae, _ = evaluate_model(
        validation_model,
        X_validation,
        y_validation,
    )

    # Retrain the selected model using train + validation.
    final_training_df = pd.concat(
        [train_df, validation_df],
        ignore_index=True,
    )

    X_final_train = final_training_df[ALL_FEATURES]
    y_final_train = final_training_df[TARGET]

    X_test = test_df[ALL_FEATURES]
    y_test = test_df[TARGET]

    final_model = build_model()
    final_model.fit(X_final_train, y_final_train)

    test_mae, test_predictions = evaluate_model(
        final_model,
        X_test,
        y_test,
    )

    # Configure local MLflow tracking.
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("seoul-bike-demand")

    with mlflow.start_run(
        run_name="tuned-random-forest"
    ) as run:
        mlflow.log_params(MODEL_PARAMS)

        mlflow.log_metric("baseline_validation_mae", baseline_mae)
        mlflow.log_metric("validation_mae", validation_mae)
        mlflow.log_metric("test_mae", test_mae)

        mlflow.set_tag(
            "model_type",
            "RandomForestRegressor",
        )
        mlflow.set_tag(
            "dataset",
            "UCI Seoul Bike Sharing Demand",
        )

        signature = infer_signature(
            X_test,
            test_predictions,
        )

        mlflow.sklearn.log_model(
            sk_model=final_model,
            name="model",
            signature=signature,
            input_example=X_test.head(3),
        )

        print(f"MLflow run ID: {run.info.run_id}")

    print(f"Loaded rows: {len(prepared_df)}")
    print(f"Baseline validation MAE: {baseline_mae:.2f}")
    print(f"Model validation MAE: {validation_mae:.2f}")
    print(f"Final test MAE: {test_mae:.2f}")

if __name__ == "__main__":
    main()