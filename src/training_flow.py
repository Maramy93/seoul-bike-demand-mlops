import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from prefect import flow, get_run_logger, task
from sklearn.metrics import mean_absolute_error

from src.train import (
    ALL_FEATURES,
    DATA_PATH,
    MODEL_PARAMS,
    TARGET,
    build_model,
    evaluate_model,
    load_data,
    prepare_data,
    split_data,
)


@task(
    name="load-bike-data",
    retries=2,
    retry_delay_seconds=5,
)
def load_data_task() -> pd.DataFrame:
    """Load the dataset as a Prefect task."""
    logger = get_run_logger()

    df = load_data(DATA_PATH)

    logger.info("Loaded %s rows", len(df))

    return df


@task(name="prepare-bike-data")
def prepare_data_task(
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """Clean the data and create time features."""
    logger = get_run_logger()

    prepared_df = prepare_data(raw_df)

    logger.info(
        "Prepared data from %s to %s",
        prepared_df["timestamp"].min(),
        prepared_df["timestamp"].max(),
    )

    return prepared_df


@task(name="split-bike-data")
def split_data_task(
    prepared_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create chronological train, validation, and test sets."""
    logger = get_run_logger()

    train_df, validation_df, test_df = split_data(
        prepared_df
    )

    logger.info(
        "Split sizes — train: %s, validation: %s, test: %s",
        len(train_df),
        len(validation_df),
        len(test_df),
    )

    return train_df, validation_df, test_df


@task(name="train-evaluate-and-log-model")
def train_model_task(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, float]:
    """Train, evaluate, and log the model with MLflow."""
    logger = get_run_logger()

    X_train = train_df[ALL_FEATURES]
    y_train = train_df[TARGET]

    X_validation = validation_df[ALL_FEATURES]
    y_validation = validation_df[TARGET]

    training_mean = y_train.mean()
    baseline_predictions = np.full(
        len(y_validation),
        training_mean,
    )

    baseline_mae = mean_absolute_error(
        y_validation,
        baseline_predictions,
    )

    validation_model = build_model()
    validation_model.fit(X_train, y_train)

    validation_mae, _ = evaluate_model(
        validation_model,
        X_validation,
        y_validation,
    )

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

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("seoul-bike-demand")

    with mlflow.start_run(
        run_name="prefect-tuned-random-forest"
    ) as run:
        mlflow.log_params(MODEL_PARAMS)

        mlflow.log_metrics(
            {
                "baseline_validation_mae": baseline_mae,
                "validation_mae": validation_mae,
                "test_mae": test_mae,
            }
        )

        mlflow.set_tag("orchestrator", "Prefect")
        mlflow.set_tag(
            "model_type",
            "RandomForestRegressor",
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

        logger.info("MLflow run ID: %s", run.info.run_id)

    metrics = {
        "baseline_validation_mae": float(baseline_mae),
        "validation_mae": float(validation_mae),
        "test_mae": float(test_mae),
    }

    logger.info("Metrics: %s", metrics)

    return metrics


@flow(
    name="seoul-bike-training-flow",
    log_prints=True,
)
def training_flow() -> dict[str, float]:
    """Run the complete bike-demand training workflow."""
    logger = get_run_logger()

    logger.info("Starting Seoul bike training flow")

    raw_df = load_data_task()
    prepared_df = prepare_data_task(raw_df)

    train_df, validation_df, test_df = split_data_task(
        prepared_df
    )

    metrics = train_model_task(
        train_df,
        validation_df,
        test_df,
    )

    logger.info("Training flow completed successfully")

    return metrics


if __name__ == "__main__":
    training_flow()