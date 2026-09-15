"""Generate an Evidently data-drift report for Seoul Bike Demand."""

from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from evidently.metrics import DriftedColumnsCount


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "SeoulBikeData.csv"
REPORTS_DIR = Path(__file__).parent / "reports"

DRIFT_THRESHOLD = 0.5

COLUMN_NAMES = {
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

DRIFT_FEATURES = list(COLUMN_NAMES.values())


def load_data() -> pd.DataFrame:
    """Load and prepare the Seoul bicycle-demand dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset was not found at: {DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH, encoding="cp1252")
    data = data.rename(columns=COLUMN_NAMES)

    missing_columns = [
        column for column in DRIFT_FEATURES if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    return data[DRIFT_FEATURES].dropna().reset_index(drop=True)


def split_reference_and_current(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use the training period as reference and final period as current."""

    reference_end = int(len(data) * 0.70)
    current_start = int(len(data) * 0.85)

    reference = data.iloc[:reference_end].copy()
    current = data.iloc[current_start:].copy()

    return reference, current


def generate_drift_report() -> None:
    """Compare reference and current data and save Evidently reports."""

    data = load_data()
    reference, current = split_reference_and_current(data)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report = Report(
        [
            DataDriftPreset(drift_share=DRIFT_THRESHOLD),
            DriftedColumnsCount(drift_share=DRIFT_THRESHOLD),
        ]
    )

    result = report.run(
        reference_data=reference,
        current_data=current,
    )

    html_path = REPORTS_DIR / "seoul_bike_drift_report.html"
    json_path = REPORTS_DIR / "seoul_bike_drift_report.json"

    result.save_html(str(html_path))
    result.save_json(str(json_path))

    print("Evidently drift analysis completed.")
    print(f"Reference rows: {len(reference)}")
    print(f"Current rows: {len(current)}")
    print(f"HTML dashboard: {html_path}")
    print(f"JSON results: {json_path}")


if __name__ == "__main__":
    generate_drift_report()