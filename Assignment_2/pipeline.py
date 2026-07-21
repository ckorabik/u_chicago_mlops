"""Run the end-to-end athlete performance pipeline."""

from pathlib import Path

import joblib
import mlflow
import mlflow.xgboost

from src.evaluation import (
    evaluate_model,
    save_metrics,
    save_predictions,
)
from src.features import TARGET_COLUMN, create_features, select_features
from src.feature_store import load_features, save_features
from src.ingestion import load_data
from src.preprocessing import preprocess_data
from src.training import split_data, train_model


RAW_DATA_PATH = Path("data/athletes.csv")
PROCESSED_DATA_PATH = Path("data/processed.csv")
FEATURE_STORE_PATH = Path("feature_store.db")

MLFLOW_DATABASE_PATH = Path("mlflow.db")
MLFLOW_ARTIFACT_PATH = Path("mlartifacts")
MLFLOW_EXPERIMENT_NAME = "athlete-performance"

ACTIVE_FEATURE_VERSION = "v1"
FEATURE_VERSIONS: dict[str, list[str] | None] = {
    # None means use every engineered feature available in this dataset.
    "v1": None,
    # Example smaller version for experimentation:
    "v2_demographics": ["age", "height", "weight", "gender_male"],
}

MODEL_PATH = Path("models/xgboost_model.joblib")
FEATURE_COLUMNS_PATH = Path("models/feature_columns.joblib")

METRICS_PATH = Path("reports/metrics.json")
PREDICTIONS_PATH = Path("reports/predictions.csv")


def create_output_directories() -> None:
    """Create all directories used for pipeline outputs."""

    for directory in [
        PROCESSED_DATA_PATH.parent,
        MODEL_PATH.parent,
        METRICS_PATH.parent,
        MLFLOW_ARTIFACT_PATH,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def configure_mlflow() -> None:
    """Configure MLflow to use only project-local storage."""

    project_root = Path(__file__).resolve().parent
    database_path = (project_root / MLFLOW_DATABASE_PATH).resolve()
    artifact_path = (project_root / MLFLOW_ARTIFACT_PATH).resolve()

    mlflow.set_tracking_uri(f"sqlite:///{database_path.as_posix()}")

    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            MLFLOW_EXPERIMENT_NAME,
            artifact_location=artifact_path.as_uri(),
        )

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def main() -> None:
    """Execute the complete machine learning pipeline."""

    create_output_directories()
    configure_mlflow()

    print("1. Loading raw athlete data...")
    raw_data = load_data(RAW_DATA_PATH)

    print(f"   Raw rows: {len(raw_data):,}")

    print("2. Preprocessing athlete data...")
    cleaned_data = preprocess_data(raw_data)

    print(f"   Cleaned rows: {len(cleaned_data):,}")

    print("3. Creating model features...")
    processed_data = create_features(cleaned_data)

    selected_features = FEATURE_VERSIONS[ACTIVE_FEATURE_VERSION]

    if selected_features is None:
        selected_features = [
            column for column in processed_data.columns if column != TARGET_COLUMN
        ]

    processed_data = select_features(
        processed_data,
        selected_features,
    )

    processed_data.to_csv(
        PROCESSED_DATA_PATH,
        index=False,
    )

    save_features(
        processed_data,
        FEATURE_STORE_PATH,
        ACTIVE_FEATURE_VERSION,
    )
    processed_data = load_features(
        FEATURE_STORE_PATH,
        ACTIVE_FEATURE_VERSION,
    )

    print(f"   Processed dataset shape: {processed_data.shape}")

    print("4. Splitting training and testing data...")
    x_train, x_test, y_train, y_test = split_data(processed_data)

    print(f"   Training rows: {len(x_train):,}")
    print(f"   Testing rows: {len(x_test):,}")
    print(f"   Model features: {x_train.shape[1]:,}")

    mlflow.xgboost.autolog(
        log_models=True,
        log_datasets=True,
        silent=True,
    )

    with mlflow.start_run():
        mlflow.log_param("feature_store", str(FEATURE_STORE_PATH))
        mlflow.log_param("feature_version", ACTIVE_FEATURE_VERSION)
        mlflow.log_param("training_rows", len(x_train))
        mlflow.log_param("testing_rows", len(x_test))
        mlflow.log_param("feature_count", x_train.shape[1])

        print("5. Training XGBoost model...")
        model = train_model(
            x_train,
            y_train,
        )

        print("6. Evaluating model...")
        metrics, predictions = evaluate_model(
            model,
            x_test,
            y_test,
        )

        mlflow.log_metrics(metrics)

        print("7. Saving pipeline outputs...")
        joblib.dump(
            model,
            MODEL_PATH,
        )

        joblib.dump(
            x_train.columns.tolist(),
            FEATURE_COLUMNS_PATH,
        )

        save_metrics(
            metrics,
            METRICS_PATH,
        )

        save_predictions(
            predictions,
            PREDICTIONS_PATH,
        )

        mlflow.log_artifact(str(METRICS_PATH), "reports")
        mlflow.log_artifact(str(PREDICTIONS_PATH), "reports")

    print("\nPipeline complete.")
    print(f"MAE:  {metrics['mae']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"R²:   {metrics['r2']:.4f}")
    print(f"MAPE: {metrics['mape']:.2%}")

    print("\nGenerated files:")
    print(f"- {PROCESSED_DATA_PATH}")
    print(f"- {FEATURE_STORE_PATH}")
    print(f"- {MLFLOW_DATABASE_PATH}")
    print(f"- {MLFLOW_ARTIFACT_PATH}")
    print(f"- {MODEL_PATH}")
    print(f"- {FEATURE_COLUMNS_PATH}")
    print(f"- {METRICS_PATH}")
    print(f"- {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
