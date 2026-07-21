"""Functions for evaluating and saving model results."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def evaluate_model(
    model: Any,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate a regression model on held-out test data."""

    predictions = model.predict(x_test)

    metrics = {
        "mae": float(
            mean_absolute_error(y_test, predictions)
        ),
        "rmse": float(
            mean_squared_error(
                y_test,
                predictions,
            ) ** 0.5
        ),
        "r2": float(
            r2_score(y_test, predictions)
        ),
        "mape": float(
            mean_absolute_percentage_error(
                y_test,
                predictions,
            )
        ),
    }

    prediction_results = pd.DataFrame(
        {
            "actual_total_lift": y_test.to_numpy(),
            "predicted_total_lift": predictions,
        },
        index=y_test.index,
    )

    prediction_results["absolute_error"] = (
        prediction_results["actual_total_lift"]
        - prediction_results["predicted_total_lift"]
    ).abs()

    prediction_results = prediction_results.reset_index(
        drop=True
    )

    return metrics, prediction_results


def save_metrics(
    metrics: dict[str, float],
    output_path: str | Path,
) -> None:
    """Save model metrics as JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)


def save_predictions(
    predictions: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save model predictions as a CSV file."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(
        path,
        index=False,
    )
