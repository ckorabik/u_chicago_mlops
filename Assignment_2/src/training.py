"""Functions for splitting data and training the model."""

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


TARGET_COLUMN = "total_lift"

TARGET_COMPONENT_COLUMNS = [
    "deadlift",
    "candj",
    "snatch",
    "backsq",
]


def split_data(
    data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """Create training and testing datasets."""

    columns_to_exclude = [
        TARGET_COLUMN,
        *TARGET_COMPONENT_COLUMNS,
    ]

    features = data.drop(
        columns=columns_to_exclude,
        errors="ignore",
    )

    target = data[TARGET_COLUMN]

    # Ensure every model feature is numeric.
    nonnumeric_columns = features.select_dtypes(
        exclude=["number", "bool"]
    ).columns.tolist()

    if nonnumeric_columns:
        raise ValueError(
            "Nonnumeric features remain after feature engineering: "
            f"{nonnumeric_columns}"
        )

    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
    )


def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> Any:
    """Train an XGBoost regression model."""

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=250,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(x_train, y_train)

    return model
