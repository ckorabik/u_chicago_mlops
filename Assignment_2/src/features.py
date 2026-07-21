"""Functions for creating model features."""

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd


SURVEY_COLUMNS = [
    "eat",
    "background",
    "experience",
    "schedule",
    "howlong",
]

LIFT_COLUMNS = [
    "deadlift",
    "candj",
    "snatch",
    "backsq",
]

TARGET_COLUMN = "total_lift"


def clean_feature_name(value: str) -> str:
    """Convert a survey response into a safe feature-column name."""

    cleaned_value = value.strip().lower()

    cleaned_value = re.sub(
        r"[^a-z0-9]+",
        "_",
        cleaned_value,
    )

    return cleaned_value.strip("_")


def encode_multiselect_column(
    data: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """Encode a pipe-separated survey column as binary features."""

    encoded_data = data[column].str.get_dummies(sep="|")

    encoded_data = encoded_data.rename(
        columns=lambda value: f"{column}_{clean_feature_name(value)}"
    )

    encoded_data = encoded_data.astype(int)

    return encoded_data


def select_features(
    data: pd.DataFrame,
    selected_features: Sequence[str],
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Select model features while always retaining the target column."""

    selected = list(selected_features)

    if len(selected) != len(set(selected)):
        raise ValueError("Selected features must not contain duplicates.")

    if target_column in selected:
        raise ValueError(
            f"Do not include the target column '{target_column}' in "
            "selected_features; it is retained automatically."
        )

    missing_features = sorted(set(selected) - set(data.columns))

    if missing_features:
        raise ValueError(f"Selected features are not available: {missing_features}")

    if target_column not in data.columns:
        raise ValueError(f"Target column is not available: {target_column}")

    return data.loc[:, [*selected, target_column]].copy()


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create the regression target and encoded model features."""

    feature_data = data.copy()

    # Create the regression target.
    feature_data[TARGET_COLUMN] = feature_data[LIFT_COLUMNS].sum(axis=1)

    feature_data[TARGET_COLUMN] = pd.to_numeric(
        feature_data[TARGET_COLUMN],
        errors="coerce",
    )

    feature_data[TARGET_COLUMN] = feature_data[TARGET_COLUMN].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    feature_data = feature_data.dropna(subset=[TARGET_COLUMN])

    # The individual lifts define the prediction target, so retaining them
    # as model features would leak the answer into the training data.
    feature_data = feature_data.drop(columns=LIFT_COLUMNS)

    encoded_frames = []

    # Encode all pipe-separated survey columns.
    for column in SURVEY_COLUMNS:
        encoded_column = encode_multiselect_column(
            feature_data,
            column,
        )

        encoded_frames.append(encoded_column)

    # Encode gender as one binary column.
    feature_data["gender_male"] = (feature_data["gender"] == "Male").astype(int)

    # Encode region as conventional one-hot features.
    region_encoded = pd.get_dummies(
        feature_data["region"],
        prefix="region",
        dtype=int,
    )

    encoded_frames.append(region_encoded)

    # Remove original categorical columns.
    columns_to_remove = SURVEY_COLUMNS + ["gender", "region"]

    feature_data = feature_data.drop(columns=columns_to_remove)

    feature_data = pd.concat(
        [feature_data, *encoded_frames],
        axis=1,
    )

    feature_data = feature_data.reset_index(drop=True)

    return feature_data
