"""Functions for cleaning and preprocessing athlete data."""

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "region",
    "age",
    "weight",
    "height",
    "howlong",
    "gender",
    "eat",
    "background",
    "experience",
    "schedule",
    "deadlift",
    "candj",
    "snatch",
    "backsq",
]

COLUMNS_TO_DROP = [
    "affiliate",
    "team",
    "name",
    "athlete_id",
    "fran",
    "helen",
    "grace",
    "filthy50",
    "fgonebad",
    "run400",
    "run5k",
    "pullups",
    "train",
]

NUMERIC_COLUMNS = [
    "age",
    "height",
    "weight",
    "deadlift",
    "candj",
    "snatch",
    "backsq",
]

SURVEY_COLUMNS = [
    "eat",
    "background",
    "experience",
    "schedule",
    "howlong",
]


def validate_columns(data: pd.DataFrame) -> None:
    """Confirm that all columns required by the pipeline exist."""

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Dataset is missing required columns: "
            f"{missing_columns}"
        )


def clean_survey_value(value: object) -> object:
    """
    Remove blank and declined responses from a pipe-separated survey value.

    For example:

    'Option A|Decline to answer|Option B|'

    becomes:

    'Option A|Option B'
    """

    if pd.isna(value):
        return np.nan

    responses = [
        response.strip()
        for response in str(value).split("|")
        if response.strip()
        and response.strip().lower() != "decline to answer"
    ]

    if not responses:
        return np.nan

    return "|".join(responses)


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw athlete dataset and remove invalid observations."""

    cleaned_data = data.copy()

    validate_columns(cleaned_data)

    # Convert relevant measurements to numeric values.
    for column in NUMERIC_COLUMNS:
        cleaned_data[column] = pd.to_numeric(
            cleaned_data[column],
            errors="coerce",
        )

    # Replace infinite numeric values with missing values.
    cleaned_data[NUMERIC_COLUMNS] = cleaned_data[
        NUMERIC_COLUMNS
    ].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # Remove rows missing required values.
    cleaned_data = cleaned_data.dropna(
        subset=REQUIRED_COLUMNS
    )

    # Remove columns that will not be used.
    cleaned_data = cleaned_data.drop(
        columns=COLUMNS_TO_DROP,
        errors="ignore",
    )

    # Apply measurement and demographic filters.
    cleaned_data = cleaned_data[
        cleaned_data["weight"] < 1500
    ]

    cleaned_data = cleaned_data[
        cleaned_data["gender"].isin(["Male", "Female"])
    ]

    cleaned_data = cleaned_data[
        cleaned_data["age"] >= 18
    ]

    cleaned_data = cleaned_data[
        cleaned_data["height"].between(
            48,
            96,
            inclusive="neither",
        )
    ]

    valid_deadlift = (
        (
            (cleaned_data["gender"] == "Male")
            & (cleaned_data["deadlift"] > 0)
            & (cleaned_data["deadlift"] <= 1105)
        )
        |
        (
            (cleaned_data["gender"] == "Female")
            & (cleaned_data["deadlift"] > 0)
            & (cleaned_data["deadlift"] <= 636)
        )
    )

    cleaned_data = cleaned_data[valid_deadlift]

    cleaned_data = cleaned_data[
        cleaned_data["candj"].between(
            0,
            395,
            inclusive="neither",
        )
    ]

    cleaned_data = cleaned_data[
        cleaned_data["snatch"].between(
            0,
            496,
            inclusive="neither",
        )
    ]

    cleaned_data = cleaned_data[
        cleaned_data["backsq"].between(
            0,
            1069,
            inclusive="neither",
        )
    ]

    # Clean each pipe-separated survey response.
    for column in SURVEY_COLUMNS:
        cleaned_data[column] = cleaned_data[column].apply(
            clean_survey_value
        )

    # Remove rows where cleaning left no valid survey responses.
    cleaned_data = cleaned_data.dropna(
        subset=SURVEY_COLUMNS
    )

    # Remove duplicate observations and reset the row index.
    cleaned_data = cleaned_data.drop_duplicates()
    cleaned_data = cleaned_data.reset_index(drop=True)

    return cleaned_data
