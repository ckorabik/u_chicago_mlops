"""Local SQLite persistence for engineered athlete features."""

import sqlite3
import re
from pathlib import Path

import pandas as pd


FEATURE_TABLE_PREFIX = "athlete_features"


def feature_table_name(version: str) -> str:
    """Build a safe SQLite table name for a feature version."""

    if not re.fullmatch(r"[A-Za-z0-9_]+", version):
        raise ValueError(
            "Feature version may contain only letters, numbers, and underscores."
        )

    return f"{FEATURE_TABLE_PREFIX}_{version}"


def save_features(
    features: pd.DataFrame,
    database_path: str | Path,
    version: str,
) -> None:
    """Replace one versioned table in the local feature store."""

    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path) as connection:
        features.to_sql(
            feature_table_name(version),
            connection,
            if_exists="replace",
            index=False,
        )


def load_features(
    database_path: str | Path,
    version: str,
) -> pd.DataFrame:
    """Load one engineered feature version from the local store."""

    path = Path(database_path)

    if not path.exists():
        raise FileNotFoundError(f"Feature store not found: {path.resolve()}")

    with sqlite3.connect(path) as connection:
        return pd.read_sql_query(
            f"SELECT * FROM {feature_table_name(version)}",
            connection,
        )
