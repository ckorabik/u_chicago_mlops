"""Functions for loading the athlete dataset."""

from pathlib import Path

import pandas as pd


def load_data(file_path: str | Path) -> pd.DataFrame:
    """Load athlete data from a CSV file."""

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")

    data = pd.read_csv(path)

    if data.empty:
        raise ValueError(f"The dataset is empty: {path.resolve()}")

    return data
