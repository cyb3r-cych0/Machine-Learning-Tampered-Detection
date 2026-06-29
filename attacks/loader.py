"""
Load Dataset
"""

from pathlib import Path

import pandas as pd


def load_dataset(
    parquet_file: Path,
    csv_file: Path
):

    if parquet_file.exists():

        df = pd.read_parquet(
            parquet_file,
            engine="pyarrow"
        )

        source = "PARQUET"

    elif csv_file.exists():

        df = pd.read_csv(
            csv_file,
            low_memory=False
        )

        source = "CSV"

    else:

        raise FileNotFoundError(
            "Baseline dataset not found."
        )

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True
    )

    return df, source