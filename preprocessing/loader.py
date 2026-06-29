"""
Dataset loader.
Loads the canonical dataset produced by EDA.
"""

from pathlib import Path
import pandas as pd


def load_dataset(parquet_file: Path, csv_file: Path):

    if parquet_file.exists():
        try:
            df = pd.read_parquet(
                parquet_file,
                engine="pyarrow"
            )
            source = "parquet"

        except ImportError:
            if not csv_file.exists():
                raise

            df = pd.read_csv(
                csv_file,
                low_memory=False
            )
            source = "csv"

    elif csv_file.exists():
        df = pd.read_csv(
            csv_file,
            low_memory=False
        )
        source = "csv"

    else:
        raise FileNotFoundError(
            f"\nParquet : {parquet_file}"
            f"\nCSV     : {csv_file}"
        )

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True
    )

    return df, source
