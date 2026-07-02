"""
detection/loader.py
"""

import pandas as pd


def _load(parquet_file, csv_file):

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

            f"{parquet_file.name} not found."

        )

    df["timestamp_utc"] = pd.to_datetime(

        df["timestamp_utc"],

        utc=True

    )

    df = df.sort_values(

        "timestamp_utc"

    ).reset_index(

        drop=True

    )

    return df, source


def load_datasets(

    baseline_parquet,

    baseline_csv,

    attacked_parquet,

    attacked_csv,

    ground_truth_file

):

    baseline_df, baseline_source = _load(

        baseline_parquet,

        baseline_csv

    )

    attacked_df, attacked_source = _load(

        attacked_parquet,

        attacked_csv

    )

    ground_truth = pd.read_csv(

        ground_truth_file

    )

    return (

        baseline_df,

        attacked_df,

        ground_truth,

        baseline_source,

        attacked_source

    )