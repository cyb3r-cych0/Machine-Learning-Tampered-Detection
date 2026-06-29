"""
Dataset cleaning.
This module performs deterministic cleaning only.
"""

import pandas as pd


def remove_duplicates(df):

    before = len(df)

    df = df.drop_duplicates(
        subset=[
            "country",
            "location_id",
            "sensor_id",
            "timestamp_utc"
        ]
    ).copy()

    removed = before - len(df)

    return df, removed


def remove_missing(df):

    before = len(df)

    df = df.dropna(
        subset=[
            "timestamp_utc",
            "value"
        ]
    ).copy()

    removed = before - len(df)

    return df, removed


def remove_negative_pm25(df):

    before = len(df)

    df = df.loc[
        df["value"] >= 0
    ].copy()

    removed = before - len(df)

    return df, removed


def normalize_timestamps(df):

    df = df.copy()

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True,
        errors="coerce"
    )

    return df


def sort_dataset(df):

    return (
        df
        .sort_values(
            [
                "country",
                "location_id",
                "sensor_id",
                "timestamp_utc"
            ]
        )
        .reset_index(
            drop=True
        )
    )


def clean_dataset(df):

    report = {}

    df = normalize_timestamps(df)

    df, removed = remove_duplicates(df)
    report["duplicates_removed"] = removed

    df, removed = remove_missing(df)
    report["missing_removed"] = removed

    df, removed = remove_negative_pm25(df)
    report["negative_removed"] = removed

    df = sort_dataset(df)

    report["sorted_after_cleaning"] = (

        df["timestamp_utc"]

        .is_monotonic_increasing

    )

    report["rows_after_cleaning"] = len(df)

    """ 
        Link baseline and attacked datasets.
        Measure exactly which observations were modified.
        Join SHAP values and detection results back to the original records.
        Simplify evaluation and debugging.
        Essential for attack simulation.
    """

    df.insert(
        0,
        "observation_id",
        range(1, len(df) + 1)
    )

    df["is_attacked"] = False

    df["attack_type"] = "None"

    df["attack_magnitude"] = 0.0

    if "observation_id" not in df.columns:
        df.insert(
            0,
            "observation_id",
            range(1, len(df) + 1)
        )

    report["rows_after_cleaning"] = len(df)

    return df, report