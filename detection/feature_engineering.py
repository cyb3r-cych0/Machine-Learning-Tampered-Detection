"""
detection/feature_engineering.py

Shared feature engineering for all detection models.
"""

import numpy as np
import pandas as pd


"""
detection/feature_engineering.py
"""

import numpy as np
import pandas as pd

# ----------------------------------------------------------
# Feature Groups
# ----------------------------------------------------------

TABULAR_FEATURES = [

    "value",

    "hour_sin",
    "hour_cos",

    "month_sin",
    "month_cos",

    "lag_1",
    "lag_3",
    "lag_6",
    "lag_12",

    "difference",

    "rate_of_change",

    "time_gap_hours"

]

STATISTICAL_FEATURES = [

    "rolling_mean",

    "rolling_std",

    "rolling_median",

    "ema",

    "zscore",

    "flatline_std"

]

ALL_FEATURES = (

    TABULAR_FEATURES +

    STATISTICAL_FEATURES

)


def _build_features(df):

    df = df.copy()

    df = (

        df

        .sort_values(

            [

                "sensor_id",

                "timestamp_utc"

            ]

        )

        .reset_index(drop=True)

    )

    g = df.groupby(

        "sensor_id",

        sort=False

    )

    # ---------------------------------
    # Time
    # ---------------------------------

    hour = df["timestamp_utc"].dt.hour

    month = df["timestamp_utc"].dt.month

    df["hour_sin"] = np.sin(

        2 * np.pi * hour / 24

    )

    df["hour_cos"] = np.cos(

        2 * np.pi * hour / 24

    )

    df["month_sin"] = np.sin(

        2 * np.pi * month / 12

    )

    df["month_cos"] = np.cos(

        2 * np.pi * month / 12

    )

    # ---------------------------------
    # Lags
    # ---------------------------------

    for lag in [1, 3, 6, 12]:

        df[f"lag_{lag}"] = (

            g["value"]

            .shift(lag)

        )

    # ---------------------------------
    # Rolling
    # ---------------------------------

    r = (

        g["value"]

        .rolling(

            24,

            min_periods=1

        )

    )

    df["rolling_mean"] = (

        r.mean()

        .reset_index(

            level=0,

            drop=True

        )

    )

    df["rolling_std"] = (

        r.std()

        .reset_index(

            level=0,

            drop=True

        )

    )

    df["rolling_median"] = (

        r.median()

        .reset_index(

            level=0,

            drop=True

        )

    )

    # ---------------------------------
    # EMA
    # ---------------------------------

    df["ema"] = (

        g["value"]

        .transform(

            lambda s:

            s.ewm(

                span=24,

                adjust=False

            ).mean()

        )

    )

    # ---------------------------------
    # Difference
    # ---------------------------------

    df["difference"] = (

        g["value"]

        .diff()

    )

    # ---------------------------------
    # Rate of change
    # ---------------------------------

    previous = (

        g["value"]

        .shift(1)

    )

    df["rate_of_change"] = (

        df["difference"]

        /

        previous.replace(

            0,

            np.nan

        )

    )

    # ---------------------------------
    # Z-score
    # ---------------------------------

    std = df["rolling_std"].replace(

        0,

        np.nan

    )

    df["zscore"] = (

        df["value"]

        -

        df["rolling_mean"]

    ) / std

    # ---------------------------------
    # Flatline
    # ---------------------------------

    df["flatline_std"] = (

        r.std()

        .reset_index(

            level=0,

            drop=True

        )

    )

    # ---------------------------------
    # Time gaps
    # ---------------------------------

    df["time_gap_hours"] = (

        g["timestamp_utc"]

        .diff()

        .dt.total_seconds()

        / 3600

    )

    # ---------------------------------
    # Clean
    # ---------------------------------

    df[ALL_FEATURES] = (

        df[ALL_FEATURES]

        .replace(

            [np.inf, -np.inf],

            np.nan

        )

        .ffill()

        .bfill()

        .fillna(0)

    )

    return df


def prepare_training_features(

    baseline

):

    baseline = _build_features(

        baseline

    )

    return (

        baseline,

        baseline[TABULAR_FEATURES]

    )


def prepare_testing_features(

    attacked

):

    attacked = _build_features(

        attacked

    )

    return (

        attacked,

        attacked[TABULAR_FEATURES]

    )


def prepare_statistical_features(

    df

):

    df = _build_features(

        df

    )

    return (

        df,

        df[STATISTICAL_FEATURES]

    )