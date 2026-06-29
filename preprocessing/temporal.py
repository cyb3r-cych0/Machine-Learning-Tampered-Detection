"""
Temporal feature engineering.
Deterministic feature generation.
"""

import numpy as np

from .config import (
    ROLLING_WINDOW,
    EMA_WINDOW
)


# ---------------------------------------------------------
# Calendar Features
# ---------------------------------------------------------

def calendar_features(df):

    df = df.copy()

    t = df["timestamp_utc"]

    df["hour"] = t.dt.hour

    df["day"] = t.dt.day

    df["weekday"] = t.dt.dayofweek

    df["week"] = t.dt.isocalendar().week.astype(int)

    df["month"] = t.dt.month

    df["quarter"] = t.dt.quarter

    df["year"] = t.dt.year

    df["dayofyear"] = t.dt.dayofyear

    df["weekend"] = (

        df["weekday"] >= 5

    ).astype(int)

    return df


# ---------------------------------------------------------
# Cyclic Encoding
# ---------------------------------------------------------

def cyclic_features(df):

    df = df.copy()

    df["hour_sin"] = np.sin(
        2*np.pi*df["hour"]/24
    )

    df["hour_cos"] = np.cos(
        2*np.pi*df["hour"]/24
    )

    df["month_sin"] = np.sin(
        2*np.pi*df["month"]/12
    )

    df["month_cos"] = np.cos(
        2*np.pi*df["month"]/12
    )

    return df


# ---------------------------------------------------------
# Lag Features
# ---------------------------------------------------------

def lag_features(df):

    df = df.copy()

    group = [

        "country",

        "location_id",

        "sensor_id"

    ]

    for lag in [1, 3, 6, 12]:

        df[f"lag_{lag}"] = (

            df

            .groupby(group)["value"]

            .shift(lag)

        )

    return df


# ---------------------------------------------------------
# Rolling Features
# ---------------------------------------------------------

def rolling_features(df):

    df = df.copy()

    group = [

        "country",

        "location_id",

        "sensor_id"

    ]

    rolling = (

        df

        .groupby(group)["value"]

        .rolling(

            ROLLING_WINDOW,

            min_periods=1

        )

    )

    df["rolling_mean"] = (

        rolling

        .mean()

        .reset_index(

            level=group,

            drop=True

        )

    )

    df["rolling_std"] = (

        rolling

        .std()

        .reset_index(

            level=group,

            drop=True

        )

    )

    df["rolling_median"] = (

        rolling

        .median()

        .reset_index(

            level=group,

            drop=True

        )

    )

    return df


# ---------------------------------------------------------
# EMA
# ---------------------------------------------------------

def ema_features(df):

    df = df.copy()

    group = [

        "country",

        "location_id",

        "sensor_id"

    ]

    df["ema"] = (

        df

        .groupby(group)["value"]

        .transform(

            lambda x:

            x.ewm(

                span=EMA_WINDOW,

                adjust=False

            ).mean()

        )

    )

    return df


# ---------------------------------------------------------
# First Difference
# ---------------------------------------------------------

def difference_features(df):

    df = df.copy()

    group = [

        "country",

        "location_id",

        "sensor_id"

    ]

    df["difference"] = (

        df

        .groupby(group)["value"]

        .diff()

    )

    lag = df["lag_1"].replace(0, np.nan)

    df["rate_of_change"] = (
            df["difference"] / lag
    )

    df["rate_of_change"] = (
        df["rate_of_change"]
        .replace([np.inf, -np.inf], np.nan)
    )

    return df


# ---------------------------------------------------------
# Master
# ---------------------------------------------------------

def temporal_features(df):

    df = calendar_features(df)

    df = cyclic_features(df)

    df = lag_features(df)

    df = rolling_features(df)

    df = ema_features(df)

    df = difference_features(df)

    return df