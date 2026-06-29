"""
Sensor integrity assessment.
Integrity flags generated.
"""

import numpy as np

from .config import (
    MAX_TIME_GAP_HOURS,
    FLATLINE_WINDOW,
    SPIKE_ZSCORE
)


def detect_time_gaps(df):

    df = df.copy()

    df["time_gap_hours"] = (

        df

        .groupby(

            ["country", "location_id", "sensor_id"]

        )["timestamp_utc"]

        .diff()

        .dt.total_seconds()

        / 3600

    )

    df["is_gap"] = (

        df["time_gap_hours"]

        >

        MAX_TIME_GAP_HOURS

    )

    return df


def detect_flatlines(df):

    df = df.copy()

    rolling_std = (

        df

        .groupby(

            ["country", "location_id", "sensor_id"]

        )["value"]

        .rolling(

            FLATLINE_WINDOW,

            min_periods=FLATLINE_WINDOW

        )

        .std()

        .reset_index(level=[0,1,2], drop=True)

    )

    df["flatline_std"] = rolling_std

    df["is_flatline"] = (

        rolling_std.fillna(1)

        == 0

    )

    return df


def detect_spikes(df):

    df = df.copy()

    grouped = (

        df

        .groupby(

            ["country", "location_id", "sensor_id"]

        )["value"]

    )

    mean = grouped.transform("mean")

    std = grouped.transform("std")

    z = (

        df["value"]

        -

        mean

    ) / std.replace(0, np.nan)

    df["zscore"] = z

    df["is_spike"] = (

        np.abs(z)

        >

        SPIKE_ZSCORE

    )

    return df


def detect_clock_duplicates(df):

    duplicate = (

        df

        .duplicated(

            subset=[

                "country",

                "location_id",

                "sensor_id",

                "timestamp_utc"

            ],

            keep=False

        )

    )

    df = df.copy()

    df["duplicate_timestamp"] = duplicate

    return df


def detect_constant_sequences(df):

    df = df.copy()

    run = (

        df

        .groupby(

            [

                "country",

                "location_id",

                "sensor_id"

            ]

        )["value"]

        .diff()

        .fillna(1)

    )

    df["constant_sequence"] = (

        run == 0

    )

    return df


def integrity_assessment(df):

    df = detect_time_gaps(df)

    df = detect_flatlines(df)

    df = detect_spikes(df)

    df = detect_clock_duplicates(df)

    df = detect_constant_sequences(df)

    report = {

        "time_gaps":

            int(df["is_gap"].sum()),

        "flatlines":

            int(df["is_flatline"].sum()),

        "spikes":

            int(df["is_spike"].sum()),

        "duplicate_timestamps":

            int(df["duplicate_timestamp"].sum()),

        "constant_sequences":

            int(df["constant_sequence"].sum())

    }

    return df, report