"""
Schema and data validation.
This module performs validation ONLY.
"""

import pandas as pd


# ----------------------------------------------------
# Canonical Schema
# ----------------------------------------------------

REQUIRED_COLUMNS = [

    "country",

    "location_id",

    "sensor_id",

    "timestamp_utc",

    "value"

]


# ----------------------------------------------------
# Required Columns
# ----------------------------------------------------

def validate_columns(df):

    missing = [

        column

        for column in REQUIRED_COLUMNS

        if column not in df.columns

    ]

    if missing:

        raise ValueError(

            f"Missing required columns: {missing}"

        )


# ----------------------------------------------------
# Datatypes
# ----------------------------------------------------

def validate_dtypes(df):

    if not pd.api.types.is_datetime64_any_dtype(

        df["timestamp_utc"]

    ):

        raise TypeError(

            "timestamp_utc must be datetime."

        )

    if not pd.api.types.is_numeric_dtype(

        df["value"]

    ):

        raise TypeError(

            "value must be numeric."

        )


# ----------------------------------------------------
# Empty Dataset
# ----------------------------------------------------

def validate_not_empty(df):

    if df.empty:

        raise ValueError(

            "Dataset contains zero records."

        )


# ----------------------------------------------------
# Missing Timestamp
# ----------------------------------------------------

def validate_missing_timestamp(df):

    missing = df["timestamp_utc"].isna().sum()

    if missing > 0:

        raise ValueError(

            f"{missing} missing timestamps detected."

        )


# ----------------------------------------------------
# Missing PM2.5
# ----------------------------------------------------

def validate_missing_values(df):

    missing = df["value"].isna().sum()

    if missing > 0:

        raise ValueError(

            f"{missing} missing PM2.5 values detected."

        )


# ----------------------------------------------------
# Duplicate Records
# ----------------------------------------------------

def validate_duplicates(df):

    duplicates = df.duplicated().sum()

    return int(duplicates)


# ----------------------------------------------------
# Country Integrity
# ----------------------------------------------------

def validate_country(df):

    invalid = (

        df["country"]

        .isna()

        .sum()

    )

    if invalid > 0:

        raise ValueError(

            "Country contains missing values."

        )


# ----------------------------------------------------
# Sensor Integrity
# ----------------------------------------------------

def validate_sensor(df):

    invalid = (

        df["sensor_id"]

        .isna()

        .sum()

    )

    if invalid > 0:

        raise ValueError(

            "Sensor IDs contain missing values."

        )


# ----------------------------------------------------
# Location Integrity
# ----------------------------------------------------

def validate_location(df):

    invalid = (

        df["location_id"]

        .isna()

        .sum()

    )

    if invalid > 0:

        raise ValueError(

            "Location IDs contain missing values."

        )


# ----------------------------------------------------
# Timestamp Ordering
# ----------------------------------------------------

def validate_timestamp_order(df):

    if not (

        df["timestamp_utc"]

        .is_monotonic_increasing

    ):

        return False

    return True


# ----------------------------------------------------
# Summary
# ----------------------------------------------------

def validation_report(df):
    return {

        "input_rows": len(df),

        "input_columns": len(df.columns),

        "countries": df["country"].nunique(),

        "stations": df["location_id"].nunique(),

        "sensors": df["sensor_id"].nunique(),

        "duplicates": validate_duplicates(df),

        "sorted_before_cleaning": validate_timestamp_order(df)

    }


# ----------------------------------------------------
# Master Validation
# ----------------------------------------------------

def validate_dataset(df):

    validate_not_empty(df)

    validate_columns(df)

    validate_dtypes(df)

    validate_missing_timestamp(df)

    validate_missing_values(df)

    validate_country(df)

    validate_sensor(df)

    validate_location(df)

    report = validation_report(df)

    return report