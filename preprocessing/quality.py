"""
Observation quality assessment.
Creates quality flags and a composite quality score.
"""

import numpy as np


# --------------------------------------------------------
# Missing
# --------------------------------------------------------

def flag_missing(df):

    df = df.copy()

    df["is_missing"] = (

        df["value"].isna()

    )

    return df


# --------------------------------------------------------
# Duplicate
# --------------------------------------------------------

def flag_duplicate(df):

    df = df.copy()

    if "duplicate_timestamp" not in df.columns:

        df["duplicate_timestamp"] = False

    df["is_duplicate"] = (

        df["duplicate_timestamp"]

    )

    return df


# --------------------------------------------------------
# Outlier
# --------------------------------------------------------

def flag_outlier(df):

    df = df.copy()

    if "is_spike" not in df.columns:

        df["is_spike"] = False

    df["is_outlier"] = (

        df["is_spike"]

    )

    return df


# --------------------------------------------------------
# Gap
# --------------------------------------------------------

def flag_gap(df):

    df = df.copy()

    if "is_gap" not in df.columns:

        df["is_gap"] = False

    return df


# --------------------------------------------------------
# Flatline
# --------------------------------------------------------

def flag_flatline(df):

    df = df.copy()

    if "is_flatline" not in df.columns:

        df["is_flatline"] = False

    return df


# --------------------------------------------------------
# Constant Sequence
# --------------------------------------------------------

def flag_constant(df):

    df = df.copy()

    if "constant_sequence" not in df.columns:

        df["constant_sequence"] = False

    return df


# --------------------------------------------------------
# Composite Quality Score
# --------------------------------------------------------

def compute_quality_score(df):

    df = df.copy()

    penalties = (

        df["is_gap"].astype(int)

        +

        df["is_flatline"].astype(int)

        +

        df["is_duplicate"].astype(int)

        +

        df["is_outlier"].astype(int)

        +

        df["constant_sequence"].astype(int)

    )

    score = 100 - penalties * 20

    score = score.clip(

        lower=0,

        upper=100

    )

    df["quality_score"] = score

    return df


# --------------------------------------------------------
# Quality Category
# --------------------------------------------------------

def quality_category(df):

    df = df.copy()

    df["quality_class"] = np.select(

        [

            df["quality_score"] >= 90,

            df["quality_score"] >= 70,

            df["quality_score"] >= 50

        ],

        [

            "Excellent",

            "Good",

            "Fair"

        ],

        default="Poor"

    )

    return df


# --------------------------------------------------------
# Summary Report
# --------------------------------------------------------

def quality_report(df):

    return {

        "excellent":

            int(

                (df["quality_class"] == "Excellent").sum()

            ),

        "good":

            int(

                (df["quality_class"] == "Good").sum()

            ),

        "fair":

            int(

                (df["quality_class"] == "Fair").sum()

            ),

        "poor":

            int(

                (df["quality_class"] == "Poor").sum()

            ),

        "mean_quality_score":

            float(

                df["quality_score"].mean()

            )

    }


# --------------------------------------------------------
# Master
# --------------------------------------------------------

def quality_assessment(df):

    df = flag_missing(df)

    df = flag_duplicate(df)

    df = flag_outlier(df)

    df = flag_gap(df)

    df = flag_flatline(df)

    df = flag_constant(df)

    df = compute_quality_score(df)

    df = quality_category(df)

    report = quality_report(df)

    return df, report