"""
Shared utilities for attack simulation.
"""

import numpy as np

from .config import (
    RANDOM_SEED,
    ATTACK_WINDOW_FRACTION,
    MIN_ATTACK_WINDOW,
)


# ----------------------------------------------------------
# Initialize attack columns
# ----------------------------------------------------------

def initialize_attack_columns(df):

    attacked = df.copy()

    defaults = {

        "original_value": attacked["value"],

        "attacked_value": attacked["value"],

        "attack_id": -1,

        "attack_type": "None",

        "attack_active": False,

        "attack_strength": 0.0,

        "attack_start": None,

        "attack_end": None

    }

    for column, value in defaults.items():

        if column not in attacked.columns:

            attacked[column] = value

    return attacked


# ----------------------------------------------------------
# Campaign Filter
# ----------------------------------------------------------

def get_campaign_subset(

    campaign,

    attack_type

):

    return (

        campaign

        .loc[
            campaign["attack_type"] == attack_type
        ]

        .copy()

        .reset_index(drop=True)

    )


# ----------------------------------------------------------
# Select attack window
# ----------------------------------------------------------

def select_attack_window(

    attacked,

    sensor_df,

    fraction=None,

    minimum=None

):

    if fraction is None:

        fraction = ATTACK_WINDOW_FRACTION

    if minimum is None:

        minimum = MIN_ATTACK_WINDOW

    rng = np.random.default_rng(

        RANDOM_SEED

    )

    window = max(

        minimum,

        int(

            len(sensor_df) * fraction

        )

    )

    window = min(

        window,

        len(sensor_df)

    )

    if len(sensor_df) == window:

        start = 0

    else:

        start = rng.integers(

            0,

            len(sensor_df) - window + 1

        )

    attacked_index = sensor_df.index[
        start:start + window
    ]

    attack_start = attacked.loc[
        attacked_index,
        "timestamp_utc"
    ].min()

    attack_end = attacked.loc[
        attacked_index,
        "timestamp_utc"
    ].max()

    return (

        attacked_index,

        attack_start,

        attack_end

    )


# ----------------------------------------------------------
# Apply metadata
# ----------------------------------------------------------

def apply_attack_metadata(

    attacked,

    attacked_index,

    attack,

    strength,

    attack_start,

    attack_end

):

    attacked.loc[
        attacked_index,
        "attack_id"
    ] = attack["attack_id"]

    attacked.loc[
        attacked_index,
        "attack_type"
    ] = attack["attack_type"]

    attacked.loc[
        attacked_index,
        "attack_active"
    ] = True

    attacked.loc[
        attacked_index,
        "attack_strength"
    ] = float(strength)

    attacked.loc[
        attacked_index,
        "attack_start"
    ] = attack_start

    attacked.loc[
        attacked_index,
        "attack_end"
    ] = attack_end

    attacked.loc[
        attacked_index,
        "attacked_value"
    ] = attacked.loc[
        attacked_index,
        "value"
    ]

    return attacked


# ----------------------------------------------------------
# Finalize dataset
# ----------------------------------------------------------

def finalize_attack_dataset(attacked):

    if "attacked_value" in attacked.columns:

        attacked["value"] = attacked["attacked_value"]

    return attacked