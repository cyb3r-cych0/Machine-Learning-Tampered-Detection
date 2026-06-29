"""
Gradual Drift Attack
"""

import numpy as np

from ..config import (
    DRIFT_STEP,
    DRIFT_MAX,
)

from ..utils import (
    initialize_attack_columns,
    get_campaign_subset,
    select_attack_window,
    apply_attack_metadata,
)


def apply_gradual_drift(df, campaign):

    attacked = initialize_attack_columns(df)

    campaign = get_campaign_subset(
        campaign,
        "Gradual Drift"
    )

    for _, attack in campaign.iterrows():

        mask = (

            (attacked["country"] == attack["country"]) &

            (attacked["location_id"] == attack["location_id"]) &

            (attacked["sensor_id"] == attack["sensor_id"])

        )

        sensor_df = attacked.loc[mask]

        if sensor_df.empty:
            continue

        attacked_index, attack_start, attack_end = (

            select_attack_window(
                attacked,
                sensor_df
            )

        )

        # ---------------------------------------
        # IMPORTANT
        # Start drift at DRIFT_STEP instead of 0
        # ---------------------------------------

        drift = (

            np.arange(

                1,

                len(attacked_index) + 1,

                dtype=float

            )

            * DRIFT_STEP

        )

        drift = np.clip(

            drift,

            0,

            DRIFT_MAX

        )

        attacked.loc[
            attacked_index,
            "value"
        ] += drift

        attacked = apply_attack_metadata(

            attacked,

            attacked_index,

            attack,

            float(drift.max()),

            attack_start,

            attack_end

        )

    return attacked