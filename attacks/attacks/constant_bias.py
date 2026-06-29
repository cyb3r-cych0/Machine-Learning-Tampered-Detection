"""
Constant Bias Injection Attack
"""

import numpy as np

from ..config import (
    RANDOM_SEED,
    BIAS_MIN,
    BIAS_MAX,
)

from ..utils import (
    initialize_attack_columns,
    select_attack_window,
    apply_attack_metadata,
    get_campaign_subset,
)


def apply_constant_bias(df, campaign):

    rng = np.random.default_rng(RANDOM_SEED)

    attacked = initialize_attack_columns(df)

    campaign = get_campaign_subset(
        campaign,
        "Constant Bias"
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

        bias = rng.uniform(

            BIAS_MIN,

            BIAS_MAX

        )

        attacked.loc[
            attacked_index,
            "value"
        ] += bias

        attacked = apply_attack_metadata(

            attacked,

            attacked_index,

            attack,

            bias,

            attack_start,

            attack_end

        )

    return attacked