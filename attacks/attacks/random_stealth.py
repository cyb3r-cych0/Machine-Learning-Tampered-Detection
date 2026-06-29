"""
Random Stealth Perturbation Attack
"""

import numpy as np

from ..config import (

    RANDOM_SEED,

    STEALTH_STD

)

from ..utils import (

    initialize_attack_columns,

    get_campaign_subset,

    select_attack_window,

    apply_attack_metadata

)


def apply_random_stealth(df, campaign):

    rng = np.random.default_rng(

        RANDOM_SEED

    )

    attacked = initialize_attack_columns(df)

    campaign = get_campaign_subset(

        campaign,

        "Random Stealth"

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

        noise = rng.normal(

            0,

            STEALTH_STD,

            len(attacked_index)

        )

        attacked.loc[

            attacked_index,

            "value"

        ] += noise

        attacked = apply_attack_metadata(

            attacked,

            attacked_index,

            attack,

            float(np.std(noise)),

            attack_start,

            attack_end

        )

    return attacked