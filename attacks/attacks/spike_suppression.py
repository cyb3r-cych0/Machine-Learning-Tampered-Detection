"""
Spike Suppression Attack

Suppresses high PM2.5 peaks within an attack window,
making pollution episodes appear less severe.
"""

from ..config import (
    SUPPRESSION_FACTOR,
)

from ..utils import (
    initialize_attack_columns,
    get_campaign_subset,
    select_attack_window,
    apply_attack_metadata,
)


def apply_spike_suppression(df, campaign):

    attacked = initialize_attack_columns(df)

    campaign = get_campaign_subset(

        campaign,

        "Spike Suppression"

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

        window = attacked.loc[

            attacked_index,

            "value"

        ]

        threshold = window.quantile(0.90)

        spikes = window > threshold

        spike_index = window[spikes].index

        if len(spike_index) == 0:

            continue

        attacked.loc[

            spike_index,

            "value"

        ] *= SUPPRESSION_FACTOR

        attacked = apply_attack_metadata(

            attacked,

            spike_index,

            attack,

            SUPPRESSION_FACTOR,

            attack_start,

            attack_end

        )

    return attacked