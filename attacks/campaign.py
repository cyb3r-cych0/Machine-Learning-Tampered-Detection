"""

Attack campaign generation.

A campaign specifies WHICH observations will be attacked.
Attack modules only modify the observations assigned here.
"""

import numpy as np
import pandas as pd

from .config import (

    RANDOM_SEED,

    DEFAULT_ATTACK_PERCENTAGE

)


ATTACK_TYPES = [

    "Constant Bias",

    "Gradual Drift",

    "Spike Suppression",

    "Random Stealth"

]


def create_campaign(

    df,

    attack_fraction=DEFAULT_ATTACK_PERCENTAGE

):

    rng = np.random.default_rng(

        RANDOM_SEED

    )

    working = df.copy()

    # -----------------------------------------
    # Campaign metadata
    # -----------------------------------------

    working["attack_id"] = -1

    working["attack_type"] = "None"

    working["attack_active"] = False

    working["attack_strength"] = 0.0

    # -----------------------------------------
    # Candidate sensors
    # -----------------------------------------

    sensors = (

        working

        [

            [

                "country",

                "location_id",

                "sensor_id"

            ]

        ]

        .drop_duplicates()

        .reset_index(

            drop=True

        )

    )

    n_attack = max(

        1,

        int(

            len(sensors)

            *

            attack_fraction

        )

    )

    selected = rng.choice(

        sensors.index,

        size=n_attack,

        replace=False

    )

    campaign = []

    attack_id = 1

    # -----------------------------------------
    # Assign attacks
    # -----------------------------------------

    for sensor_index in selected:

        sensor = sensors.loc[

            sensor_index

        ]

        attack_type = rng.choice(

            ATTACK_TYPES,

            p=[

                0.25,

                0.25,

                0.25,

                0.25

            ]

        )

        campaign.append(

            {
                "random_seed": RANDOM_SEED,

                "attack_id":

                    attack_id,

                "country":

                    sensor["country"],

                "location_id":

                    sensor["location_id"],

                "sensor_id":

                    sensor["sensor_id"],

                "attack_type":

                    attack_type

            }

        )

        attack_id += 1

    campaign = pd.DataFrame(

        campaign

    )

    return campaign