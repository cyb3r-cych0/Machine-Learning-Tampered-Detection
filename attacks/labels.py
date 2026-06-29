"""
Ground-truth label generation.
"""


def create_ground_truth(df):

    labels = (

        df

        .loc[

            df["attack_active"]

        ]

        [

            [

                "observation_id",

                "country",

                "location_id",

                "sensor_id",

                "timestamp_utc",

                "attack_id",

                "attack_type",

                "attack_strength",

                "attack_start",

                "attack_end",

                "original_value",

                "attacked_value"

            ]

        ]

        .copy()

    )

    labels["difference"] = (

        labels["attacked_value"]

        -

        labels["original_value"]

    )

    labels = labels.sort_values(

        [

            "attack_id",

            "timestamp_utc"

        ]

    )

    labels.reset_index(

        drop=True,

        inplace=True

    )

    return labels