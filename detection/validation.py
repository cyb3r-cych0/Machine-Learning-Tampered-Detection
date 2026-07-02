"""
detection/validation.py
"""

REQUIRED_COLUMNS = [

    "observation_id",

    "sensor_id",

    "timestamp_utc",

    "value"

]


def validate(

    baseline,

    attacked,

    ground_truth

):

    for dataset_name, dataset in [

        ("Baseline", baseline),

        ("Attacked", attacked)

    ]:

        missing = [

            c

            for c in REQUIRED_COLUMNS

            if c not in dataset.columns

        ]

        if missing:

            raise ValueError(

                f"{dataset_name}: missing {missing}"

            )

    if ground_truth.empty:

        raise ValueError(

            "Ground truth is empty."

        )

    if baseline["observation_id"].duplicated().any():

        raise ValueError(

            "Duplicate observation_id in baseline."

        )

    if attacked["observation_id"].duplicated().any():

        raise ValueError(

            "Duplicate observation_id in attacked dataset."

        )

    return True