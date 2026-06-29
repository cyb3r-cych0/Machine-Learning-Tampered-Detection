"""
Validation for attack simulation outputs.
"""

import pandas as pd


REQUIRED_COLUMNS = [

    "observation_id",

    "country",

    "location_id",

    "sensor_id",

    "timestamp_utc",

    "value"

]


def validate(df):

    missing = [

        column

        for column in REQUIRED_COLUMNS

        if column not in df.columns

    ]

    if missing:

        raise ValueError(

            f"Missing required columns: {missing}"

        )

    return True


def validate_attacked_dataset(

    attacked,

    campaign

):

    report = {}

    # --------------------------------------------------
    # attack_id uniqueness
    # --------------------------------------------------

    ids = campaign["attack_id"]

    report["unique_attack_ids"] = ids.is_unique

    if not ids.is_unique:

        raise ValueError(

            "Duplicate attack_id detected."

        )

    # --------------------------------------------------
    # attack type
    # --------------------------------------------------

    attacked_rows = attacked[

        attacked["attack_active"]

    ]

    valid = attacked_rows[

        "attack_type"

    ].ne(

        "None"

    ).all()

    report["valid_attack_type"] = valid

    if not valid:

        raise ValueError(

            "Invalid attack_type detected."

        )

    # --------------------------------------------------
    # attack window
    # --------------------------------------------------

    valid_window = (

        attacked_rows["attack_start"]

        <=

        attacked_rows["attack_end"]

    ).all()

    report["valid_attack_window"] = valid_window

    if not valid_window:

        raise ValueError(

            "Invalid attack window."

        )

    # --------------------------------------------------
    # value modified
    # --------------------------------------------------

    difference = (

            attacked_rows["attacked_value"]

            -

            attacked_rows["original_value"]

    ).abs()

    unchanged = attacked_rows.loc[

        difference <= 1e-9,

        [

            "attack_id",

            "attack_type",

            "country",

            "sensor_id",

            "original_value",

            "attacked_value"

        ]

    ]

    report["values_modified"] = len(unchanged) == 0

    if not unchanged.empty:
        print("\nRows that were marked as attacked but unchanged:\n")

        print(unchanged.head(20))

        raise ValueError(

            f"{len(unchanged)} attacked observations were unchanged."

        )

    # --------------------------------------------------
    # attacked value consistency
    # --------------------------------------------------

    consistent = (

        attacked_rows["value"]

        ==

        attacked_rows["attacked_value"]

    ).all()

    report["value_consistency"] = consistent

    if not consistent:

        raise ValueError(

            "value != attacked_value"

        )

    # --------------------------------------------------
    # campaign integrity
    # --------------------------------------------------

    campaign_ids = set(

        campaign["attack_id"]

    )

    dataset_ids = set(

        attacked_rows["attack_id"]

    )

    missing = dataset_ids - campaign_ids

    report["campaign_integrity"] = len(missing) == 0

    if missing:

        raise ValueError(

            f"Unknown attack IDs: {missing}"

        )

    # --------------------------------------------------
    # campaign effectiveness
    # --------------------------------------------------

    for attack_id in campaign["attack_id"]:

        window = attacked_rows.loc[
            attacked_rows["attack_id"] == attack_id
            ]

        if window.empty:
            raise ValueError(
                f"Campaign {attack_id} produced no attacked observations."
            )

        if (
                window["original_value"]
                ==
                window["attacked_value"]
        ).all():
            raise ValueError(
                f"Campaign {attack_id} produced no effective modifications."
            )

    report["campaign_effectiveness"] = True

    # --------------------------------------------------
    # observation uniqueness
    # --------------------------------------------------

    duplicated = attacked_rows.duplicated(

        subset=[

            "observation_id"

        ]

    ).any()

    report["duplicate_observations"] = duplicated

    if duplicated:

        raise ValueError(

            "Observation attacked multiple times."

        )

    report["attacked_rows"] = int(

        len(attacked_rows)

    )

    report["campaigns"] = int(

        len(campaign)

    )

    return report