"""
Attack campaign analytics.
"""

from pathlib import Path

import pandas as pd


def generate_attack_analytics(

    attacked,

    campaign,

    output_dir

):

    analytics_dir = (

        output_dir /

        "analytics"

    )

    analytics_dir.mkdir(

        parents=True,

        exist_ok=True

    )

    attacked_rows = attacked.loc[

        attacked["attack_active"]

    ].copy()

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------

    total_rows = len(attacked)

    attacked_rows_count = len(attacked_rows)

    attack_percentage = round(

        attacked_rows_count * 100 / total_rows,

        3

    )

    summary = pd.DataFrame({

        "Metric": [

            "Total Observations",

            "Attacked Observations",

            "Attack Percentage",

            "Attack Campaigns",

            "Countries",

            "Sensors"

        ],

        "Value": [

            total_rows,

            attacked_rows_count,

            attack_percentage,

            campaign["attack_id"].nunique(),

            attacked_rows["country"].nunique(),

            attacked_rows["sensor_id"].nunique()

        ]

    })

    summary.to_csv(

        analytics_dir /

        "attack_summary.csv",

        index=False

    )

    # -------------------------------------------------
    # Attack Types
    # -------------------------------------------------

    attack_types = (

        campaign

        ["attack_type"]

        .value_counts()

        .rename_axis(

            "attack_type"

        )

        .reset_index(

            name="count"

        )

    )

    attack_types.to_csv(

        analytics_dir /

        "attack_types.csv",

        index=False

    )

    # -------------------------------------------------
    # Country Distribution
    # -------------------------------------------------

    countries = (

        attacked_rows

        .groupby(

            "country"

        )

        .size()

        .reset_index(

            name="attacked_rows"

        )

        .sort_values(

            "attacked_rows",

            ascending=False

        )

    )

    countries.to_csv(

        analytics_dir /

        "attack_by_country.csv",

        index=False

    )

    # -------------------------------------------------
    # Sensor Distribution
    # -------------------------------------------------

    sensors = (

        attacked_rows

        .groupby(

            [

                "country",

                "location_id",

                "sensor_id"

            ]

        )

        .size()

        .reset_index(

            name="attacked_rows"

        )

        .sort_values(

            "attacked_rows",

            ascending=False

        )

    )

    sensors.to_csv(

        analytics_dir /

        "attack_by_sensor.csv",

        index=False

    )

    # -------------------------------------------------
    # Attack Duration
    # -------------------------------------------------

    duration = (

        attacked_rows

        [

            [

                "attack_id",

                "attack_type",

                "attack_start",

                "attack_end"

            ]

        ]

        .drop_duplicates()

    )

    duration["duration_hours"] = (

        pd.to_datetime(

            duration["attack_end"]

        )

        -

        pd.to_datetime(

            duration["attack_start"]

        )

    ).dt.total_seconds() / 3600

    duration.to_csv(

        analytics_dir /

        "attack_duration.csv",

        index=False

    )

    # -------------------------------------------------
    # Attack Strength
    # -------------------------------------------------

    magnitude = (

        attacked_rows

        .groupby(

            [

                "attack_id",

                "attack_type"

            ]

        )

        ["attack_strength"]

        .mean()

        .reset_index(

            name="mean_attack_strength"

        )

    )

    magnitude.to_csv(

        analytics_dir /

        "attack_strength.csv",

        index=False

    )

    # -------------------------------------------------
    # End
    # -------------------------------------------------

    return