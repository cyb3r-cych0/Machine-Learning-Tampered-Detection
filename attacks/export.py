"""
Export Data
"""

import json

from pathlib import Path


def export_dataset(

    attacked,

    labels,

    campaign,

    output_dir

):

    output_dir.mkdir(

        parents=True,

        exist_ok=True

    )

    attacked.to_parquet(

        output_dir /

        "attacked_dataset.parquet",

        index=False,

        engine="pyarrow"

    )

    attacked.to_csv(

        output_dir /

        "attacked_dataset.csv",

        index=False

    )

    labels.to_csv(

        output_dir /

        "ground_truth.csv",

        index=False

    )

    campaign.to_csv(

        output_dir /

        "campaign.csv",

        index=False

    )


def export_report(

    attacked,

    labels,

    campaign,

    output_dir

):

    report = {

        "rows":

            int(len(attacked)),

        "attacked_rows":

            int(

                attacked["attack_active"].sum()

            ),

        "campaigns":

            int(len(campaign)),

        "attack_types":

            campaign["attack_type"]

            .value_counts()

            .to_dict()

    }

    with open(

        output_dir /

        "attack_report.json",

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(

            report,

            f,

            indent=4

        )