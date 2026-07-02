"""
detection/manifest.py
"""

import hashlib
import json
from datetime import datetime


def sha256_dataframe(df):

    return hashlib.sha256(

        df.to_csv(

            index=False

        ).encode()

    ).hexdigest()


def create_manifest(

    baseline,

    attacked,

    reports

):

    return {

        "generated_at":

            datetime.now().isoformat(),

        "baseline_rows":

            len(baseline),

        "attacked_rows":

            len(attacked),

        "detectors":

            list(reports.keys()),

        "baseline_sha256":

            sha256_dataframe(

                baseline

            ),

        "attacked_sha256":

            sha256_dataframe(

                attacked

            )

    }


def save_manifest(

    manifest,

    output_dir

):

    with open(

        output_dir /

        "detection_manifest.json",

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(

            manifest,

            f,

            indent=4
        )