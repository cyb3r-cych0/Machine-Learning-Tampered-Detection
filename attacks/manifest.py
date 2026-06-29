"""
Manifest
"""

import hashlib
import json
from datetime import datetime


def sha256(df):

    h = hashlib.sha256()

    h.update(

        df.to_csv(

            index=False

        ).encode()

    )

    return h.hexdigest()


def create_manifest(

    attacked,

    campaign

):

    return {

        "generated_at":

            datetime.now().isoformat(),

        "rows":

            int(

                len(attacked)

            ),

        "columns":

            int(

                len(attacked.columns)

            ),

        "campaigns":

            int(

                len(campaign)

            ),

        "sha256":

            sha256(

                attacked

            )

    }


def save_manifest(

    manifest,

    output_dir

):

    with open(

        output_dir /

        "attack_manifest.json",

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(

            manifest,

            f,

            indent=4
        )