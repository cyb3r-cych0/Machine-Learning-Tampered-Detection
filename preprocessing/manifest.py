"""
Dataset provenance manifest
Every preprocessing run produces one immutable
manifest describing the dataset used in experiments.
"""

import hashlib
import json
from datetime import datetime


def dataframe_hash(df):

    h = hashlib.sha256()

    h.update(

        df.to_csv(

            index=False

        ).encode("utf-8")

    )

    return h.hexdigest()


def build_manifest(

    df,

    source,

    validation,

    cleaning,

    integrity,

    quality

):

    manifest = {

        "generated_at":

            datetime.now().isoformat(),

        "dataset": {

            "rows":

                int(len(df)),

            "columns":

                int(len(df.columns)),

            "countries":

                int(df["country"].nunique()),

            "stations":

                int(df["location_id"].nunique()),

            "sensors":

                int(df["sensor_id"].nunique())

        },

        "source":

            source,

        "validation":

            validation,

        "cleaning":

            cleaning,

        "integrity":

            integrity,

        "quality":

            quality,

        "sha256":

            dataframe_hash(df)

    }

    return manifest


def save_manifest(

    manifest,

    output_dir

):

    with open(

        output_dir /

        "dataset_manifest.json",

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(

            manifest,

            f,

            indent=4,

            default=str

        )