"""
Export tests.
"""

import tempfile

import pandas as pd

from pathlib import Path

from detection.export import (

    export_detector_results

)


def test_export():

    tmp = Path(

        tempfile.mkdtemp()

    )

    report = {

        "model":

            "Isolation Forest",

        "metrics": {

            "accuracy": 0.90

        },

        "confidence_intervals": {

            "f1": {

                "mean": 0.8,

                "lower": 0.7,

                "upper": 0.9

            }

        },

        "attack_recall":

            pd.DataFrame(

                {

                    "attack_type":

                    ["Bias"],

                    "recall":

                    [0.9]

                }

            ),

        "country_recall":

            pd.DataFrame(

                {

                    "country":

                    ["Kenya"],

                    "recall":

                    [0.8]

                }

            )

    }

    export_detector_results(

        report,

        tmp

    )

    assert (

        tmp /

        "isolation_forest" /

        "metrics.csv"

    ).exists()