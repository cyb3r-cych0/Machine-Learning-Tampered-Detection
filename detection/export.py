"""
detection/export.py
"""

import json


def export_detector_results(

    report,

    output_dir

):

    detector = (

        report["model"]

        .lower()

        .replace(

            " ",

            "_"

        )

    )

    detector_dir = (

        output_dir /

        detector

    )

    detector_dir.mkdir(

        parents=True,

        exist_ok=True

    )

    # ------------------------------------------
    # Metrics
    # ------------------------------------------

    import pandas as pd

    pd.DataFrame(

        [report["metrics"]]

    ).to_csv(

        detector_dir /

        "metrics.csv",

        index=False

    )

    # ------------------------------------------
    # Confidence intervals
    # ------------------------------------------

    pd.DataFrame(

        report["confidence_intervals"]

    ).T.to_csv(

        detector_dir /

        "confidence_intervals.csv"

    )

    # ------------------------------------------
    # Attack recall
    # ------------------------------------------

    report["attack_recall"].to_csv(

        detector_dir /

        "attack_recall.csv",

        index=False

    )

    # ------------------------------------------
    # Country recall
    # ------------------------------------------

    report["country_recall"].to_csv(

        detector_dir /

        "country_recall.csv",

        index=False

    )


def export_detector_comparison(

    comparison,

    output_dir,

    detector_a,

    detector_b

):

    comparison_dir = (

        output_dir /

        "comparison"

    )

    comparison_dir.mkdir(

        parents=True,

        exist_ok=True

    )

    filename = (

        detector_a.lower().replace(" ", "_")

        +

        "_vs_"

        +

        detector_b.lower().replace(" ", "_")

        +

        ".json"

    )

    with open(

        comparison_dir /

        filename,

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(

            comparison,

            f,

            indent=4

        )

def export_comparison(

    comparison,

    statistics,

    output_dir

):

    comparison.to_csv(

        output_dir /

        "detector_comparison.csv",

        index=False

    )

    statistics.to_csv(

        output_dir /

        "pairwise_statistics.csv",

        index=False

    )