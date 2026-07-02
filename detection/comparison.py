"""
detection/comparison.py

Detector comparison.
"""

from itertools import combinations

import pandas as pd

from detection.statistics import (

    mcnemar_test,

    wilcoxon_test

)


def build_comparison_table(

    reports

):

    rows = []

    for detector, report in reports.items():

        metrics = report["metrics"]

        ci = report["confidence_intervals"]

        rows.append({

            "Detector": detector,

            "Accuracy": metrics["accuracy"],

            "Precision": metrics["precision"],

            "Recall": metrics["recall"],

            "F1": metrics["f1"],

            "ROC_AUC": metrics["roc_auc"],

            "PR_AUC": metrics["pr_auc"],

            "FPR": metrics["false_positive_rate"],

            "F1_CI_Lower": ci["f1"]["lower"],

            "F1_CI_Upper": ci["f1"]["upper"],

            "Recall_CI_Lower": ci["recall"]["lower"],

            "Recall_CI_Upper": ci["recall"]["upper"],

            "ROC_CI_Lower": ci["roc_auc"]["lower"],

            "ROC_CI_Upper": ci["roc_auc"]["upper"]

        })

    return pd.DataFrame(

        rows

    )


def pairwise_statistics(

    reports,

    y_true

):

    rows = []

    detectors = list(

        reports.keys()

    )

    for i in range(

        len(detectors)

    ):

        for j in range(

            i + 1,

            len(detectors)

        ):

            a = reports[detectors[i]]

            b = reports[detectors[j]]

            length = min(

                len(a["prediction"]),

                len(b["prediction"]),

                len(y_true)

            )

            mcnemar = mcnemar_test(

                a["prediction"][:length],

                b["prediction"][:length],

                y_true[:length]

            )

            wilcoxon = wilcoxon_test(

                a["score"][:length],

                b["score"][:length]

            )

            rows.append({

                "Detector_A": detectors[i],

                "Detector_B": detectors[j],

                "McNemar_Statistic": mcnemar["statistic"],

                "McNemar_p": mcnemar["p_value"],

                "Wilcoxon_Statistic": wilcoxon["statistic"],

                "Wilcoxon_p": wilcoxon["p_value"]

            })

    return pd.DataFrame(rows)