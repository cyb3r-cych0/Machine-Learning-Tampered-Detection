"""
Comparison tests.
"""

import numpy as np

from detection.comparison import (

    build_comparison_table,

    pairwise_statistics

)


def test_comparison():

    reports = {

        "Detector A": {

            "metrics": {

                "accuracy": 0.90,

                "precision": 0.85,

                "recall": 0.80,

                "f1": 0.82,

                "roc_auc": 0.91,

                "pr_auc": 0.88,

                "false_positive_rate": 0.05

            },

            "confidence_intervals": {

                "f1": {

                    "lower": 0.79,

                    "upper": 0.85

                },

                "recall": {

                    "lower": 0.77,

                    "upper": 0.83

                },

                "roc_auc": {

                    "lower": 0.89,

                    "upper": 0.93

                }

            },

            "prediction": np.array(

                [0, 0, 1, 1]

            ),

            "score": np.array(

                [0.10, 0.20, 0.80, 0.90]

            )

        },

        "Detector B": {

            "metrics": {

                "accuracy": 0.88,

                "precision": 0.82,

                "recall": 0.78,

                "f1": 0.80,

                "roc_auc": 0.89,

                "pr_auc": 0.86,

                "false_positive_rate": 0.07

            },

            "confidence_intervals": {

                "f1": {

                    "lower": 0.76,

                    "upper": 0.83

                },

                "recall": {

                    "lower": 0.74,

                    "upper": 0.81

                },

                "roc_auc": {

                    "lower": 0.86,

                    "upper": 0.91

                }

            },

            "prediction": np.array(

                [0, 1, 1, 1]

            ),

            "score": np.array(

                [0.15, 0.60, 0.75, 0.95]

            )

        }

    }

    comparison = build_comparison_table(

        reports

    )

    statistics = pairwise_statistics(

        reports,

        np.array(

            [0, 0, 1, 1]

        )

    )

    assert len(

        comparison

    ) == 2

    assert len(

        statistics

    ) == 1

    assert "Accuracy" in comparison.columns

    assert "McNemar_p" in statistics.columns

    assert "Wilcoxon_p" in statistics.columns