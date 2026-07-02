"""
Metrics tests.
"""

import numpy as np

from detection.metrics import (

    compute_metrics

)


def test_metrics():

    y_true = np.array(

        [0,0,0,1,1,1]

    )

    prediction = np.array(

        [0,0,1,1,0,1]

    )

    score = np.array(

        [

            0.1,

            0.2,

            0.8,

            0.9,

            0.3,

            0.7

        ]

    )

    metrics = compute_metrics(

        y_true,

        prediction,

        score

    )

    assert metrics["accuracy"] >= 0

    assert metrics["precision"] >= 0

    assert metrics["recall"] >= 0

    assert metrics["f1"] >= 0

    assert metrics["roc_auc"] >= 0

    assert metrics["pr_auc"] >= 0

    assert metrics["tp"] == 2

    assert metrics["fp"] == 1

    assert metrics["fn"] == 1

    assert metrics["tn"] == 2