"""
detection/statistics.py

Statistical evaluation utilities.
"""

import numpy as np

from scipy.stats import wilcoxon

from statsmodels.stats.contingency_tables import mcnemar

from sklearn.metrics import (

    f1_score,

    recall_score,

    roc_auc_score

)


def bootstrap_metric(

    y_true,

    prediction,

    score,

    metric,

    n_bootstrap=1000,

    confidence=95,

    random_state=42

):

    rng = np.random.default_rng(

        random_state

    )

    values = []

    n = len(y_true)

    for _ in range(

        n_bootstrap

    ):

        idx = rng.integers(

            0,

            n,

            n

        )

        yt = y_true[idx]

        yp = prediction[idx]

        sc = score[idx]

        if metric == "f1":

            values.append(

                f1_score(

                    yt,

                    yp,

                    zero_division=0

                )

            )

        elif metric == "recall":

            values.append(

                recall_score(

                    yt,

                    yp,

                    zero_division=0

                )

            )

        elif metric == "roc_auc":

            try:

                values.append(

                    roc_auc_score(

                        yt,

                        sc

                    )

                )

            except ValueError:

                continue

    alpha = (

        100 - confidence

    ) / 2

    return {

        "mean": float(

            np.mean(values)

        ),

        "lower": float(

            np.percentile(

                values,

                alpha

            )

        ),

        "upper": float(

            np.percentile(

                values,

                100-alpha

            )

        )

    }


def mcnemar_test(

    prediction_a,

    prediction_b,

    y_true

):

    both_correct = np.sum(

        (

            prediction_a == y_true

        )

        &

        (

            prediction_b == y_true

        )

    )

    a_correct = np.sum(

        (

            prediction_a == y_true

        )

        &

        (

            prediction_b != y_true

        )

    )

    b_correct = np.sum(

        (

            prediction_a != y_true

        )

        &

        (

            prediction_b == y_true

        )

    )

    both_wrong = np.sum(

        (

            prediction_a != y_true

        )

        &

        (

            prediction_b != y_true

        )

    )

    table = [

        [

            both_correct,

            a_correct

        ],

        [

            b_correct,

            both_wrong

        ]

    ]

    result = mcnemar(

        table,

        exact=False,

        correction=True

    )

    return {

        "statistic":

            float(

                result.statistic

            ),

        "p_value":

            float(

                result.pvalue

            )

    }


def wilcoxon_test(

    scores_a,

    scores_b

):

    statistic, p = wilcoxon(

        scores_a,

        scores_b

    )

    return {

        "statistic":

            float(

                statistic

            ),

        "p_value":

            float(

                p

            )

    }