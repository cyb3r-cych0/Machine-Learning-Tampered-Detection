"""
detection/metrics.py
"""

import numpy as np

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    roc_auc_score,

    average_precision_score,

    confusion_matrix

)


def compute_metrics(

    y_true,

    prediction,

    score

):
    score = np.nan_to_num(

        score,

        nan=0.0,

        posinf=0.0,

        neginf=0.0

    )

    tn, fp, fn, tp = confusion_matrix(

        y_true,

        prediction,

        labels=[0, 1]

    ).ravel()

    fpr = (

        fp / (fp + tn)

        if (fp + tn) > 0

        else 0.0

    )

    return {

        "accuracy":

            accuracy_score(

                y_true,

                prediction

            ),

        "precision":

            precision_score(

                y_true,

                prediction,

                zero_division=0

            ),

        "recall":

            recall_score(

                y_true,

                prediction,

                zero_division=0

            ),

        "f1":

            f1_score(

                y_true,

                prediction,

                zero_division=0

            ),

        "roc_auc":

            roc_auc_score(

                y_true,

                score

            ),

        "pr_auc":

            average_precision_score(

                y_true,

                score

            ),

        "false_positive_rate":

            fpr,

        "tp": tp,

        "fp": fp,

        "tn": tn,

        "fn": fn

    }