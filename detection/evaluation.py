"""
detection/evaluation.py
"""

import pandas as pd

from detection.metrics import compute_metrics

from detection.statistics import (

    bootstrap_metric,

    mcnemar_test,

    wilcoxon_test

)


class DetectionEvaluator:

    def __init__(

        self,

        attacked_df,

        ground_truth

    ):

        self.attacked = attacked_df

        self.truth = ground_truth

    def evaluate_detector(

        self,

        detector_name,

        result

    ):

        df = self.attacked.copy()

        if len(result.prediction) != len(df):
            offset = len(df) - len(result.prediction)

            df = (

                df

                .iloc[offset:]

                .reset_index(drop=True)

            )

        y_true = df["attack_active"].astype(int).to_numpy()

        df["prediction"] = result.prediction

        df["score"] = result.score

        df["ground_truth"] = (

            df["attack_active"]

            .astype(int)

        )

        df["ground_truth"] = y_true

        prediction = df["prediction"].to_numpy()

        score = df["score"].to_numpy()

        metrics = compute_metrics(

            y_true,

            prediction,

            score

        )

        ci = {

            metric:

            bootstrap_metric(

                y_true,

                prediction,

                score,

                metric

            )

            for metric in [

                "f1",

                "recall",

                "roc_auc"

            ]

        }

        attack_recall = (

            df

            .groupby(

                "attack_type"

            )

            .apply(

                lambda x:

                (

                    (

                        x["prediction"] == 1

                    )

                    &

                    (

                        x["ground_truth"] == 1

                    )

                ).sum()

                /

                max(

                    1,

                    (

                        x["ground_truth"] == 1

                    ).sum()

                )

            )

            .reset_index(

                name="recall"

            )

        )

        country_recall = (

            df

            .groupby(

                "country"

            )

            .apply(

                lambda x:

                (

                    (

                        x["prediction"] == 1

                    )

                    &

                    (

                        x["ground_truth"] == 1

                    )

                ).sum()

                /

                max(

                    1,

                    (

                        x["ground_truth"] == 1

                    ).sum()

                )

            )

            .reset_index(

                name="recall"

            )

        )

        return {

            "model":

                detector_name,

            "metrics":

                metrics,

            "confidence_intervals":

                ci,

            "attack_recall":

                attack_recall,

            "country_recall":

                country_recall,

            "prediction":

                prediction,

            "score":

                score

        }

    @staticmethod
    def compare(

        evaluation_a,

        evaluation_b,

        y_true

    ):

        return {

            "mcnemar":

                mcnemar_test(

                    evaluation_a["prediction"],

                    evaluation_b["prediction"],

                    y_true

                ),

            "wilcoxon":

                wilcoxon_test(

                    evaluation_a["score"],

                    evaluation_b["score"]

                )

        }