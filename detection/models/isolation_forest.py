"""
detection/models/isolation_forest.py
"""

from sklearn.ensemble import IsolationForest
from detection.types import DetectorResult

from detection.config import (

    RANDOM_SEED,

    IF_N_ESTIMATORS,

    IF_CONTAMINATION,

    IF_MAX_SAMPLES,

    IF_BOOTSTRAP

)


class IsolationForestDetector:

    def __init__(self):

        self.model = IsolationForest(

            n_estimators=IF_N_ESTIMATORS,

            contamination=IF_CONTAMINATION,

            max_samples=IF_MAX_SAMPLES,

            bootstrap=IF_BOOTSTRAP,

            random_state=RANDOM_SEED,

            n_jobs=-1

        )

    def fit(

        self,

        X_train

    ):

        self.model.fit(

            X_train

        )

        return self

    def predict(

        self,

        X_test

    ):

        score = -self.model.score_samples(

            X_test

        )

        prediction = (

            self.model.predict(

                X_test

            ) == -1

        ).astype(int)

        threshold = (

            score[prediction == 1].min()

            if prediction.any()

            else None

        )

        return DetectorResult(

            model_name="Isolation Forest",

            prediction=prediction,

            score=score,

            threshold=threshold,

            model=self.model

        )