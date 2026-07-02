"""
Isolation Forest tests.
"""

import numpy as np

from detection.loader import load_datasets

from detection.feature_engineering import (

    prepare_training_features,

    prepare_testing_features

)

from detection.models import (

    MODEL_REGISTRY

)

from detection.config import *


def test_isolation_forest():

    baseline, attacked, _, _, _ = (

        load_datasets(

            BASELINE_DATASET,

            BASELINE_CSV,

            ATTACKED_DATASET,

            ATTACKED_CSV,

            GROUND_TRUTH

        )

    )

    _, X_train = (

        prepare_training_features(

            baseline

        )

    )

    _, X_test = (

        prepare_testing_features(

            attacked

        )

    )

    detector = (

        MODEL_REGISTRY

        [

            "Isolation Forest"

        ]()

    )

    detector.fit(

        X_train

    )

    result = detector.predict(

        X_test

    )

    assert len(

        result.prediction

    ) == len(

        X_test

    )

    assert len(

        result.score

    ) == len(

        X_test

    )

    assert np.isin(

        result.prediction,

        [0, 1]

    ).all()

    assert np.isfinite(

        result.score

    ).all()

    assert result.model_name == "Isolation Forest"