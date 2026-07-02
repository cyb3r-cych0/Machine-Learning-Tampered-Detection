"""
Feature engineering tests.
"""

from detection.loader import load_datasets

from detection.feature_engineering import (

    prepare_training_features,

    prepare_testing_features,

    TABULAR_FEATURES

)

from detection.config import *


def test_feature_engineering():

    baseline, attacked, _, _, _ = (

        load_datasets(

            BASELINE_DATASET,

            BASELINE_CSV,

            ATTACKED_DATASET,

            ATTACKED_CSV,

            GROUND_TRUTH

        )

    )

    baseline, X_train = (

        prepare_training_features(

            baseline

        )

    )

    attacked, X_test = (

        prepare_testing_features(

            attacked

        )

    )

    assert list(

        X_train.columns

    ) == TABULAR_FEATURES

    assert list(

        X_test.columns

    ) == TABULAR_FEATURES