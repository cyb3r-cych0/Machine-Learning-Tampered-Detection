"""
Tests for LSTM Autoencoder.
"""

from detection.loader import load_datasets

from detection.feature_engineering import (

    prepare_training_features,

    prepare_testing_features

)

from detection.sequence import (

    build_sequences

)

from detection.models import (

    MODEL_REGISTRY

)

from detection.config import *


def test_lstm_autoencoder():

    baseline, attacked, _, _, _ = (

        load_datasets(

            BASELINE_DATASET,

            BASELINE_CSV,

            ATTACKED_DATASET,

            ATTACKED_CSV,

            GROUND_TRUTH

        )

    )

    _, X_train = prepare_training_features(

        baseline

    )

    _, X_test = prepare_testing_features(

        attacked

    )

    from detection.config import TEST_SAMPLE_SIZE

    train_seq, _ = build_sequences(

        X_train.iloc[:TEST_SAMPLE_SIZE]

    )

    test_seq, _ = build_sequences(

        X_test.iloc[:TEST_SAMPLE_SIZE]

    )

    detector = (

        MODEL_REGISTRY

        [

            "LSTM Autoencoder"

        ]()

    )

    detector.fit(

        train_seq,

        epochs=LSTM_TEST_EPOCHS

    )

    result = detector.predict(

        test_seq

    )

    assert len(

        result.prediction

    ) == len(

        test_seq

    )

    assert len(

        result.score

    ) == len(

        test_seq

    )

    assert result.threshold > 0

    assert result.model_name == "LSTM Autoencoder"