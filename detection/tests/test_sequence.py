"""
Sequence generation tests.
"""

from detection.loader import load_datasets

from detection.feature_engineering import (

    prepare_training_features

)

from detection.sequence import (

    build_sequences

)

from detection.config import *


def test_sequence_builder():

    baseline, _, _, _, _ = (

        load_datasets(

            BASELINE_DATASET,

            BASELINE_CSV,

            ATTACKED_DATASET,

            ATTACKED_CSV,

            GROUND_TRUTH

        )

    )

    _, X = (

        prepare_training_features(

            baseline

        )

    )

    sequences, index = (

        build_sequences(

            X

        )

    )

    assert sequences.ndim == 3

    assert len(index) == len(sequences)

    assert sequences.shape[1] == LSTM_SEQUENCE_LENGTH

    assert sequences.shape[2] == len(X.columns)