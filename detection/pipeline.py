"""
detection/pipeline.py
"""

from detection.models import MODEL_REGISTRY

from detection.feature_engineering import (
    prepare_training_features,
    prepare_testing_features
)

from detection.sequence import build_sequences


def run_detector(

    detector_name,

    baseline,

    attacked

):

    detector = MODEL_REGISTRY[detector_name]()

    # ----------------------------------------------------
    # Historical Rolling Z-Score
    # ----------------------------------------------------

    if detector_name == "Rolling Z-Score":

        detector.fit(baseline)

        result = detector.predict(attacked)

        return result

    # ----------------------------------------------------
    # Isolation Forest
    # ----------------------------------------------------

    if detector_name == "Isolation Forest":

        _, X_train = prepare_training_features(baseline)

        _, X_test = prepare_testing_features(attacked)

        detector.fit(X_train)

        result = detector.predict(X_test)

        return result

    # ----------------------------------------------------
    # LSTM
    # ----------------------------------------------------

    _, X_train = prepare_training_features(
        baseline
    )

    _, X_test = prepare_testing_features(
        attacked
    )

    X_train, _ = build_sequences(
        X_train
    )

    detector.fit(
        X_train
    )

    result = detector.predict_from_features(
        X_test
    )

    return result