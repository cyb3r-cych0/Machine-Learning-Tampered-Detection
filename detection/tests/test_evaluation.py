"""
Evaluation tests.
"""

from detection.loader import load_datasets

from detection.feature_engineering import (
    prepare_training_features,
    prepare_testing_features
)

from detection.models import MODEL_REGISTRY

from detection.evaluation import DetectionEvaluator

from detection.config import *


def test_evaluation():

    baseline, attacked, ground_truth, _, _ = (

        load_datasets(

            BASELINE_DATASET,

            BASELINE_CSV,

            ATTACKED_DATASET,

            ATTACKED_CSV,

            GROUND_TRUTH

        )

    )

    baseline = baseline.iloc[:TEST_SAMPLE_SIZE].copy()

    attacked = attacked.iloc[:TEST_SAMPLE_SIZE].copy()

    _, X_train = prepare_training_features(

        baseline

    )

    _, X_test = prepare_testing_features(

        attacked

    )

    detector = MODEL_REGISTRY[
        "Isolation Forest"
    ]()

    detector.fit(
        X_train
    )

    result = detector.predict(
        X_test
    )

    evaluator = DetectionEvaluator(

        attacked,

        ground_truth

    )

    report = evaluator.evaluate_detector(

        "Isolation Forest",

        result

    )

    assert "metrics" in report

    assert "confidence_intervals" in report

    assert "attack_recall" in report

    assert "country_recall" in report