"""
detection/manager.py
"""

import time

from detection.config import *

from detection.loader import load_datasets

from detection.validation import validate

from detection.pipeline import run_detector

from detection.evaluation import DetectionEvaluator

from detection.comparison import (
    build_comparison_table,
    pairwise_statistics
)

from detection.export import (
    export_detector_results,
    export_comparison
)

from detection.manifest import (
    create_manifest,
    save_manifest
)

from detection.logger import setup_logger

from detection.models import MODEL_REGISTRY


def run():

    logger = setup_logger(LOG_DIR)

    start = time.perf_counter()

    baseline, attacked, truth, _, _ = load_datasets(

        BASELINE_DATASET,

        BASELINE_CSV,

        ATTACKED_DATASET,

        ATTACKED_CSV,

        GROUND_TRUTH

    )

    validate(

        baseline,

        attacked,

        truth

    )

    print(attacked.columns.tolist())

    # -----------------------------------------
    # Common evaluation dataset
    # -----------------------------------------

    evaluation_attacked = attacked.reset_index(drop=True)

    evaluation_truth = truth.reset_index(drop=True)

    reports = {}

    for detector in MODEL_REGISTRY:

        logger.info(

            f"Running {detector}"

        )

        result = run_detector(

            detector,

            baseline,

            evaluation_attacked

        )

        evaluator = DetectionEvaluator(

            evaluation_attacked,

            evaluation_truth

        )

        report = evaluator.evaluate_detector(

            detector,

            result

        )

        reports[detector] = report

        export_detector_results(

            report,

            OUTPUT_DIR

        )

    comparison = build_comparison_table(

        reports

    )

    min_length = min(

        len(report["prediction"])

        for report in reports.values()

    )

    statistics = pairwise_statistics(

        reports,

        evaluation_attacked["attack_active"]

        .astype(int)

        .to_numpy()[:min_length]

    )

    export_comparison(

        comparison,

        statistics,

        OUTPUT_DIR

    )

    manifest = create_manifest(

        baseline,

        evaluation_attacked,

        reports

    )

    save_manifest(

        manifest,

        OUTPUT_DIR

    )

    elapsed = (

        time.perf_counter()

        - start

    )

    logger.info(

        f"Completed in {elapsed:.2f}s"

    )