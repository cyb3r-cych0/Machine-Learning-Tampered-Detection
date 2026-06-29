import time
from .config import *
from .loader import load_dataset
from .validation import (validate, validate_attacked_dataset)
from .pipeline import run_attack_pipeline
from .labels import create_ground_truth
from .analytics import generate_attack_analytics
from .logger import setup_logger
from .export import (
    export_dataset,
    export_report
)
from .manifest import (
    create_manifest,
    save_manifest
)


def run():

    logger = setup_logger(

        LOG_DIR

    )

    start = time.perf_counter()

    logger.info(

        "Attack simulation started."

    )

    df, source = load_dataset(

        INPUT_DATASET,

        CSV_FALLBACK

    )

    validate(df)

    logger.info(

        f"Baseline loaded ({source})"

    )

    attacked, campaign, executed = (

        run_attack_pipeline(df)

    )

    labels = create_ground_truth(

        attacked

    )

    validation_report = validate_attacked_dataset(

        attacked,

        campaign

    )

    logger.info("Attack validation")

    for key, value in validation_report.items():
        logger.info(

            f"{key}: {value}"

        )

    export_dataset(

        attacked,

        labels,

        campaign,

        OUTPUT_DIR

    )

    export_report(

        attacked,

        labels,

        campaign,

        OUTPUT_DIR

    )

    manifest = create_manifest(

        attacked,

        campaign

    )

    save_manifest(

        manifest,

        OUTPUT_DIR

    )

    generate_attack_analytics(

        attacked,

        campaign,

        OUTPUT_DIR

    )

    logger.info(

        "Attack analytics generated."

    )

    elapsed = (

        time.perf_counter()

        - start

    )

    logger.info("")

    logger.info("Executed attacks:")

    for attack in executed:

        logger.info(

            f"  • {attack}"

        )

    logger.info("")

    logger.info("========== Attack Summary ==========")

    logger.info(

        f"Baseline Rows     : {len(df):,}"

    )

    logger.info(

        f"Sensors           : {df['sensor_id'].nunique()}"

    )

    logger.info(

        f"Campaigns         : {len(campaign)}"

    )

    logger.info(

        f"Attack Coverage   : {100 * len(campaign) / df['sensor_id'].nunique():.2f}%"

    )

    logger.info(

        f"Attacked Rows     : {len(labels):,}"

    )

    logger.info(

        f"Modified Fraction : {100 * len(labels) / len(df):.2f}%"

    )

    logger.info(

        f"Runtime           : {elapsed:.2f} sec"

    )

    logger.info("====================================")

    logger.info(

        "Attack simulation completed."

    )