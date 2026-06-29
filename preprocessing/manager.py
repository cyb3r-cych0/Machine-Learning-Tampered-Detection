"""
preprocessing/manager.py
"""

import time

from .config import (

    INPUT_DATASET,

    CSV_FALLBACK,

    OUTPUT_DIR,

    LOG_DIR

)

from .loader import load_dataset

from .pipeline import preprocess

from .logger import setup_logger


def run():

    logger = setup_logger(
        LOG_DIR
    )

    start = time.perf_counter()

    logger.info(
        "Preprocessing started."
    )

    df, source = load_dataset(

        INPUT_DATASET,

        CSV_FALLBACK

    )

    logger.info(

        f"Dataset loaded ({source})"

    )

    logger.info(

        f"Rows : {len(df):,}"

    )

    logger.info(

        f"Columns : {len(df.columns)}"

    )

    baseline = preprocess(

        df,

        OUTPUT_DIR

    )

    elapsed = (

        time.perf_counter()

        - start

    )

    logger.info(

        f"Rows after preprocessing : {len(baseline):,}"

    )

    logger.info(

        f"Runtime : {elapsed:.2f} seconds"

    )

    logger.info(

        "Baseline dataset exported successfully."

    )

    logger.info(

        "Preprocessing completed."

    )