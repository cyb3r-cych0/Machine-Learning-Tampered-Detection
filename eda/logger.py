"""
Centralized logging for the EDA package.
"""

import logging
from pathlib import Path


def setup_logger(log_dir: Path):

    log_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    logger = logging.getLogger("EDA")

    logger.setLevel(logging.INFO)

    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    from logging.handlers import RotatingFileHandler

    log_file = RotatingFileHandler(
        log_dir / "eda.log",
        maxBytes=5_000_000,
        backupCount=5,
        encoding="utf-8"
    )

    log_file.setFormatter(formatter)

    console_handler = logging.StreamHandler()

    console_handler.setFormatter(formatter)

    logger.addHandler(log_file)

    logger.addHandler(console_handler)

    return logger