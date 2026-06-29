"""
Centralized logger for the preprocessing package.
"""

from pathlib import Path
import logging


LOGGER_NAME = "Preprocessing"


def setup_logger(log_directory: Path):

    log_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    logger = logging.getLogger(
        LOGGER_NAME
    )

    logger.setLevel(logging.INFO)

    logger.handlers.clear()

    formatter = logging.Formatter(

        "%(asctime)s | %(levelname)s | %(message)s",

        "%Y-%m-%d %H:%M:%S"

    )

    # -----------------------------
    # Console
    # -----------------------------

    console = logging.StreamHandler()

    console.setFormatter(
        formatter
    )

    # -----------------------------
    # File
    # -----------------------------

    from logging.handlers import RotatingFileHandler

    logfile = RotatingFileHandler(
        log_directory / "preprocessing.log",
        maxBytes=5_000_000,
        backupCount=5,
        encoding="utf-8"
    )

    logfile.setFormatter(
        formatter
    )

    logger.addHandler(console)

    logger.addHandler(logfile)

    logger.propagate = False

    return logger