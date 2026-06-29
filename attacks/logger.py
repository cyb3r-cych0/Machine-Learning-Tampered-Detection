"""
Centralised Attacks Logger
"""

import logging

from logging.handlers import RotatingFileHandler


LOGGER = "AttackSimulation"


def setup_logger(log_dir):

    log_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    logger = logging.getLogger(
        LOGGER
    )

    logger.handlers.clear()

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(

        "%(asctime)s | %(levelname)-4s | %(message)s"

    )

    console = logging.StreamHandler()

    console.setFormatter(formatter)

    file = RotatingFileHandler(

        log_dir / "attacks.log",

        maxBytes=5_000_000,

        backupCount=5,

        encoding="utf-8"

    )

    file.setFormatter(
        formatter
    )

    logger.addHandler(console)

    logger.addHandler(file)

    logger.propagate = False

    return logger