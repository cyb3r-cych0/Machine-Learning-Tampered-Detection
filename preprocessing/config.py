"""
Configuration for the preprocessing package.
"""

from pathlib import Path

# ----------------------------------------------------
# Input
# ----------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DATASET = (
    PROJECT_ROOT /
    "outputs" /
    "continental" /
    "combined_dataset.parquet"
)

CSV_FALLBACK = (
    PROJECT_ROOT /
    "outputs" /
    "continental" /
    "combined_dataset.csv"
)

# ----------------------------------------------------
# Output
# ----------------------------------------------------

OUTPUT_DIR = (
    PROJECT_ROOT /
    "outputs" /
    "preprocessing"
)

LOG_DIR = (
    PROJECT_ROOT /
    "outputs" /
    "preprocessing" /
    "logs"
)

REPORT_DIR = (
    PROJECT_ROOT /
    "outputs" /
    "reports"
)

# ----------------------------------------------------
# Cleaning
# ----------------------------------------------------

REMOVE_NEGATIVE_PM25 = True

REMOVE_DUPLICATES = True

DROP_MISSING_VALUES = True

SORT_BY_TIMESTAMP = True

# ----------------------------------------------------
# Integrity Checks
# ----------------------------------------------------

MAX_TIME_GAP_HOURS = 24

FLATLINE_WINDOW = 24

SPIKE_ZSCORE = 4.0

# ----------------------------------------------------
# Rolling Windows
# ----------------------------------------------------

ROLLING_WINDOW = 24

EMA_WINDOW = 24

# ----------------------------------------------------
# Export
# ----------------------------------------------------

EXPORT_CSV = True

EXPORT_PARQUET = True