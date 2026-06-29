"""
Configuration for attack simulation.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------
# Input
# --------------------------------------------------

INPUT_DATASET = (
    PROJECT_ROOT /
    "outputs" /
    "preprocessing" /
    "baseline_dataset.parquet"
)

CSV_FALLBACK = (
    PROJECT_ROOT /
    "outputs" /
    "preprocessing" /
    "baseline_dataset.csv"
)

# --------------------------------------------------
# Output
# --------------------------------------------------

OUTPUT_DIR = (
    PROJECT_ROOT /
    "outputs" /
    "attacks"
)

LOG_DIR = OUTPUT_DIR / "logs"

# --------------------------------------------------
# Randomness
# --------------------------------------------------

RANDOM_SEED = 42

# --------------------------------------------------
# Attack Coverage
# --------------------------------------------------

DEFAULT_ATTACK_PERCENTAGE = 0.10

ATTACK_WINDOW_FRACTION = 0.20

MIN_ATTACK_WINDOW = 24

# --------------------------------------------------
# Constant Bias
# --------------------------------------------------

BIAS_MIN = 5

BIAS_MAX = 30

# --------------------------------------------------
# Gradual Drift
# --------------------------------------------------

DRIFT_STEP = 0.25

DRIFT_MAX = 20

# --------------------------------------------------
# Spike Suppression
# --------------------------------------------------

SUPPRESSION_FACTOR = 0.40

# --------------------------------------------------
# Random Stealth
# --------------------------------------------------

STEALTH_STD = 1.0