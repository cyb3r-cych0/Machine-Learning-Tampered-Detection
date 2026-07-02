"""
detection/config.py
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ==========================================================
# INPUTS
# ==========================================================

BASELINE_DATASET = (
    PROJECT_ROOT /
    "outputs" /
    "preprocessing" /
    "baseline_dataset.parquet"
)

BASELINE_CSV = (
    PROJECT_ROOT /
    "outputs" /
    "preprocessing" /
    "baseline_dataset.csv"
)

ATTACKED_DATASET = (
    PROJECT_ROOT /
    "outputs" /
    "attacks" /
    "attacked_dataset.parquet"
)

ATTACKED_CSV = (
    PROJECT_ROOT /
    "outputs" /
    "attacks" /
    "attacked_dataset.csv"
)

GROUND_TRUTH = (
    PROJECT_ROOT /
    "outputs" /
    "attacks" /
    "ground_truth.csv"
)

# ==========================================================
# OUTPUT
# ==========================================================

OUTPUT_DIR = (
    PROJECT_ROOT /
    "outputs" /
    "detection"
)

LOG_DIR = OUTPUT_DIR / "logs"

# ==========================================================
# RANDOMNESS
# ==========================================================

RANDOM_SEED = 42

# ==========================================================
# TRAIN / TEST
# ==========================================================

TRAIN_ON_BASELINE_ONLY = True

TEMPORAL_SPLIT = False

TRAIN_END_DATE = None

TEST_START_DATE = None

# ==========================================================
# ISOLATION FOREST
# ==========================================================

IF_N_ESTIMATORS = 200

IF_CONTAMINATION = "auto"

IF_MAX_SAMPLES = "auto"

IF_BOOTSTRAP = False

# ==========================================================
# ROLLING Z-SCORE
# ==========================================================

ROLLING_WINDOW = 24

ZSCORE_THRESHOLD = 3.0

# ==========================================================
# LSTM AUTOENCODER
# ==========================================================

LSTM_SEQUENCE_LENGTH = 24

LSTM_EPOCHS = 30

LSTM_TEST_EPOCHS = 2

LSTM_BATCH_SIZE = 256

LSTM_LATENT_DIM = 16

LSTM_THRESHOLD_PERCENTILE = 95

# ==========================================================
# EXPERIMENTS
# ==========================================================

ENABLE_SENSITIVITY_ANALYSIS = True

WINDOW_SIZES = [24, 48, 72]

CONTAMINATION_LEVELS = [

    0.01,

    0.05,

    0.10

]

# ==========================================================
# TESTING
# ==========================================================

TEST_SAMPLE_SIZE = 5000
