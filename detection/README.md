# Detection

The `detection` package trains anomaly detectors on the clean preprocessing
baseline and evaluates them against attacked PM2.5 observations from the attack
simulation stage.

It loads baseline, attacked, and ground-truth datasets, validates their schema,
runs each registered detector, computes evaluation metrics and confidence
intervals, compares detectors statistically, and exports per-detector reports,
comparison tables, a provenance manifest, and a run log.

## Pipeline

```text
baseline_dataset.parquet or baseline_dataset.csv
attacked_dataset.parquet or attacked_dataset.csv
ground_truth.csv
    -> load_datasets
    -> validate
    -> run_detector for each registered model
        -> Rolling Z-Score
        -> Isolation Forest
        -> LSTM Autoencoder
    -> evaluate_detector
    -> export_detector_results
    -> build_comparison_table
    -> pairwise_statistics
    -> export_comparison
    -> create_manifest and save_manifest
```

The main entry point from the repository root is:

```powershell
python run_detection.py
```

Programmatic use:

```python
from detection.manager import run

run()
```

## Inputs

Configured in `config.py`:

```text
outputs/preprocessing/baseline_dataset.parquet
outputs/preprocessing/baseline_dataset.csv
outputs/attacks/attacked_dataset.parquet
outputs/attacks/attacked_dataset.csv
outputs/attacks/ground_truth.csv
```

`loader.py` prefers parquet inputs when they exist and falls back to CSV.
Timestamps are normalized to UTC datetimes and each dataset is sorted by
`timestamp_utc`.

The baseline and attacked datasets must include:

```text
observation_id
sensor_id
timestamp_utc
value
```

The attacked dataset is also expected to preserve attack metadata from the
`attacks` package, especially:

```text
attack_active
attack_type
country
```

These columns are used for ground-truth evaluation, attack-type recall, and
country-level recall.

## Outputs

Configured output directory:

```text
outputs/detection/
```

Generated comparison and manifest files:

```text
detector_comparison.csv
pairwise_statistics.csv
detection_manifest.json
```

Per-detector outputs are written under normalized detector directories:

```text
outputs/detection/isolation_forest/
outputs/detection/rolling_z-score/
outputs/detection/lstm_autoencoder/
```

Each detector directory contains:

```text
metrics.csv
confidence_intervals.csv
attack_recall.csv
country_recall.csv
```

Run logs are written separately under:

```text
outputs/detection/logs/detection.log
```

The log file rotates at 5 MB and keeps up to 5 backups.

## Detectors

### Rolling Z-Score

Fits a historical sensor-level reference from the clean baseline. For each
sensor, the detector stores the final rolling mean and rolling standard
deviation from the configured baseline window.

At prediction time, observations are scored by absolute z-score and flagged
when the score is greater than or equal to `ZSCORE_THRESHOLD`.

Configured by:

```text
ROLLING_WINDOW
ZSCORE_THRESHOLD
```

### Isolation Forest

Trains a scikit-learn `IsolationForest` on tabular features engineered from the
clean baseline and predicts anomalies on engineered features from the attacked
dataset.

Configured by:

```text
IF_N_ESTIMATORS
IF_CONTAMINATION
IF_MAX_SAMPLES
IF_BOOTSTRAP
RANDOM_SEED
```

### LSTM Autoencoder

Builds fixed-length feature sequences from the clean baseline, trains a Keras
LSTM autoencoder to reconstruct those sequences, and flags attacked sequences
whose reconstruction error exceeds the configured percentile threshold.

Configured by:

```text
LSTM_SEQUENCE_LENGTH
LSTM_EPOCHS
LSTM_BATCH_SIZE
LSTM_LATENT_DIM
LSTM_THRESHOLD_PERCENTILE
```

Because sequence models emit one prediction per sequence, evaluation aligns the
attacked dataframe to the prediction length before scoring.

## Engineered Features

`feature_engineering.py` builds deterministic sensor-level features for the
machine-learning detectors.

Tabular features:

```text
value
hour_sin
hour_cos
month_sin
month_cos
lag_1
lag_3
lag_6
lag_12
difference
rate_of_change
time_gap_hours
```

Statistical features:

```text
rolling_mean
rolling_std
rolling_median
ema
zscore
flatline_std
```

Missing, infinite, and undefined feature values are forward-filled,
back-filled, and finally filled with zero.

## Evaluation

`DetectionEvaluator` compares detector predictions with the attacked dataset's
`attack_active` column.

Computed metrics:

```text
accuracy
precision
recall
f1
roc_auc
pr_auc
false_positive_rate
tp
fp
tn
fn
```

The evaluator also exports:

- bootstrap confidence intervals for F1, recall, and ROC AUC
- recall grouped by attack type
- recall grouped by country

Detector comparison includes:

- aggregate detector metric table
- pairwise McNemar tests on detector correctness
- pairwise Wilcoxon tests on detector scores

## Modules

### `config.py`

Defines input paths, output paths, random seed, detector hyperparameters,
sequence settings, and sensitivity experiment settings.

### `loader.py`

Loads baseline and attacked datasets from parquet or CSV and loads attack
ground truth from CSV.

### `validation.py`

Validates required baseline and attacked columns, rejects duplicate
`observation_id` values, and ensures ground truth is not empty.

### `feature_engineering.py`

Builds time, lag, rolling, exponential moving average, difference, rate of
change, z-score, flatline, and time-gap features.

### `sequence.py`

Builds fixed-length arrays for sequence-based models.

### `models/rolling_zscore.py`

Implements the historical rolling z-score detector.

### `models/isolation_forest.py`

Implements the scikit-learn Isolation Forest detector.

### `models/lstm_autoencoder.py`

Implements the Keras LSTM autoencoder detector.

### `models/__init__.py`

Defines `MODEL_REGISTRY`, the ordered mapping used by the detection manager:

```text
Isolation Forest
Rolling Z-Score
LSTM Autoencoder
```

### `pipeline.py`

Dispatches detector-specific training and prediction logic.

### `evaluation.py`

Computes detector metrics, confidence intervals, attack-type recall, and
country-level recall.

### `metrics.py`

Computes standard binary classification metrics from ground-truth labels,
predictions, and anomaly scores.

### `statistics.py`

Provides bootstrap confidence intervals, McNemar tests, and Wilcoxon signed-rank
tests.

### `comparison.py`

Builds the detector comparison table and pairwise statistical test table.

### `export.py`

Writes per-detector metrics, confidence intervals, attack recall, country
recall, detector comparison, and pairwise statistics.

### `manifest.py`

Writes a provenance manifest for each detection run.

The manifest includes:

- generation timestamp
- baseline row count
- attacked row count
- detector names
- SHA-256 hash of the baseline dataframe
- SHA-256 hash of the attacked dataframe

Output file:

```text
detection_manifest.json
```

### `logger.py`

Configures detection logging.

Logs are written to both the console and:

```text
outputs/detection/logs/detection.log
```

The logger uses `RotatingFileHandler` with:

- maximum log file size: 5 MB
- backup count: 5
- log format: timestamp, level, message

### `manager.py`

Loads configured datasets, sets up logging, validates inputs, runs every
registered detector, evaluates results, exports outputs, creates the manifest,
and logs completion time.

### `experiment.py`

Defines the default experiment and optional sensitivity experiment definitions
for rolling-window and contamination settings.

## Tests

The package is tested from `detection/tests/`:

```text
detection/tests/test_comparison.py
detection/tests/test_evaluation.py
detection/tests/test_experiment.py
detection/tests/test_export.py
detection/tests/test_feature_engineering.py
detection/tests/test_isolation_forest.py
detection/tests/test_lstm_autoencoder.py
detection/tests/test_metrics.py
detection/tests/test_rolling_zscore.py
detection/tests/test_sequence.py
detection/tests/test_statistics.py
```

Run all tests from the project root:

```powershell
pytest detection/tests
```

or:

```powershell
python -m pytest detection/tests
```

## Package Layout

```text
detection/
    __init__.py
    comparison.py
    config.py
    evaluation.py
    experiment.py
    export.py
    feature_engineering.py
    loader.py
    logger.py
    manager.py
    manifest.py
    metrics.py
    pipeline.py
    sequence.py
    statistics.py
    types.py
    validation.py
    models/
        __init__.py
        isolation_forest.py
        lstm_autoencoder.py
        rolling_zscore.py
    tests/
        __init__.py
        test_comparison.py
        test_evaluation.py
        test_experiment.py
        test_export.py
        test_feature_engineering.py
        test_isolation_forest.py
        test_lstm_autoencoder.py
        test_metrics.py
        test_rolling_zscore.py
        test_sequence.py
        test_statistics.py
```
