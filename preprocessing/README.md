# Preprocessing

The `preprocessing` package turns the combined EDA dataset into a cleaned,
feature-enriched baseline dataset for environmental cybersecurity analysis.

It validates the input schema, removes invalid observations, adds integrity and
quality flags, generates temporal model features, and exports the baseline
dataset with a preprocessing report, provenance manifest, and run log.

## Pipeline

```text
combined_dataset.parquet or combined_dataset.csv
    -> load_dataset
    -> validate_dataset
    -> clean_dataset
    -> integrity_assessment
    -> temporal_features
    -> quality_assessment
    -> export_dataset
    -> export_report
    -> build_manifest and save_manifest
```

The main entry point is:

```powershell
python -m preprocessing.run_preprocessing
```

or:

```powershell
python preprocessing/run_preprocessing.py
```

## Inputs

Configured in `config.py`:

```text
outputs/continental/combined_dataset.parquet
outputs/continental/combined_dataset.csv
```

`loader.py` prefers the parquet file when it exists and falls back to CSV when
CSV is the available source. The expected canonical columns are:

```text
country
location_id
sensor_id
timestamp_utc
value
```

## Outputs

Configured output directory:

```text
outputs/preprocessing/
```

Generated files:

```text
baseline_dataset.csv
baseline_dataset.parquet
preprocessing_report.json
dataset_manifest.json
```

Run logs are written separately under:

```text
outputs/preprocessing/logs/preprocessing.log
outputs/preprocessing/logs/preprocessing.log.1
outputs/preprocessing/logs/preprocessing.log.2
```

The log file rotates at 5 MB and keeps up to 5 backups.

The exported baseline preserves sensor-level observations and adds derived
features and flags. It does not aggregate the dataset into continental or
country-level means.

## Modules

### `config.py`

Defines input paths, output paths, and preprocessing constants:

- input parquet and CSV fallback paths
- preprocessing output directory
- cleaning options
- integrity thresholds
- rolling and EMA windows
- export settings

### `loader.py`

Loads the canonical combined dataset.

- reads parquet with `pyarrow` when available
- falls back to CSV when parquet support is unavailable and the CSV exists
- normalizes `timestamp_utc` to timezone-aware datetimes
- returns both the dataframe and the source name

### `validation.py`

Validates the input dataset without modifying it.

Checks:

- required columns
- timestamp datatype
- numeric PM2.5 values
- empty dataset
- missing timestamps
- missing PM2.5 values
- missing country, location, or sensor identifiers
- duplicate row count
- timestamp ordering

Returns a validation report with row counts, column counts, unique entity
counts, duplicate count, and sorted status.

### `cleaning.py`

Performs deterministic row-level cleaning.

Steps:

- converts timestamps with invalid values coerced to missing
- removes duplicate sensor timestamp observations
- removes rows missing `timestamp_utc` or `value`
- removes negative PM2.5 values
- sorts by country, location, sensor, and timestamp
- resets the index
- inserts `observation_id` as the first column
- initializes attack metadata columns:
  - `is_attacked`
  - `attack_type`
  - `attack_magnitude`

Returns the cleaned dataframe and a cleaning report.

`observation_id` gives each cleaned baseline row a stable identifier for later
attack simulation, detection outputs, explanations, and experiment debugging.

The attack metadata columns mark the exported baseline as clean by default:

- `is_attacked = False`
- `attack_type = "None"`
- `attack_magnitude = 0.0`

The cleaning report includes:

- `duplicates_removed`
- `missing_removed`
- `negative_removed`
- `sorted_after_cleaning`
- `rows_after_cleaning`

### `integrity.py`

Adds sensor integrity flags without removing observations.

Detects:

- time gaps per sensor
- flatlines using rolling standard deviation
- spikes using sensor-specific z-scores
- duplicate timestamps
- constant measurement sequences

Returns the augmented dataframe and an integrity summary report.

### `temporal.py`

Adds deterministic temporal and time-series features.

Features:

- calendar fields: hour, day, weekday, week, month, quarter, year, day of year
- weekend indicator
- cyclic hour and month encodings
- lag features: 1, 3, 6, and 12 observations
- rolling mean, standard deviation, and median
- exponential moving average
- first difference
- rate of change

### `quality.py`

Creates quality flags and a composite quality score.

Flags:

- `is_missing`
- `is_duplicate`
- `is_outlier`
- `is_gap`
- `is_flatline`
- `constant_sequence`

The quality score starts at 100 and subtracts 20 points for each integrity or
quality issue. Scores are clipped to the range 0 to 100.

Quality classes:

- `Excellent`: score >= 90
- `Good`: score >= 70
- `Fair`: score >= 50
- `Poor`: score < 50

Returns the augmented dataframe and a quality report.

### `export.py`

Writes the final baseline dataset and report.

- `baseline_dataset.csv`
- `baseline_dataset.parquet`
- `preprocessing_report.json`

### `manifest.py`

Writes a provenance manifest for each preprocessing run.

The manifest includes:

- generation timestamp
- row, column, country, station, and sensor counts
- source label currently set by the pipeline as `EDA`
- validation, cleaning, integrity, and quality reports
- SHA-256 hash of the final dataframe

Output file:

- `dataset_manifest.json`

### `logger.py`

Configures preprocessing logging.

Logs are written to both the console and:

- `outputs/preprocessing/logs/preprocessing.log`

The logger uses `RotatingFileHandler` with:

- maximum log file size: 5 MB
- backup count: 5
- log format: timestamp, level, message

### `pipeline.py`

Orchestrates validation, cleaning, integrity assessment, temporal feature
generation, quality assessment, export, and manifest generation.

### `manager.py`

Loads the configured input dataset, sets up logging, runs the full pipeline,
and logs row counts, source information, runtime, and completion status.

### `run_preprocessing.py`

Command-line entry point for running the preprocessing package.

## Tests

The package is tested from the top-level `tests/` directory:

```text
tests/test_validation.py
tests/test_cleaning.py
tests/test_integrity.py
tests/test_temporal.py
tests/test_quality.py
```

Run all tests from the repository root:

```powershell
pytest
```

or:

```powershell
python -m pytest
```

## Package Layout

```text
preprocessing/
    __init__.py
    config.py
    loader.py
    validation.py
    cleaning.py
    integrity.py
    temporal.py
    quality.py
    export.py
    manifest.py
    logger.py
    pipeline.py
    manager.py
    run_preprocessing.py
```
