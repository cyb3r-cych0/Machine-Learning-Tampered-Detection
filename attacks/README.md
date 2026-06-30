# Attacks

The `attacks` package turns the clean preprocessing baseline into an attacked
environmental sensor dataset for cybersecurity experiments.

It creates a deterministic attack campaign, injects adversarial PM2.5
manipulations, validates the attacked output, generates ground-truth labels,
exports attacked datasets, writes analytics tables, and records a provenance
manifest and run log.

## Pipeline

```text
baseline_dataset.parquet or baseline_dataset.csv
    -> load_dataset
    -> validate
    -> create_campaign
    -> apply_constant_bias
    -> apply_gradual_drift
    -> apply_spike_suppression
    -> apply_random_stealth
    -> create_ground_truth
    -> validate_attacked_dataset
    -> export_dataset
    -> export_report
    -> create_manifest and save_manifest
    -> generate_attack_analytics
```

The main entry point is:

```powershell
python run_attacks.py
```

Programmatic use:

```python
from attacks.manager import run

run()
```

## Inputs

Configured in `config.py`:

```text
outputs/preprocessing/baseline_dataset.parquet
outputs/preprocessing/baseline_dataset.csv
```

`loader.py` prefers the parquet file when it exists and falls back to CSV when
CSV is the available source. The expected canonical columns are:

```text
observation_id
country
location_id
sensor_id
timestamp_utc
value
```

The input dataset is expected to come from the `preprocessing` package.

## Outputs

Configured output directory:

```text
outputs/attacks/
```

Generated files:

```text
attacked_dataset.csv
attacked_dataset.parquet
ground_truth.csv
campaign.csv
attack_report.json
attack_manifest.json
```

Generated analytics:

```text
outputs/attacks/analytics/attack_summary.csv
outputs/attacks/analytics/attack_types.csv
outputs/attacks/analytics/attack_by_country.csv
outputs/attacks/analytics/attack_by_sensor.csv
outputs/attacks/analytics/attack_duration.csv
outputs/attacks/analytics/attack_strength.csv
```

Run logs are written separately under:

```text
outputs/attacks/logs/attacks.log
```

The log file rotates at 5 MB and keeps up to 5 backups.

The attacked dataset preserves sensor-level observations and adds attack
metadata. It does not aggregate the dataset into continental or country-level
means.

## Attack Metadata

Attack modules initialize and maintain the following columns:

```text
original_value
attacked_value
attack_id
attack_type
attack_active
attack_strength
attack_start
attack_end
```

Clean observations use:

```text
attack_id = -1
attack_type = "None"
attack_active = False
attack_strength = 0.0
```

`original_value` stores the pre-attack value. `attacked_value` stores the
modified value for attacked observations and is kept consistent with `value`.

## Attack Types

### Constant Bias

Adds a fixed positive offset to all observations in the selected attack window.

Configured by:

```text
BIAS_MIN
BIAS_MAX
```

The sampled bias is stored as `attack_strength`.

### Gradual Drift

Adds an increasing drift across the selected attack window.

Configured by:

```text
DRIFT_STEP
DRIFT_MAX
```

The first attacked row receives `DRIFT_STEP`, subsequent rows increase by the
same step, and the drift is clipped at `DRIFT_MAX`. The maximum applied drift is
stored as `attack_strength`.

### Spike Suppression

Suppresses high PM2.5 peaks inside the selected attack window.

Configured by:

```text
SUPPRESSION_FACTOR
```

Rows above the 90th percentile of the selected window are multiplied by the
suppression factor. The suppression factor is stored as `attack_strength`.

### Random Stealth

Adds small normally distributed noise to the selected attack window.

Configured by:

```text
STEALTH_STD
```

The standard deviation of the sampled noise is stored as `attack_strength`.

## Modules

### `config.py`

Defines input paths, output paths, and attack constants:

- baseline input parquet and CSV fallback paths
- attack output and log directories
- random seed
- default attack coverage
- attack window sizing
- attack-specific parameters

### `loader.py`

Loads the preprocessing baseline dataset.

- reads parquet with `pyarrow` when available
- falls back to CSV when parquet is unavailable
- normalizes `timestamp_utc` to timezone-aware datetimes
- returns both the dataframe and the source name

### `validation.py`

Validates both input and attacked output.

Input validation checks:

- required canonical columns

Attacked dataset validation checks:

- unique campaign attack IDs
- valid attack types on attacked rows
- valid attack windows
- modified attacked values
- consistency between `value` and `attacked_value`
- campaign ID integrity
- campaign effectiveness
- duplicate attacked observations

### `campaign.py`

Creates the attack campaign.

- initializes campaign metadata defaults
- selects a deterministic subset of sensors using `RANDOM_SEED`
- assigns one of the supported attack types to each selected sensor
- creates one campaign row per attacked sensor

Campaign columns:

```text
random_seed
attack_id
country
location_id
sensor_id
attack_type
```

### `utils.py`

Provides shared attack helpers.

- initializes attack metadata columns
- filters campaign rows by attack type
- selects deterministic sensor-level attack windows
- applies attack metadata to modified rows
- finalizes attacked values when needed

### `attacks/constant_bias.py`

Implements constant bias injection.

### `attacks/gradual_drift.py`

Implements gradual drift injection.

### `attacks/spike_suppression.py`

Implements spike suppression.

### `attacks/random_stealth.py`

Implements random stealth perturbation.

### `attacks/__init__.py`

Defines `ATTACK_REGISTRY`, the ordered mapping used by the attack pipeline:

```text
Constant Bias
Gradual Drift
Spike Suppression
Random Stealth
```

### `pipeline.py`

Creates the campaign and applies every registered attack module.

Returns:

```text
attacked
campaign
executed
```

### `labels.py`

Generates ground-truth labels for attacked observations.

The label output includes observation identity, attack metadata, original and
attacked values, and the value difference.

### `analytics.py`

Writes attack analytics tables.

Generated summaries include:

- total and attacked observations
- attack percentage
- campaign count
- countries and sensors affected
- attack type distribution
- country distribution
- sensor distribution
- attack duration
- attack strength

### `export.py`

Writes attacked datasets, ground-truth labels, campaign assignments, and the
attack report.

Output files:

- `attacked_dataset.csv`
- `attacked_dataset.parquet`
- `ground_truth.csv`
- `campaign.csv`
- `attack_report.json`

### `manifest.py`

Writes a provenance manifest for each attack simulation run.

The manifest includes:

- generation timestamp
- row count
- column count
- campaign count
- SHA-256 hash of the attacked dataframe

Output file:

- `attack_manifest.json`

### `logger.py`

Configures attack simulation logging.

Logs are written to both the console and:

```text
outputs/attacks/logs/attacks.log
```

The logger uses `RotatingFileHandler` with:

- maximum log file size: 5 MB
- backup count: 5
- log format: timestamp, level, message

### `manager.py`

Loads the configured baseline, sets up logging, runs the full attack pipeline,
validates the attacked output, exports all artifacts, generates analytics, and
logs run summary metrics.

## Tests

The package is tested from `attacks/tests/`:

```text
attacks/tests/test_campaign.py
attacks/tests/test_constant_bias.py
attacks/tests/test_gradual_drift.py
attacks/tests/test_random_stealth.py
attacks/tests/test_spike_suppression.py
```

Run all tests from the project root:

```powershell
pytest attacks/tests
```

or:

```powershell
python -m pytest attacks/tests
```

## Package Layout

```text
attacks/
    __init__.py
    analytics.py
    campaign.py
    config.py
    export.py
    labels.py
    loader.py
    logger.py
    manager.py
    manifest.py
    pipeline.py
    utils.py
    validation.py
    attacks/
        __init__.py
        constant_bias.py
        gradual_drift.py
        random_stealth.py
        spike_suppression.py
    tests/
        __init__.py
        test_campaign.py
        test_constant_bias.py
        test_gradual_drift.py
        test_random_stealth.py
        test_spike_suppression.py
```
