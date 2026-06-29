# EDA

The `eda` package runs exploratory data analysis on preprocessed country-level
PM2.5 CSV files. It discovers input datasets, validates and loads each country,
adds basic time-series features, computes descriptive statistics, writes country
reports, generates plots, and exports continental summary artifacts.

## Pipeline

```text
data/preprocessed/*.csv
    -> discover_datasets
    -> process_country
        -> load_dataset
        -> validate_dataset
        -> engineer_features
        -> calculate_statistics
        -> save_country_summary
        -> save_country_metadata
        -> generate_country_plots
    -> save_continental_outputs
    -> generate_continental_plots
    -> eda_execution_summary.json
```

The main entry point from the repository root is:

```powershell
python run_eda.py
```

## Inputs

Configured in `config.py`:

```text
data/preprocessed/*.csv
```

Each input CSV is expected to contain PM2.5 observations with at least the
columns used by the package:

```text
country
location_id
sensor_id
timestamp_utc
value
```

Some plots also use `location_name` when it is available.

## Outputs

Country-level outputs are written under:

```text
outputs/country_reports/<country>/
```

Generated country files:

```text
summary.csv
metadata.json
plots/time_series.png
plots/rolling_statistics.png
plots/distribution.png
plots/seasonality.png
plots/station_comparison.png
plots/sensor_availability.png
plots/outliers.png
plots/publication_summary.png
```

Continental outputs are written under:

```text
outputs/continental/
```

Generated continental files:

```text
country_summary.csv
combined_dataset.csv
combined_dataset.parquet
eda_execution_summary.json
plots/dataset_size.png
plots/mean_pm25.png
plots/station_count.png
plots/sensor_count.png
plots/missing.png
plots/coverage.png
plots/country_boxplots.png
plots/country_distribution.png
```

## Modules

### `config.py`

Defines package paths and plotting/statistical constants:

- input directory
- output directory
- country report directory
- continental output directory
- rolling window
- z-score threshold
- monthly resampling frequency
- plot export flags

### `loader.py`

Finds and loads country CSV datasets.

- `discover_datasets()` returns sorted CSV paths from the input directory
- `load_dataset()` reads a CSV, converts `timestamp_utc` to UTC datetimes, and
  sorts the data by timestamp

### `validator.py`

Builds a lightweight dataset validation summary.

The report includes:

- row count
- missing value count
- duplicate row count
- station count
- sensor count
- start and end timestamps
- country name

### `features.py`

Adds exploratory time features and creates aggregated time series.

Adds to the country dataframe:

- hour
- day of week
- month
- year

Creates `country_ts`, a timestamp-level PM2.5 mean time series with:

- rolling mean
- rolling standard deviation
- z-score
- z-score outlier flag

Creates `monthly_ts`, a month-level time series with:

- monthly mean
- monthly standard deviation
- interpolated mean for plotting
- interpolated standard deviation for plotting

### `statistics.py`

Computes descriptive statistics for each country.

The report includes:

- dataset size and missingness
- station and sensor counts
- temporal coverage
- mean, median, standard deviation, variance, minimum, maximum, and range
- quartiles and IQR
- p90, p95, and p99
- skewness and kurtosis
- z-score and IQR outlier counts
- negative, zero, and NaN value counts
- missing percentage

### `report.py`

Writes country and continental report artifacts.

Country outputs:

- `summary.csv`
- `metadata.json`

Continental outputs:

- `country_summary.csv`
- `combined_dataset.csv`
- `combined_dataset.parquet`

### `plotting/country.py`

Generates country-level visualizations:

- PM2.5 time series
- rolling statistics
- value distribution
- hourly seasonality
- station comparison
- sensor availability
- outlier plot
- publication summary figure

### `plotting/continental.py`

Generates cross-country visualizations:

- dataset size by country
- mean PM2.5 by country
- station count by country
- sensor count by country
- missingness
- temporal coverage
- country boxplots
- country distribution comparison

### `plotting/utils.py`

Provides the shared `save_figure()` helper used by country and continental
plotting functions.

### `verify.py`

Checks that expected country-level report and plot files were produced for each
processed country.

### `pipeline.py`

Runs the full country-level EDA workflow for one CSV file and returns:

- the processed dataframe with engineered features
- the country statistics report

### `manager.py`

Runs the package across all discovered country CSV files.

Responsibilities:

- set up logging
- discover datasets
- process each country
- verify country outputs
- concatenate processed country data
- save continental reports and combined datasets
- generate continental plots
- write `eda_execution_summary.json`

## Package Layout

```text
eda/
    __init__.py
    config.py
    loader.py
    validator.py
    features.py
    statistics.py
    report.py
    verify.py
    logger.py
    pipeline.py
    manager.py
    plotting/
        __init__.py
        utils.py
        country.py
        continental.py
```
