# Environmental Cybersecurity PM2.5 Analysis

This repository contains a Python research pipeline for collecting OpenAQ PM2.5 measurements, building environmental baselines, simulating sensor data tampering, detecting anomalous behavior, and producing evaluation figures for an environmental cybersecurity study.

The workflow is designed around reproducible CSV inputs and generated research artifacts. Raw data, processed data, logs, plots, draft papers, and reference papers are ignored by Git.

## Repository Contents

| Path | Purpose |
| --- | --- |
| `fetch_openaq.py` | Fetch PM2.5 measurements for a single OpenAQ location. |
| `fetch_country_pm25.py` | Fetch PM2.5 measurements across locations in a country, with checkpointing and resumable collection. |
| `eda_openaq.py` | Run publication-oriented exploratory data analysis on a processed PM2.5 CSV. |
| `baseline_analysis.py` | Build behavioral baselines and baseline anomaly candidates. |
| `simulate_attacks.py` | Inject adversarial tampering scenarios into clean PM2.5 observations. |
| `detect_anomalies.py` | Run anomaly detectors against attacked datasets. |
| `explain_anomalies.py` | Generate explanation tables and plots for detected anomalies. |
| `robustness_testing.py` | Test detector robustness under varied perturbation conditions. |
| `evaluate_framework.py` | Evaluate detector precision, recall, F1 score, false positives, and attack detectability. |
| `check_rate_limit.py` | Utility for checking OpenAQ API rate-limit behavior. |
| `eda_openaq_legacy.py` | Legacy EDA script kept for reference. |

Generated outputs are written under `data/`, `data_results/`, `plots/`, and `logs/`.

## Requirements

- Python 3.10 or newer
- OpenAQ API key
- Python packages:
  - `requests`
  - `python-dotenv`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scikit-learn`

Install the dependencies into a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install requests python-dotenv pandas numpy matplotlib scikit-learn
```

## Configuration

Create a `.env` file in the repository root:

```env
OPENAQ_API_KEY=your_api_key_here
```

The fetch scripts load this value automatically with `python-dotenv`.

## Typical Workflow

### 1. Collect PM2.5 data

For a single OpenAQ location:

```powershell
python fetch_openaq.py --location-id 8118 --start-date 2025-01-01 --end-date 2025-12-31
```

For country-level collection:

```powershell
python fetch_country_pm25.py --country ET --start-date 2025-01-01 --end-date 2025-12-31
```

Processed CSV files are written to `data/processed/`. Raw API responses are preserved under `data/raw/openaq/`.

### 2. Run exploratory analysis

```powershell
python eda_openaq.py --csv data/processed/<processed_dataset>.csv
```

EDA figures are written to `plots/`.

### 3. Build the behavioral baseline

```powershell
python baseline_analysis.py --csv data/processed/<processed_dataset>.csv
```

Baseline figures are written to `plots/baseline/`.

### 4. Simulate attacks

```powershell
python simulate_attacks.py --csv data/processed/<processed_dataset>.csv
```

This creates `data/attacked/attacked_pm25_dataset.csv` and attack visualizations under `plots/attacks/`.

The implemented attack scenarios are:

- Constant bias injection
- Gradual drift attack
- Spike suppression
- Random stealth perturbation

### 5. Detect anomalies

```powershell
python detect_anomalies.py --csv data/attacked/attacked_pm25_dataset.csv
```

This creates `data/anomalies/anomaly_predictions.csv` and detection plots under `plots/anomalies/`.

The implemented detection methods are:

- Rolling z-score detection
- Isolation Forest
- Rolling reconstruction error

### 6. Explain anomaly results

```powershell
python explain_anomalies.py --csv data/anomalies/anomaly_predictions.csv
```

Explanation tables are written to `data/explanations/`; plots are written to `plots/explanations/`.

### 7. Evaluate detector performance

```powershell
python evaluate_framework.py --csv data/anomalies/anomaly_predictions.csv
```

Evaluation summaries are written to `data/evaluation/`; publication figures are written to `plots/evaluation/`.

### 8. Run robustness testing

```powershell
python robustness_testing.py --csv data/processed/<processed_dataset>.csv
```

Robustness summaries are written to `data/robustness/`; plots are written to `plots/robustness/`.

## Output Directories

| Directory | Contents |
| --- | --- |
| `data/raw/openaq/` | Raw OpenAQ JSON responses. |
| `data/processed/` | Cleaned PM2.5 CSV datasets. |
| `data/attacked/` | Attack-labeled datasets. |
| `data/anomalies/` | Detector predictions. |
| `data/explanations/` | Explanation and suspicious-region tables. |
| `data/evaluation/` | Evaluation summaries and detector rankings. |
| `data/robustness/` | Robustness experiment summaries. |
| `plots/` | EDA and publication figures. |
| `logs/` | Fetch and pipeline logs. |

## Notes

- The fetch scripts depend on OpenAQ v3 API availability and rate limits.
- `fetch_country_pm25.py` is the safer option for large country-level collection because it checkpoints progress and handles interruption.
- Data and plots are intentionally ignored in `.gitignore` because they may be large, regenerated, or tied to private research drafts.
- If you publish results from this repository, keep the exact input CSV, script version, date range, and OpenAQ query parameters with the paper artifacts for reproducibility.
