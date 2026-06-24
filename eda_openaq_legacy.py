#!/usr/bin/env python3
"""
eda_openaq.py

Exploratory Data Analysis (EDA)
for the Environmental Cybersecurity project.

Purpose:
    - Validate collected PM2.5 dataset
    - Analyze temporal behavior
    - Establish environmental baseline
    - Generate research-quality plots

Outputs:
    - Dataset statistics
    - Missing value analysis
    - Duplicate analysis
    - Time-series visualization
    - Distribution plots
    - Rolling statistics
    - Daily seasonality plots

    Load Dataset
          ↓
    Validate Dataset
          ↓
    Feature Engineering
          ↓
    Summary Statistics Report
          ↓
    Individual Figures
          ↓
    Combined Publication Figure
          ↓
    Export PNG + PDF

"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# DIRECTORIES
# ============================================================

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(csv_path):

    print(f"[INFO] Loading dataset: {csv_path}")

    # df = pd.read_csv(csv_path)
    df = pd.read_csv(
        csv_path,
        encoding="utf-8-sig"
    )

    print(df.columns.tolist())

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True,
        errors="coerce"
    )

    df = df.sort_values("timestamp_utc")

    return df


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_dataset(df):

    print("\n==============================")
    print("DATASET VALIDATION")
    print("==============================")

    print(f"Total Rows        : {len(df)}")
    print(f"Total Columns     : {len(df.columns)}")

    print("\nMissing Values:")
    print(df.isnull().sum())

    duplicates = df.duplicated().sum()

    print(f"\nDuplicate Rows    : {duplicates}")

    print("\nPM2.5 Statistics:")
    print(df["value"].describe())


# ============================================================
# TIME SERIES PLOT
# ============================================================

def plot_time_series(df):

    plt.figure(figsize=(14, 6))

    plt.plot(
        df["timestamp_utc"],
        df["value"]
    )

    plt.title("PM2.5 Time Series")
    plt.xlabel("Timestamp")
    plt.ylabel("PM2.5 (µg/m³)")

    plt.tight_layout()

    output_path = PLOTS_DIR / "pm25_time_series.png"

    plt.savefig(output_path)

    plt.close()

    print(f"[INFO] Saved: {output_path}")


# ============================================================
# HISTOGRAM
# ============================================================

def plot_distribution(df):

    plt.figure(figsize=(10, 6))

    plt.hist(
        df["value"],
        bins=50
    )

    plt.title("PM2.5 Distribution")
    plt.xlabel("PM2.5 (µg/m³)")
    plt.ylabel("Frequency")

    plt.tight_layout()

    output_path = PLOTS_DIR / "pm25_distribution.png"

    plt.savefig(output_path)

    plt.close()

    print(f"[INFO] Saved: {output_path}")


# ============================================================
# ROLLING STATISTICS
# ============================================================

def plot_rolling_statistics(df):

    rolling_mean = df["value"].rolling(window=24).mean()
    rolling_std = df["value"].rolling(window=24).std()

    plt.figure(figsize=(14, 6))

    plt.plot(
        df["timestamp_utc"],
        df["value"],
        label="PM2.5"
    )

    plt.plot(
        df["timestamp_utc"],
        rolling_mean,
        label="24H Rolling Mean"
    )

    plt.plot(
        df["timestamp_utc"],
        rolling_std,
        label="24H Rolling Std"
    )

    plt.title("Rolling Mean and Standard Deviation")
    plt.xlabel("Timestamp")
    plt.ylabel("PM2.5")

    plt.legend()

    plt.tight_layout()

    output_path = PLOTS_DIR / "pm25_rolling_stats.png"

    plt.savefig(output_path)

    plt.close()

    print(f"[INFO] Saved: {output_path}")


# ============================================================
# DAILY SEASONALITY
# ============================================================

def plot_daily_seasonality(df):

    df["hour"] = df["timestamp_utc"].dt.hour

    hourly_avg = (
        df.groupby("hour")["value"]
        .mean()
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        hourly_avg.index,
        hourly_avg.values
    )

    plt.title("Average PM2.5 by Hour of Day")
    plt.xlabel("Hour")
    plt.ylabel("Average PM2.5")

    plt.tight_layout()

    output_path = PLOTS_DIR / "pm25_daily_seasonality.png"

    plt.savefig(output_path)

    plt.close()

    print(f"[INFO] Saved: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="EDA for OpenAQ PM2.5 dataset"
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="Path to processed CSV dataset"
    )

    args = parser.parse_args()

    df = load_dataset(args.csv)

    print(df.columns.tolist())

    # validate_dataset(df)
    #
    # print("\n[INFO] Generating plots...")
    #
    # plot_time_series(df)
    #
    # plot_distribution(df)
    #
    # plot_rolling_statistics(df)
    #
    # plot_daily_seasonality(df)

    print("\n[INFO] EDA completed successfully.")


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    main()