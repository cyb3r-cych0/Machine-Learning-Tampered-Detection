#!/usr/bin/env python3
"""
baseline_analysis.py -- behavioral based analysis

Behavioral Baseline Analysis
for the Environmental Cybersecurity project.

Purpose:
    - Learn normal PM2.5 behavior
    - Identify natural environmental dynamics
    - Establish anomaly baselines
    - Prepare for tampering simulation

Outputs:
    - Temporal continuity analysis
    - Outlier density analysis
    - Hourly consistency plots
    - Daily variance plots
    - Z-score anomaly candidates
    - Rolling behavior analysis
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# DIRECTORIES
# ============================================================

PLOTS_DIR = Path("plots/baseline")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (12, 6),
    "font.size": 12,
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
    "savefig.dpi": 600,
    "axes.grid": True,
    "grid.alpha": 0.3
})

# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(csv_path):

    print(f"[INFO] Loading dataset: {csv_path}")

    df = pd.read_csv(csv_path)

    df["timestamp_utc"] = pd.to_datetime(
        df["timestamp_utc"],
        utc=True,
        errors="coerce"
    )

    df = df.sort_values("timestamp_utc")

    return df

def build_monthly_baseline(df):

    monthly = (
        df.set_index("timestamp_utc")
        .resample("ME")["value"]
        .agg(["mean", "std"])
        .reset_index()
    )

    monthly.columns = [
        "timestamp_utc",
        "monthly_mean",
        "monthly_std"
    ]

    monthly["monthly_mean_plot"] = (
        monthly["monthly_mean"]
        .interpolate(limit_direction="both")
    )

    monthly["monthly_std_plot"] = (
        monthly["monthly_std"]
        .interpolate(limit_direction="both")
    )

    return monthly

# ============================================================
# TEMPORAL GAP ANALYSIS
# ============================================================

def temporal_gap_analysis(df):

    print("\n==============================")
    print("TEMPORAL GAP ANALYSIS")
    print("==============================")

    df["time_diff"] = (
        df["timestamp_utc"]
        .diff()
        .dt.total_seconds() / 3600
    )

    gaps = df[df["time_diff"] > 2]

    print(f"Large Gaps (>2h): {len(gaps)}")

    if len(gaps) > 0:
        print(gaps[["timestamp_utc", "time_diff"]].head())


# ============================================================
# OUTLIER ANALYSIS
# ============================================================

def outlier_analysis(df):

    print("\n==============================")
    print("OUTLIER ANALYSIS")
    print("==============================")

    mean = df["value"].mean()
    std = df["value"].std()

    df["z_score"] = (
        (df["value"] - mean) / std
    )

    anomalies = df[df["z_score"].abs() > 3]

    print(f"Potential Outliers (|Z| > 3): {len(anomalies)}")

    return anomalies


# ============================================================
# HOURLY CONSISTENCY
# ============================================================

def plot_hourly_consistency(df):

    df["hour"] = df["timestamp_utc"].dt.hour

    stats = (
        df.groupby("hour")["value"]
        .agg(["mean","std"])
    )

    fig, ax = plt.subplots(figsize=(10,6))

    ax.plot(
        stats.index,
        stats["mean"],
        linewidth=3,
        marker="o",
        label="Mean PM$_{2.5}$"
    )

    ax.fill_between(
        stats.index,
        stats["mean"] - stats["std"],
        stats["mean"] + stats["std"],
        alpha=0.2,
        label="±1 Std Dev"
    )

    ax.set_title(
        "Hourly PM$_{2.5}$ Behavioral Baseline"
    )

    ax.set_xlabel("Hour")
    ax.set_ylabel("Average PM$_{2.5}$")

    ax.legend(loc="best")

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR /
        "hourly_behavioral_baseline.png",
        dpi=600
    )

    plt.close()

# ============================================================
# DAILY VARIANCE
# ============================================================

def plot_monthly_variance(monthly):

    plot_df = (
        monthly
        .dropna(subset=["monthly_mean"])
        .copy()
    )

    fig, ax = plt.subplots(figsize=(14,6))

    ax.plot(
        plot_df["timestamp_utc"],
        plot_df["monthly_mean_plot"],
        linewidth=3,
        label="Monthly Mean"
    )

    ax.fill_between(
        plot_df["timestamp_utc"],
        plot_df["monthly_mean_plot"]
        - plot_df["monthly_std_plot"],
        plot_df["monthly_mean_plot"]
        + plot_df["monthly_std_plot"],
        alpha=0.25,
        label="±1 Std Dev"
    )

    ax.set_title("Monthly PM$_{2.5}$ Variability")
    ax.set_ylabel("PM$_{2.5}$ (µg/m³)")
    ax.legend(loc="best")

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR /
        "monthly_variability.png",
        dpi=600
    )

    plt.close()

# ============================================================
# Z-SCORE ANOMALIES
# ============================================================

def plot_zscore_anomalies(df):

    anomaly_df = df.copy()

    anomaly_df["anomaly"] = (
        anomaly_df["z_score"].abs() > 3
    )

    monthly_counts = (
        anomaly_df
        .set_index("timestamp_utc")
        .resample("ME")["anomaly"]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(
        figsize=(14, 6)
    )

    ax.bar(
        monthly_counts["timestamp_utc"],
        monthly_counts["anomaly"],
        width=25,
        alpha=0.8
    )

    rolling_trend = (
        monthly_counts["anomaly"]
        .rolling(
            window=6,
            min_periods=1
        )
        .mean()
    )

    ax.plot(
        monthly_counts["timestamp_utc"],
        rolling_trend,
        linewidth=3,
        linestyle="--",
        color="crimson",
        label="6-Month Trend"
    )

    ax.set_title(
        "Monthly PM$_{2.5}$ Anomaly Frequency"
    )

    ax.set_xlabel("Year")

    ax.set_ylabel(
        "Number of Anomalies"
    )

    ax.legend(loc="best")

    plt.tight_layout()

    output_path = (
        PLOTS_DIR /
        "monthly_anomaly_count.png"
    )

    plt.savefig(
        output_path,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()

    print(f"[INFO] Saved: {output_path}")


def plot_combined_baseline(df, monthly):

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(18, 12)
    )

    # =====================================================
    # (a) Monthly Trend
    # =====================================================

    plot_df = (
        monthly
        .dropna(subset=["monthly_mean"])
        .copy()
    )

    axes[0, 0].plot(
        plot_df["timestamp_utc"],
        plot_df["monthly_mean_plot"],
        linewidth=3,
        label="Monthly Mean"
    )

    # # 6-month moving average
    # trend = (
    #     plot_df["monthly_mean_plot"]
    #     .rolling(6, min_periods=1)
    #     .mean()
    # )

    import numpy as np

    x = np.arange(len(plot_df))

    coef = np.polyfit(
        x,
        plot_df["monthly_mean_plot"],
        1
    )

    trend = np.poly1d(coef)(x)

    axes[0, 0].plot(
        plot_df["timestamp_utc"],
        trend,
        color="orange",
        linestyle="--",
        linewidth=3,
        label="Linear Trend"
    )

    axes[0, 0].legend()

    axes[0, 0].set_title("(a) Monthly PM$_{2.5}$ Trend")
    axes[0, 0].set_ylabel("PM$_{2.5}$ (µg/m³)")

    # =====================================================
    # (b) Monthly Variability
    # =====================================================

    axes[0, 1].plot(
        plot_df["timestamp_utc"],
        plot_df["monthly_mean_plot"],
        linewidth=3
    )

    axes[0, 1].fill_between(
        plot_df["timestamp_utc"],
        plot_df["monthly_mean_plot"] -
        plot_df["monthly_std_plot"],
        plot_df["monthly_mean_plot"] +
        plot_df["monthly_std_plot"],
        alpha=0.25
    )

    axes[0, 1].legend(
        ["Monthly Mean", "±1 Std Dev"]
    )

    axes[0, 1].set_title("(b) Monthly Variability")
    axes[0, 1].set_ylabel("PM$_{2.5}$ (µg/m³)")

    # =====================================================
    # (c) Hourly Baseline
    # =====================================================

    hourly_stats = (
        df.groupby(
            df["timestamp_utc"].dt.hour
        )["value"]
        .agg(["mean", "std"])
    )

    axes[1, 0].plot(
        hourly_stats.index,
        hourly_stats["mean"],
        linewidth=3,
        marker="o",
        label="Mean PM$_{2.5}$"
    )

    axes[1, 0].fill_between(
        hourly_stats.index,
        hourly_stats["mean"] -
        hourly_stats["std"],
        hourly_stats["mean"] +
        hourly_stats["std"],
        alpha=0.2,
        label="±1 Std Dev"
    )

    axes[1, 0].legend()

    axes[1, 0].set_title("(c) Hourly Behavioral Baseline")
    axes[1, 0].set_xlabel("Hour")
    axes[1, 0].set_ylabel("PM$_{2.5}$ (µg/m³)")

    # =====================================================
    # (d) Monthly Anomaly Count
    # =====================================================

    anomaly_df = df.copy()

    anomaly_df["anomaly"] = (
            anomaly_df["z_score"].abs() > 3
    )

    monthly_counts = (
        anomaly_df
        .set_index("timestamp_utc")
        .resample("ME")["anomaly"]
        .sum()
        .reset_index()
    )

    axes[1, 1].bar(
        monthly_counts["timestamp_utc"],
        monthly_counts["anomaly"],
        width=25,
        alpha=0.8
    )

    trend = (
        monthly_counts["anomaly"]
        .rolling(
            6,
            min_periods=1
        )
        .mean()
    )

    axes[1, 1].plot(
        monthly_counts["timestamp_utc"],
        trend,
        linewidth=3,
        linestyle="--",
        color="crimson",
        label="6-Month Trend"
    )

    axes[1, 1].legend(
        ["6-Month Trend", "Monthly Count"]
    )

    axes[1, 1].set_title(
        "(d) Monthly Anomaly Count"
    )

    axes[1, 1].set_ylabel(
        "Count"
    )

    # =====================================================
    # Figure Title
    # =====================================================

    country = (
        df["country"]
        .dropna()
        .iloc[0]
    )

    fig.suptitle(
        f"{country} PM$_{{2.5}}$ Behavioral Baseline Analysis",
        fontsize=22,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR /
        "baseline_combined_figure.png",
        dpi=600,
        bbox_inches="tight"
    )

    plt.savefig(
        PLOTS_DIR /
        "baseline_combined_figure.pdf",
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"[INFO] Saved: "
        f"{PLOTS_DIR / 'baseline_combined_figure.png'}"
    )

# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Behavioral Baseline Analysis"
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="Path to processed CSV dataset"
    )

    args = parser.parse_args()

    df = load_dataset(args.csv)

    monthly = build_monthly_baseline(df)

    temporal_gap_analysis(df)

    outlier_analysis(df)

    print("\n[INFO] Generating baseline plots...")

    plot_hourly_consistency(df)

    plot_monthly_variance(monthly)

    plot_zscore_anomalies(df)

    # plot_rolling_baseline(monthly)

    plot_combined_baseline(
        df,
        monthly
    )

    print("\n[INFO] Baseline analysis completed.")


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    main()