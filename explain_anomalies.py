#!/usr/bin/env python3
"""
explain_anomalies.py

Explainable Environmental Cybersecurity Analysis

Purpose:
    - Explain detected PM2.5 anomalies
    - Compare normal vs attacked behavior
    - Interpret suspicious temporal regions
    - Support explainable environmental cybersecurity

Outputs:
    - anomaly explanation tables
    - temporal explanation plots
    - attack-type summaries
    - feature deviation analysis
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# DIRECTORIES
# ============================================================

OUTPUT_DIR = Path("data/explanations")
PLOTS_DIR = Path("plots/explanations")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

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


# ============================================================
# ANOMALY SUMMARY
# ============================================================

def anomaly_summary(df):

    print("\n==============================")
    print("ANOMALY SUMMARY")
    print("==============================")

    summary = (
        df["attack_label"]
        .value_counts()
    )

    print(summary)

    return summary


# ============================================================
# DETECTION PERFORMANCE SUMMARY
# ============================================================

def detection_summary(df):

    print("\n==============================")
    print("DETECTION SUMMARY")
    print("==============================")

    methods = [
        "zscore_anomaly",
        "iforest_anomaly",
        "reconstruction_anomaly"
    ]

    for method in methods:

        total = df[method].sum()

        print(f"{method}: {total} anomalies")


# ============================================================
# FEATURE DEVIATION ANALYSIS
# ============================================================

def feature_deviation_analysis(df):

    print("\n==============================")
    print("FEATURE DEVIATION ANALYSIS")
    print("==============================")

    grouped = (
        df.groupby("attack_label")["value"]
        .agg(["mean", "std", "min", "max"])
    )

    print(grouped)

    output_path = (
        OUTPUT_DIR /
        "feature_deviation_summary.csv"
    )

    grouped.to_csv(output_path)

    print(f"\n[INFO] Saved: {output_path}")


# ============================================================
# TEMPORAL EXPLANATION PLOT
# ============================================================

# ============================================================
# ATTACK DISTRIBUTION
# ============================================================

def plot_attack_distribution(df):

    attack_counts = (
        df["attack_label"]
        .value_counts()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(10, 6))

    attack_counts.plot(
        kind="bar",
        color="steelblue",
        edgecolor="black"
    )

    plt.title(
        "Attack Scenario Distribution",
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel("Attack Type")
    plt.ylabel("Number of Observations")

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    output_path = (
        PLOTS_DIR /
        "attack_distribution.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"[INFO] Saved: {output_path}")

# ============================================================
# ATTACK REGION ANALYSIS
# ============================================================

def suspicious_region_analysis(df):

    print("\n==============================")
    print("SUSPICIOUS REGION ANALYSIS")
    print("==============================")

    suspicious = df[
        (
            (df["zscore_anomaly"] == 1) |
            (df["iforest_anomaly"] == 1) |
            (df["reconstruction_anomaly"] == 1)
        )
    ]

    suspicious = suspicious[
        [
            "timestamp_utc",
            "value",
            "attack_label",
            "zscore_anomaly",
            "iforest_anomaly",
            "reconstruction_anomaly"
        ]
    ]

    print(suspicious.head(20))

    output_path = (
        OUTPUT_DIR /
        "suspicious_regions.csv"
    )

    suspicious.to_csv(output_path, index=False)

    print(f"\n[INFO] Saved: {output_path}")


# ============================================================
# COMPARISON PLOT
# ============================================================

# ============================================================
# DETECTOR AGREEMENT MATRIX
# ============================================================

def plot_detector_agreement(df):

    agreement = pd.DataFrame({

        "Z-Score":
            df["zscore_anomaly"],

        "Isolation Forest":
            df["iforest_anomaly"],

        "Reconstruction":
            df["reconstruction_anomaly"]

    })

    corr = agreement.corr()

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    im = ax.imshow(
        corr,
        cmap="Blues",
        vmin=0,
        vmax=1
    )

    ax.set_xticks(
        range(len(corr.columns))
    )

    ax.set_yticks(
        range(len(corr.columns))
    )

    ax.set_xticklabels(
        corr.columns,
        rotation=20
    )

    ax.set_yticklabels(
        corr.columns
    )

    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(
                j,
                i,
                f"{corr.iloc[i,j]:.2f}",
                ha="center",
                va="center",
                fontsize=10
            )

    plt.colorbar(
        im,
        ax=ax,
        label="Correlation"
    )

    plt.title(
        "Detector Agreement Matrix",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout()

    output_path = (
        PLOTS_DIR /
        "detector_agreement.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"[INFO] Saved: {output_path}")

# ============================================================
# ATTACK TYPE DETECTION EFFECTIVENESS
# ============================================================

def plot_attack_detection_rate(df):

    detector_cols = [

        "zscore_anomaly",

        "iforest_anomaly",

        "reconstruction_anomaly"

    ]

    attacked = df[
        df["attack_label"] != "normal"
    ].copy()

    attacked["detected"] = (
        attacked[detector_cols]
        .max(axis=1)
    )

    detection_rate = (

        attacked.groupby(
            "attack_label"
        )["detected"]

        .mean()

        * 100

    )

    plt.figure(figsize=(10, 6))

    bars = plt.bar(

        detection_rate.index,

        detection_rate.values,

        color="darkorange",

        edgecolor="black"

    )

    plt.ylabel(
        "Detection Rate (%)"
    )

    plt.xlabel(
        "Attack Type"
    )

    plt.ylim(
        0,
        100
    )

    plt.title(
        "Attack-Type Detection Effectiveness",
        fontsize=14,
        fontweight="bold"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    for bar in bars:

        plt.text(

            bar.get_x()
            + bar.get_width()/2,

            bar.get_height()+2,

            f"{bar.get_height():.1f}%",

            ha="center"

        )

    plt.tight_layout()

    output_path = (
        PLOTS_DIR /
        "attack_detection_rate.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"[INFO] Saved: {output_path}")

# ============================================================
# COMBINED EXPLAINABILITY FIGURE
# ============================================================

def plot_combined_explainability(df):

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5)
    )

    # ---------------------------------
    # Attack Distribution
    # ---------------------------------

    attack_counts = (
        df["attack_label"]
        .value_counts()
    )

    attack_counts.plot(
        kind="bar",
        ax=axes[0],
        color="steelblue"
    )

    axes[0].set_title(
        "(a) Attack Distribution"
    )

    # ---------------------------------
    # Detector Agreement
    # ---------------------------------

    agreement = pd.DataFrame({

        "Z":
            df["zscore_anomaly"],

        "IF":
            df["iforest_anomaly"],

        "RE":
            df["reconstruction_anomaly"]

    })

    corr = agreement.corr()

    axes[1].imshow(
        corr,
        cmap="Blues",
        vmin=0,
        vmax=1
    )

    axes[1].set_xticks(
        range(len(corr.columns))
    )

    axes[1].set_yticks(
        range(len(corr.columns))
    )

    axes[1].set_xticklabels(
        corr.columns
    )

    axes[1].set_yticklabels(
        corr.columns
    )

    axes[1].set_title(
        "(b) Detector Agreement"
    )

    # ---------------------------------
    # Detection Effectiveness
    # ---------------------------------

    attacked = df[
        df["attack_label"] != "normal"
    ].copy()

    attacked["detected"] = (

        attacked[
            [
                "zscore_anomaly",
                "iforest_anomaly",
                "reconstruction_anomaly"
            ]
        ]

        .max(axis=1)

    )

    rates = (

        attacked.groupby(
            "attack_label"
        )["detected"]

        .mean()

        * 100

    )

    axes[2].bar(
        rates.index,
        rates.values,
        color="darkorange"
    )

    axes[2].set_ylim(
        0,
        100
    )

    axes[2].set_title(
        "(c) Detection Rate"
    )

    fig.suptitle(
        "Environmental Cybersecurity Explainability Analysis",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()

    output_path = (
        PLOTS_DIR /
        "combined_explainability_analysis.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"[INFO] Saved: {output_path}")

# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Explainable Environmental Cybersecurity"
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="Path to anomaly prediction dataset"
    )

    args = parser.parse_args()

    df = load_dataset(args.csv)

    anomaly_summary(df)

    detection_summary(df)

    feature_deviation_analysis(df)

    suspicious_region_analysis(df)

    print("\n[INFO] Generating explanation plots...")

    plot_attack_distribution(df)

    plot_detector_agreement(df)

    plot_attack_detection_rate(df)

    plot_combined_explainability(df)

    print("\n[INFO] Explainability analysis completed.")


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    main()