#!/usr/bin/env python3
"""
detect_anomalies.py

Environmental Sensor Anomaly Detection
for the Environmental Cybersecurity project.

Purpose:
    - Detect manipulated PM2.5 observations
    - Identify adversarial perturbations
    - Evaluate anomaly detection performance

Detection Methods:
    1. Rolling Z-Score Detection
    2. Isolation Forest
    3. Rolling Reconstruction Error

Outputs:
    - anomaly_predictions.csv
    - anomaly plots
    - detection metrics
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# DIRECTORIES
# ============================================================

OUTPUT_DIR = Path("data/anomalies")
PLOTS_DIR = Path("plots/anomalies")

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
# LABEL PREPARATION
# ============================================================

def prepare_labels(df):

    df["is_attack"] = (
        df["attack_label"] != "normal"
    ).astype(int)

    return df


# ============================================================
# METHOD 1 — ROLLING Z-SCORE
# ============================================================

def rolling_zscore_detection(
        df,
        window=24,
        threshold=3
):

    rolling_mean = (
        df["value"]
        .rolling(window=window)
        .mean()
    )

    rolling_std = (
        df["value"]
        .rolling(window=window)
        .std()
    )

    z_scores = (
        (df["value"] - rolling_mean)
        / rolling_std
    )

    df["zscore_anomaly"] = (
        z_scores.abs() > threshold
    ).astype(int)

    return df


# ============================================================
# METHOD 2 — ISOLATION FOREST
# ============================================================

def isolation_forest_detection(df):

    model = IsolationForest(
        contamination=0.10,
        random_state=42
    )

    values = df[["value"]].fillna(0)

    predictions = model.fit_predict(values)

    df["iforest_anomaly"] = (
        predictions == -1
    ).astype(int)

    return df


# ============================================================
# METHOD 3 — RECONSTRUCTION ERROR
# ============================================================

def reconstruction_error_detection(
        df,
        window=24,
        threshold=20
):

    rolling_mean = (
        df["value"]
        .rolling(window=window)
        .mean()
    )

    reconstruction_error = (
        df["value"] - rolling_mean
    ).abs()

    df["reconstruction_error"] = (
        reconstruction_error
    )

    df["reconstruction_anomaly"] = (
        reconstruction_error > threshold
    ).astype(int)

    return df


# ============================================================
# MONTHLY ANOMALY AGGREGATION
# ============================================================

def build_monthly_anomaly_summary(df):

    monthly = (
        df.set_index("timestamp_utc")
          .resample("ME")
          .agg({
              "zscore_anomaly": "sum",
              "iforest_anomaly": "sum",
              "reconstruction_anomaly": "sum"
          })
          .reset_index()
    )

    return monthly

# ============================================================
# MONTHLY ANOMALY COMPARISON FIGURE
# ============================================================

def plot_monthly_anomaly_counts(monthly):

    fig, ax = plt.subplots(figsize=(14,6))

    ax.plot(
        monthly["timestamp_utc"],
        monthly["zscore_anomaly"],
        linewidth=2.5,
        marker="o",
        label="Rolling Z-Score"
    )

    ax.plot(
        monthly["timestamp_utc"],
        monthly["iforest_anomaly"],
        linewidth=2.5,
        marker="s",
        label="Isolation Forest"
    )

    ax.plot(
        monthly["timestamp_utc"],
        monthly["reconstruction_anomaly"],
        linewidth=2.5,
        marker="^",
        label="Reconstruction Error"
    )

    ax.set_title(
        "Monthly Detected Anomalies"
    )

    ax.set_ylabel(
        "Anomaly Count"
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR /
        "monthly_anomaly_counts.png",
        dpi=600
    )

    plt.close()

# ============================================================
# EVALUATION
# ============================================================

def evaluate_method(
        y_true,
        y_pred,
        method_name
):

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    print(f"\n{method_name}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    return {
        "Method": method_name,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }


# ============================================================
# PERFORMANCE COMPARISON FIGURE
# ============================================================

def plot_performance(metrics):

    methods = [
        m["Method"]
        for m in metrics
    ]

    precision = [
        m["Precision"]
        for m in metrics
    ]

    recall = [
        m["Recall"]
        for m in metrics
    ]

    f1 = [
        m["F1"]
        for m in metrics
    ]

    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10,6))

    ax.bar(
        x-width,
        precision,
        width,
        label="Precision"
    )

    ax.bar(
        x,
        recall,
        width,
        label="Recall"
    )

    ax.bar(
        x+width,
        f1,
        width,
        label="F1"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(methods)

    ax.set_ylim(0,1)

    ax.set_title(
        "Anomaly Detection Performance"
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR /
        "detection_performance.png",
        dpi=600
    )

    plt.close()


# ============================================================
# COMBINED FIGURE
# ============================================================
def plot_combined_detection(monthly, metrics):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16,6)
    )

    axes[0].plot(
        monthly["timestamp_utc"],
        monthly["zscore_anomaly"],
        label="Z-Score"
    )

    axes[0].plot(
        monthly["timestamp_utc"],
        monthly["iforest_anomaly"],
        label="Isolation Forest"
    )

    axes[0].plot(
        monthly["timestamp_utc"],
        monthly["reconstruction_anomaly"],
        label="Reconstruction Error"
    )

    axes[0].set_title(
        "(a) Monthly Detected Anomalies"
    )

    axes[0].legend()

    methods = [
        m["Method"]
        for m in metrics
    ]

    f1 = [
        m["F1"]
        for m in metrics
    ]

    axes[1].bar(
        methods,
        f1
    )

    axes[1].set_ylim(0,1)

    axes[1].set_title(
        "(b) F1-Score Comparison"
    )

    fig.suptitle(
        "Anomaly Detection Results"
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR /
        "anomaly_detection_publication_figure.png",
        dpi=600
    )

    plt.close()

# ============================================================
# ANOMALY VISUALIZATION
# ============================================================

# def plot_anomalies(
#         df,
#         column,
#         filename,
#         title
# ):
#
#     anomalies = df[df[column] == 1]
#
#     plt.figure(figsize=(14, 6))
#
#     plt.plot(
#         df["timestamp_utc"],
#         df["value"],
#         label="PM2.5"
#     )
#
#     plt.scatter(
#         anomalies["timestamp_utc"],
#         anomalies["value"]
#     )
#
#     plt.title(title)
#
#     plt.xlabel("Timestamp")
#     plt.ylabel("PM2.5")
#
#     plt.legend()
#
#     plt.tight_layout()
#
#     output_path = (
#         PLOTS_DIR / filename
#     )
#
#     plt.savefig(output_path)
#
#     plt.close()
#
#     print(f"[INFO] Saved: {output_path}")


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(df):

    output_path = (
        OUTPUT_DIR /
        "anomaly_predictions.csv"
    )

    df.to_csv(output_path, index=False)

    print(f"[INFO] Saved: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Environmental Anomaly Detection"
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="Path to attacked dataset"
    )

    args = parser.parse_args()

    df = load_dataset(args.csv)

    df = prepare_labels(df)

    print("[INFO] Running anomaly detection...")

    # ========================================================
    # METHOD 1 — ROLLING Z-SCORE
    # ========================================================

    df = rolling_zscore_detection(df)

    zscore_metrics = evaluate_method(
        df["is_attack"],
        df["zscore_anomaly"],
        "Rolling Z-Score"
    )

    # ========================================================
    # METHOD 2 — ISOLATION FOREST
    # ========================================================

    df = isolation_forest_detection(df)

    iforest_metrics = evaluate_method(
        df["is_attack"],
        df["iforest_anomaly"],
        "Isolation Forest"
    )

    # ========================================================
    # METHOD 3 — RECONSTRUCTION ERROR
    # ========================================================

    df = reconstruction_error_detection(df)

    recon_metrics = evaluate_method(
        df["is_attack"],
        df["reconstruction_anomaly"],
        "Reconstruction Error"
    )

    # ========================================================
    # COLLECT METRICS
    # ========================================================

    metrics = [
        zscore_metrics,
        iforest_metrics,
        recon_metrics
    ]

    # ========================================================
    # MONTHLY SUMMARY
    # ========================================================

    monthly = build_monthly_anomaly_summary(df)

    # ========================================================
    # PUBLICATION FIGURES
    # ========================================================

    print("\n[INFO] Generating publication figures...")

    plot_monthly_anomaly_counts(monthly)

    plot_performance(metrics)

    plot_combined_detection(
        monthly,
        metrics
    )

    # ========================================================
    # OPTIONAL DEBUG FIGURES
    # ========================================================
    #
    # Uncomment only if needed.
    #
    # plot_anomalies(
    #     df,
    #     "zscore_anomaly",
    #     "zscore_anomalies.png",
    #     "Rolling Z-Score Anomalies"
    # )
    #
    # plot_anomalies(
    #     df,
    #     "iforest_anomaly",
    #     "iforest_anomalies.png",
    #     "Isolation Forest Anomalies"
    # )
    #
    # plot_anomalies(
    #     df,
    #     "reconstruction_anomaly",
    #     "reconstruction_anomalies.png",
    #     "Reconstruction Error Anomalies"
    # )

    save_results(df)

    print("\n[INFO] Anomaly detection completed.")

# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    main()