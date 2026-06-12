#!/usr/bin/env python3
"""
evaluate_framework.py

Research Evaluation Framework
for the Environmental Cybersecurity project.

Purpose:
    - Evaluate environmental anomaly detection methods
    - Compare cybersecurity detection performance
    - Analyze attack detectability
    - Generate publication-ready metrics and plots

Evaluation Focus:
    1. Precision
    2. Recall
    3. F1-Score
    4. False Positive Rate
    5. Attack Detectability
    6. Detection Method Comparison

Outputs:
    - evaluation_summary.csv
    - attack_detectability.csv
    - publication-quality plots
    - ranked detector performance
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    accuracy_score
)

# ============================================================
# DIRECTORIES
# ============================================================

OUTPUT_DIR = Path("data/evaluation")
PLOTS_DIR = Path("plots/evaluation")

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
# PREPARE GROUND TRUTH
# ============================================================

def prepare_ground_truth(df):

    df["is_attack"] = (
        df["attack_label"] != "normal"
    ).astype(int)

    return df


# ============================================================
# FALSE POSITIVE RATE
# ============================================================

def calculate_false_positive_rate(
        y_true,
        y_pred
):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred
    ).ravel()

    if (fp + tn) == 0:
        return 0

    return fp / (fp + tn)


# ============================================================
# METHOD EVALUATION
# ============================================================

def evaluate_method(
        df,
        method_column,
        method_name
):

    y_true = df["is_attack"]
    y_pred = df[method_column]

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

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    fpr = calculate_false_positive_rate(
        y_true,
        y_pred
    )

    roc_auc = roc_auc_score(
        y_true,
        y_pred
    )

    return {
        "method": method_name,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
        "false_positive_rate": fpr,
        "roc_auc": roc_auc
    }


# ============================================================
# EVALUATE ALL METHODS
# ============================================================

def evaluate_all_methods(df):

    print("\n==============================")
    print("DETECTION METHOD EVALUATION")
    print("==============================")

    methods = [
        (
            "zscore_anomaly",
            "Rolling Z-Score"
        ),
        (
            "iforest_anomaly",
            "Isolation Forest"
        ),
        (
            "reconstruction_anomaly",
            "Reconstruction Error"
        )
    ]

    results = []

    for column, name in methods:

        metrics = evaluate_method(
            df,
            column,
            name
        )

        results.append(metrics)

        print(f"\n{name}")

        print(
            f"Precision          : "
            f"{metrics['precision']:.4f}"
        )

        print(
            f"Recall             : "
            f"{metrics['recall']:.4f}"
        )

        print(
            f"F1-Score           : "
            f"{metrics['f1_score']:.4f}"
        )

        print(
            f"Accuracy           : "
            f"{metrics['accuracy']:.4f}"
        )

        print(
            f"False Positive Rate: "
            f"{metrics['false_positive_rate']:.4f}"
        )

    results_df = pd.DataFrame(results)

    return results_df


# ============================================================
# SAVE EVALUATION SUMMARY
# ============================================================

def save_evaluation_summary(results_df):

    output_path = (
        OUTPUT_DIR /
        "evaluation_summary.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print(f"\n[INFO] Saved: {output_path}")


# ============================================================
# DETECTOR RANKING
# ============================================================

def detector_ranking(results_df):

    ranked = results_df.sort_values(
        by="f1_score",
        ascending=False
    )

    print("\n==============================")
    print("DETECTOR RANKING")
    print("==============================")

    print(
        ranked[
            [
                "method",
                "f1_score",
                "precision",
                "recall"
            ]
        ]
    )

    output_path = (
        OUTPUT_DIR /
        "detector_ranking.csv"
    )

    ranked.to_csv(
        output_path,
        index=False
    )

    print(f"\n[INFO] Saved: {output_path}")

    return ranked


# ============================================================
# ATTACK DETECTABILITY ANALYSIS
# ============================================================

def attack_detectability_analysis(df):

    print("\n==============================")
    print("ATTACK DETECTABILITY ANALYSIS")
    print("==============================")

    attack_types = [
        attack
        for attack in df["attack_label"].unique()
        if attack != "normal"
    ]

    results = []

    for attack in attack_types:

        attack_df = df[
            (
                (df["attack_label"] == attack) |
                (df["attack_label"] == "normal")
            )
        ]

        y_true = (
            attack_df["attack_label"] != "normal"
        ).astype(int)

        methods = {
            "Rolling Z-Score":
                "zscore_anomaly",

            "Isolation Forest":
                "iforest_anomaly",

            "Reconstruction Error":
                "reconstruction_anomaly"
        }

        for method_name, column in methods.items():

            y_pred = attack_df[column]

            recall = recall_score(
                y_true,
                y_pred,
                zero_division=0
            )

            results.append({
                "attack_type": attack,
                "method": method_name,
                "detectability_recall": recall
            })

    detectability_df = pd.DataFrame(results)

    print(detectability_df)

    output_path = (
        OUTPUT_DIR /
        "attack_detectability.csv"
    )

    detectability_df.to_csv(
        output_path,
        index=False
    )

    print(f"\n[INFO] Saved: {output_path}")

    return detectability_df


# ============================================================
# PERFORMANCE COMPARISON PLOT
# ============================================================

def performance_comparison_plot(results_df):

    metrics = [
        "precision",
        "recall",
        "f1_score"
    ]

    x = np.arange(len(results_df))

    width = 0.25

    plt.figure(figsize=(12, 6))

    for i, metric in enumerate(metrics):

        plt.bar(
            x + (i * width),
            results_df[metric],
            width=width,
            label=metric
        )

    plt.xticks(
        x + width,
        results_df["method"]
    )

    plt.ylabel("Score")

    plt.title(
        "Detection Method Performance Comparison"
    )

    plt.legend()

    plt.tight_layout()

    output_path = (
        PLOTS_DIR /
        "detector_performance_comparison.png"
    )

    plt.savefig(output_path)

    plt.close()

    print(f"[INFO] Saved: {output_path}")


# ============================================================
# ATTACK DETECTABILITY PLOT
# ============================================================

def attack_detectability_table(df):

    table = (
        df.pivot(
            index="attack_type",
            columns="method",
            values="detectability_recall"
        )
        .reset_index()
    )

    # Optional: prettier column names
    table = table.rename(columns={
        "Rolling Z-Score": "Rolling Z-score",
        "Reconstruction Error": "Reconstruction Error",
        "Isolation Forest": "Isolation Forest"
    })

    print("\n==============================")
    print("ATTACK DETECTABILITY TABLE")
    print("==============================")
    print(table)

    output_path = (
        OUTPUT_DIR /
        "attack_detectability_table.csv"
    )

    table.to_csv(
        output_path,
        index=False
    )

    print(f"\n[INFO] Saved: {output_path}")

    return table

# ============================================================
# ATTACK DETECTABILITY MATRIX
# ============================================================

def attack_detectability_matrix(detectability_df):

    pivot = (
        detectability_df
        .pivot_table(
            index="attack_type",
            columns="method",
            values="detectability_recall",
            aggfunc="mean"
        )
        .fillna(0)
    )

    print("\n==============================")
    print("ATTACK DETECTABILITY MATRIX")
    print("==============================")
    print(pivot)

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    heatmap = ax.imshow(
        pivot.values,
        aspect="auto",
        cmap="YlOrRd",
        vmin=0,
        vmax=1
    )

    ax.set_xticks(
        np.arange(len(pivot.columns))
    )

    ax.set_xticklabels(
        pivot.columns,
        rotation=20,
        ha="right"
    )

    ax.set_yticks(
        np.arange(len(pivot.index))
    )

    ax.set_yticklabels([
        attack.replace("_", " ").title()
        for attack in pivot.index
    ])

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):

            value = pivot.iloc[i, j]

            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
                fontweight="bold"
            )

    cbar = plt.colorbar(
        heatmap,
        ax=ax
    )

    cbar.set_label(
        "Detection Recall",
        fontsize=11
    )

    ax.set_title(
        "Attack Detectability Matrix",
        fontsize=14,
        fontweight="bold"
    )

    ax.set_xlabel(
        "Detection Method",
        fontsize=12
    )

    ax.set_ylabel(
        "Attack Type",
        fontsize=12
    )

    plt.tight_layout()

    output_png = (
        PLOTS_DIR /
        "attack_detectability_matrix.png"
    )

    output_pdf = (
        PLOTS_DIR /
        "attack_detectability_matrix.pdf"
    )

    plt.savefig(
        output_png,
        dpi=600,
        bbox_inches="tight"
    )

    plt.savefig(
        output_pdf,
        bbox_inches="tight"
    )

    plt.close()

    print(f"[INFO] Saved: {output_png}")
# ============================================================
# PUBLICATION SUMMARY
# ============================================================

def publication_summary(results_df):

    best_method = results_df.sort_values(
        by="f1_score",
        ascending=False
    ).iloc[0]

    print("\n==============================")
    print("PUBLICATION SUMMARY")
    print("==============================")

    print(
        f"Best Performing Detector : "
        f"{best_method['method']}"
    )

    print(
        f"Best F1-Score            : "
        f"{best_method['f1_score']:.4f}"
    )

    print(
        f"Precision                : "
        f"{best_method['precision']:.4f}"
    )

    print(
        f"Recall                   : "
        f"{best_method['recall']:.4f}"
    )

    print(
        f"False Positive Rate      : "
        f"{best_method['false_positive_rate']:.4f}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Environmental Cybersecurity Evaluation"
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="Path to anomaly prediction dataset"
    )

    args = parser.parse_args()

    df = load_dataset(args.csv)

    df = prepare_ground_truth(df)

    # ========================================================
    # METHOD EVALUATION
    # ========================================================

    results_df = evaluate_all_methods(df)

    save_evaluation_summary(results_df)

    ranked = detector_ranking(results_df)

    # ========================================================
    # ATTACK DETECTABILITY
    # ========================================================

    detectability_df = (
        attack_detectability_analysis(df)
    )

    # ========================================================
    # PLOTS
    # ========================================================

    print("\n[INFO] Generating evaluation plots...")

    performance_comparison_plot(results_df)

    # attack_detectability_plot(
    #     detectability_df
    # )

    print(detectability_df.head())
    print(detectability_df.columns)
    print(detectability_df.shape)

    attack_detectability_matrix(
        detectability_df
    )

    attack_detectability_table(
        detectability_df
    )
    # ========================================================
    # PUBLICATION SUMMARY
    # ========================================================

    publication_summary(results_df)

    print("\n[INFO] Evaluation completed.")


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    main()