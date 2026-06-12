#!/usr/bin/env python3
"""
simulate_attacks.py

Adversarial Tampering Simulation
for the Environmental Cybersecurity project.

Purpose:
    - Simulate realistic manipulation attacks
      against PM2.5 environmental sensor data
    - Generate attacked datasets
    - Preserve attack labels for evaluation
    - Support anomaly detection experiments

Attack Scenarios:
    1. Constant Bias Injection
    2. Gradual Drift Attack
    3. Spike Suppression
    4. Random Stealth Perturbation

Outputs:
    - attacked_dataset.csv
    - attack labels
    - comparison plots
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# DIRECTORIES
# ============================================================

OUTPUT_DIR = Path("data/attacked")
PLOTS_DIR = Path("plots/attacks")

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
# ATTACK 1 — CONSTANT BIAS
# ============================================================

def constant_bias_attack(
        df,
        bias=25,
        start_ratio=0.30,
        end_ratio=0.45
):

    attacked = df.copy()

    start = int(len(df) * start_ratio)
    end = int(len(df) * end_ratio)

    attacked.loc[start:end, "value"] += bias

    attacked.loc[start:end, "attack_label"] = (
        "constant_bias"
    )

    return attacked


# ============================================================
# ATTACK 2 — GRADUAL DRIFT
# ============================================================

def gradual_drift_attack(
        df,
        drift_max=40,
        start_ratio=0.50,
        end_ratio=0.70
):

    attacked = df.copy()

    start = int(len(df) * start_ratio)
    end = int(len(df) * end_ratio)

    drift_values = np.linspace(
        0,
        drift_max,
        end - start + 1
    )

    attacked.loc[start:end, "value"] += drift_values

    attacked.loc[start:end, "attack_label"] = (
        "gradual_drift"
    )

    return attacked


# ============================================================
# ATTACK 3 — SPIKE SUPPRESSION
# ============================================================

def spike_suppression_attack(
        df,
        threshold=150,
        suppression_factor=0.5
):

    attacked = df.copy()

    spikes = attacked["value"] > threshold

    attacked.loc[spikes, "value"] *= suppression_factor

    attacked.loc[spikes, "attack_label"] = (
        "spike_suppression"
    )

    return attacked


# ============================================================
# ATTACK 4 — RANDOM STEALTH NOISE
# ============================================================

def stealth_perturbation_attack(
        df,
        noise_std=5,
        fraction=0.10
):

    attacked = df.copy()

    sample_size = int(len(df) * fraction)

    indices = np.random.choice(
        attacked.index,
        size=sample_size,
        replace=False
    )

    noise = np.random.normal(
        0,
        noise_std,
        sample_size
    )

    attacked.loc[indices, "value"] += noise

    attacked.loc[indices, "attack_label"] = (
        "stealth_perturbation"
    )

    return attacked


# ============================================================
# INITIALIZE LABELS
# ============================================================

def initialize_labels(df):

    df["attack_label"] = "normal"

    return df


# ============================================================
# INDIVIDUAL ATTACK VISUALIZATION
# ============================================================

def plot_attack_scenarios(original_df):

    bias_df = constant_bias_attack(
        initialize_labels(original_df.copy())
    )

    drift_df = gradual_drift_attack(
        initialize_labels(original_df.copy())
    )

    suppress_df = spike_suppression_attack(
        initialize_labels(original_df.copy())
    )

    stealth_df = stealth_perturbation_attack(
        initialize_labels(original_df.copy())
    )

    attack_sets = [
        ("(a) Constant Bias Injection", bias_df),
        ("(b) Gradual Drift Attack", drift_df),
        ("(c) Spike Suppression Attack", suppress_df),
        ("(d) Random Stealth Perturbation", stealth_df)
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 8)
    )

    axes = axes.flatten()

    for ax, (title, attack_df) in zip(
        axes,
        attack_sets
    ):

        original_monthly = (
            original_df
            .set_index("timestamp_utc")
            .resample("ME")["value"]
            .mean()
            .dropna()
        )

        attacked_monthly = (
            attack_df
            .set_index("timestamp_utc")
            .resample("ME")["value"]
            .mean()
            .dropna()
        )

        ax.plot(
            original_monthly.values,
            linewidth=2,
            marker="o",
            markersize=3,
            label="Original"
        )

        ax.plot(
            attacked_monthly.values,
            linestyle="--",
            linewidth=1.5,
            marker="s",
            markersize=2,
            label="Attacked"
        )

        ax.set_title(
            title,
            fontsize=12,
            fontweight="bold"
        )

        ax.set_xlabel(
            "Monthly Observation Index",
            fontsize=10
        )

        ax.set_ylabel(
            "PM$_{2.5}$ (µg/m³)",
            fontsize=10
        )

        ax.legend(
            fontsize=8,
            frameon=True
        )

        ax.grid(
            alpha=0.3,
            linestyle="--"
        )

    fig.suptitle(
        "Simulated Environmental Sensor Integrity Attacks",
        fontsize=16,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR /
        "attack_scenarios.png",
        dpi=600,
        bbox_inches="tight"
    )

    plt.savefig(
        PLOTS_DIR /
        "attack_scenarios.pdf",
        bbox_inches="tight"
    )

    plt.close()

    print(
        "[INFO] Saved: attack_scenarios.png"
    )


# ============================================================
# COMPARISON PLOT
# ============================================================

def plot_comparison(original_df, attacked_df):
    # Monthly aggregation
    original_monthly = (
        original_df
        .set_index("timestamp_utc")
        .resample("ME")["value"]
        .mean()
    )

    attacked_monthly = (
        attacked_df
        .set_index("timestamp_utc")
        .resample("ME")["value"]
        .mean()
    )

    # REMOVE missing months
    original_monthly = original_monthly.dropna()
    attacked_monthly = attacked_monthly.dropna()

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        range(len(original_monthly)),
        original_monthly.values,
        marker="o",
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.5,
        linewidth=2.5,
        label="Original Baseline"
    )

    ax.plot(
        range(len(attacked_monthly)),
        attacked_monthly.values,
        linestyle="--",
        linewidth=2,
        marker="s",
        markersize=4,
        label="Attacked Dataset"
    )

    tick_positions = np.linspace(
        0,
        len(original_monthly) - 1,
        8,
        dtype=int
    )

    tick_labels = [
        original_monthly.index[i].strftime("%Y")
        for i in tick_positions
    ]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    ax.set_title(
        "Monthly PM$_{2.5}$ Before and After Attack Injection"
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("PM$_{2.5}$ (µg/m³)")

    ax.legend()

    plt.tight_layout()

    output_path = (
        PLOTS_DIR /
        "attack_comparison_publication.png"
    )

    plt.savefig(
        output_path,
        dpi=600,
        bbox_inches="tight"
    )

    plt.savefig(
        output_path.with_suffix(".pdf"),
        bbox_inches="tight"
    )

    plt.close()

    print(f"[INFO] Saved: {output_path}")

# ============================================================
# ATTACK DISTRIBUTION
# ============================================================

def plot_attack_distribution(df):

    counts = (
        df["attack_label"]
        .value_counts()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Attack Injection Distribution"
    )

    ax.set_ylabel("Observations")

    plt.tight_layout()

    output_path = (
        PLOTS_DIR /
        "attack_distribution.png"
    )

    plt.savefig(
        output_path,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()

# ============================================================
# ATTACK TIMELINE
# ============================================================

def plot_attack_timeline(df):

    attack_df = df[
        df["attack_label"] != "normal"
    ].copy()

    mapping = {
        "constant_bias": 1,
        "gradual_drift": 2,
        "spike_suppression": 3,
        "stealth_perturbation": 4
    }

    attack_df["attack_id"] = (
        attack_df["attack_label"]
        .map(mapping)
    )

    fig, ax = plt.subplots(figsize=(14, 4))

    ax.scatter(
        attack_df["timestamp_utc"],
        attack_df["attack_id"],
        s=8
    )

    ax.set_yticks([1,2,3,4])

    ax.set_yticklabels([
        "Bias",
        "Drift",
        "Suppression",
        "Stealth"
    ])

    ax.set_title(
        "Temporal Distribution of Simulated Attacks"
    )

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR /
        "attack_timeline.png",
        dpi=600
    )

    plt.close()

# ============================================================
# SAVE ATTACKED DATASET
# ============================================================

def save_dataset(df):

    output_path = (
        OUTPUT_DIR /
        "attacked_pm25_dataset.csv"
    )

    df.to_csv(output_path, index=False)

    print(f"[INFO] Saved: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Environmental Sensor Attack Simulation"
    )

    parser.add_argument(
        "--csv",
        required=True,
        help="Path to processed CSV dataset"
    )

    args = parser.parse_args()

    df = load_dataset(args.csv)

    df = initialize_labels(df)

    original_df = df.copy()

    print("[INFO] Applying attacks...")

    df = constant_bias_attack(df)

    df = gradual_drift_attack(df)

    df = spike_suppression_attack(df)

    df = stealth_perturbation_attack(df)

    save_dataset(df)

    plot_attack_scenarios(original_df)

    plot_comparison(original_df, df)

    plot_attack_distribution(df)

    plot_attack_timeline(df)



    print("\n[INFO] Attack simulation completed.")


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    main()