#!/usr/bin/env python3
"""
eda_openaq.py

Publication-ready EDA for OpenAQ PM2.5 datasets.
Supports multi-sensor country-level datasets.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.figsize": (12, 6),
    "font.size": 12,
    "axes.titlesize": 16,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "savefig.dpi": 600,
    "axes.grid": True,
    "grid.alpha": 0.3
})

def load_dataset(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8", low_memory=False)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df = df.sort_values("timestamp_utc")
    return df

def validate_dataset(df):
    print("\n================ DATASET SUMMARY ================")
    print("Rows:", len(df))
    print("Locations:", df["location_id"].nunique())
    print("Sensors:", df["sensor_id"].nunique())
    print("Country:", df["country"].dropna().iloc[0])
    print("Date Range:", df["timestamp_utc"].min(), "->", df["timestamp_utc"].max())
    print("\nPM2.5 Statistics")
    print(df["value"].describe())

def engineer_features(df):
    df = df.copy()
    df["hour"] = df["timestamp_utc"].dt.hour
    df["date"] = df["timestamp_utc"].dt.date

    country_ts = (df.groupby("timestamp_utc")["value"].mean().reset_index().sort_values("timestamp_utc"))

    country_ts["rolling_mean_24h"] = country_ts["value"].rolling(24).mean()
    country_ts["rolling_std_24h"] = country_ts["value"].rolling(24).std()

    mean = country_ts["value"].mean()
    std = country_ts["value"].std()

    country_ts["z_score"] = ((country_ts["value"] - mean) / std)

    country_ts["is_outlier"] = (np.abs(country_ts["z_score"]) > 3)

    # ========================================================
    # MONTHLY AGGREGATION FOR PUBLICATION FIGURES
    # ========================================================

    monthly_ts = (country_ts.set_index("timestamp_utc").resample("ME").agg({"value": ["mean", "std"]}))

    monthly_ts.columns = ["monthly_mean","monthly_std"]

    # ========================================================
    # VISUALIZATION-ONLY GAP INTERPOLATION
    # (NOT USED FOR ANALYSIS)
    # ========================================================

    monthly_ts["monthly_mean_plot"] = (
        monthly_ts["monthly_mean"]
        .interpolate(
            method="linear",
            limit=3
        )
    )

    monthly_ts["monthly_std_plot"] = (
        monthly_ts["monthly_std"]
        .interpolate(
            method="linear",
            limit=3
        )
    )

    monthly_ts = monthly_ts.reset_index()

    return df, country_ts, monthly_ts

def save_both(fig, name):
    fig.savefig(PLOTS_DIR / f"{name}.png", bbox_inches="tight", dpi=600)
    fig.savefig(PLOTS_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)

def plot_time_series(monthly_ts, country):

    # ====================================================
    # REMOVE EMPTY MONTHS FOR VISUALIZATION ONLY
    # ====================================================

    plot_df = (
        monthly_ts
        .dropna(
            subset=["monthly_mean"]
        )
        .copy()
    )

    fig, ax = plt.subplots(
        figsize=(14, 6)
    )

    ax.plot(
        plot_df["timestamp_utc"],
        plot_df["monthly_mean_plot"],
        linewidth=2.5,
        marker="o",
        markersize=4,
        label="Monthly Mean PM$_{2.5}$"
    )

    # ====================================================
    # TREND LINE
    # ====================================================

    x = np.arange(
        len(plot_df)
    )

    z = np.polyfit(
        x,
        plot_df["monthly_mean_plot"],
        1
    )
    trend = np.poly1d(z)

    ax.plot(
        plot_df["timestamp_utc"],
        trend(x),
        linestyle="--",
        linewidth=3,
        label="Trend"
    )

    ax.set_title(
        f"(a) {country} PM$_{{2.5}}$ Long-Term Trend (Observed Months)"
    )
    ax.set_ylabel(
        "PM$_{2.5}$ (µg/m³)"
    )

    ax.legend()

    save_both(
        fig,
        "pm25_time_series"
    )

def plot_distribution(country_ts):
    fig, ax = plt.subplots(figsize=(10,6))

    vals = country_ts["value"].dropna()
    vals = vals[vals >= 0]
    ax.hist(
        vals,
        bins=40,
        density=True,
        alpha=0.7
    )

    ax.axvline(vals.mean(), linestyle="--", label=f"Mean={vals.mean():.1f}")
    ax.axvline(vals.median(), linestyle=":", label=f"Median={vals.median():.1f}")
    ax.axvline(vals.quantile(.95), linestyle="-.", label=f"95th={vals.quantile(.95):.1f}")

    stats_text = (
        f"Mean: {vals.mean():.1f}\n"
        f"Median: {vals.median():.1f}\n"
        f"95th: {vals.quantile(0.95):.1f}"
    )

    ax.text(
        0.97,
        0.97,
        stats_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(alpha=0.8)
    )

    ax.set_title("(d) PM$_{2.5}$ Distribution")
    ax.legend()
    save_both(fig, "pm25_distribution")

def plot_rolling(monthly_ts):

    # ==========================================
    # REMOVE EMPTY MONTHS FOR VISUALIZATION ONLY
    # ==========================================

    plot_df = (
        monthly_ts
        .dropna(
            subset=["monthly_mean"]
        )
        .copy()
    )

    fig, ax = plt.subplots(
        figsize=(14, 6)
    )

    ax.plot(
        plot_df["timestamp_utc"],
        plot_df["monthly_mean_plot"],
        linewidth=2.5,
        label="Monthly Mean"
    )

    ax.fill_between(
        plot_df["timestamp_utc"],

        plot_df["monthly_mean_plot"] -
        plot_df["monthly_std_plot"],

        plot_df["monthly_mean_plot"] +
        plot_df["monthly_std_plot"],

        alpha=0.25,
        label="±1 Std Dev"
    )

    ax.set_title(
        "(b) Monthly PM$_{2.5}$ Variability"
    )

    ax.set_ylabel(
        "PM$_{2.5}$ (µg/m³)"
    )

    ax.legend()

    save_both(
        fig,
        "pm25_rolling_stats"
    )
def plot_seasonality(df):
    stats = df.groupby("hour")["value"].agg(["mean","std"])

    fig, ax = plt.subplots(figsize=(10,6))

    ax.plot(
        stats.index,
        stats["mean"],
        marker="o",
        linewidth=3
    )

    ax.fill_between(stats.index,
                    stats["mean"]-stats["std"],
                    stats["mean"]+stats["std"],
                    alpha=0.2)

    ax.set_title("(c) Diurnal PM$_{2.5}$ Pattern")
    ax.set_xlabel("Hour")
    save_both(fig, "pm25_daily_seasonality")

def plot_station_comparison(df):
    station = df.groupby("location_name")["value"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(12,6))
    station.plot(kind="barh", ax=ax)
    ax.set_title("Average PM$_{2.5}$ by Location")
    save_both(fig, "station_comparison")

def plot_availability(df):
    counts = df.groupby("sensor_id").size()
    fig, ax = plt.subplots(figsize=(10,6))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Observation Count per Sensor")
    save_both(fig, "data_availability")

def plot_outliers(country_ts):

    # ====================================================
    # TOP 20 MOST EXTREME EVENTS
    # ====================================================

    top_events = (
        country_ts
        .sort_values(
            by="z_score",
            ascending=False
        )
        .head(20)
        .copy()
    )

    fig, ax = plt.subplots(
        figsize=(14, 7)
    )

    # Background signal
    ax.plot(
        country_ts["timestamp_utc"],
        country_ts["value"],
        alpha=0.20,
        linewidth=1,
        label="PM$_{2.5}$"
    )

    # Extreme events
    ax.scatter(
        top_events["timestamp_utc"],
        top_events["value"],
        s=120,
        zorder=5,
        label="Top 20 Extreme Events"
    )

    # Annotate top 10
    for _, row in top_events.head(10).iterrows():

        ax.annotate(
            f'{row["value"]:.0f}',
            (
                row["timestamp_utc"],
                row["value"]
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8
        )

    ax.set_title(
        "Top 20 Extreme PM$_{2.5}$ Events (Baseline Dataset)"
    )

    ax.set_ylabel(
        "PM$_{2.5}$ (µg/m³)"
    )

    ax.legend()

    save_both(
        fig,
        "zscore_outliers"
    )

def plot_combined(df, country_ts, monthly_ts, country):
    fig, axes = plt.subplots(2,2, figsize=(16,10))

    # ==========================================
    # VISUALIZATION DATAFRAME
    # ==========================================

    plot_df = (
        monthly_ts
        .dropna(
            subset=["monthly_mean"]
        )
        .copy()
    )

    axes[0, 0].plot(
        plot_df["timestamp_utc"],
        plot_df["monthly_mean_plot"],
        linewidth=2.5
    )

    x = np.arange(len(plot_df))

    trend = np.poly1d(
        np.polyfit(
            x,
            plot_df["monthly_mean_plot"],
            1
        )
    )

    axes[0, 0].plot(
        plot_df["timestamp_utc"],
        trend(x),
        linestyle="--",
        linewidth=2
    )

    axes[0,0].set_title("(a) Time Series")

    axes[0, 1].plot(
        plot_df["timestamp_utc"],
        plot_df["monthly_mean_plot"],
        linewidth=2
    )

    axes[0, 1].fill_between(
        plot_df["timestamp_utc"],

        plot_df["monthly_mean_plot"] -
        plot_df["monthly_std_plot"],

        plot_df["monthly_mean_plot"] +
        plot_df["monthly_std_plot"],

        alpha=0.25
    )
    axes[0,1].set_title("(b) Rolling Statistics")

    stats = df.groupby("hour")["value"].agg(["mean","std"])
    axes[1,0].plot(stats.index, stats["mean"])
    axes[1,0].fill_between(stats.index,
                           stats["mean"]-stats["std"],
                           stats["mean"]+stats["std"],
                           alpha=.2)
    axes[1,0].set_title("(c) Diurnal Pattern")

    axes[1,1].hist(country_ts["value"], bins=40)
    axes[1,1].set_title("(d) Distribution")

    fig.suptitle(f"{country} PM$_{{2.5}}$ Dataset Characteristics", fontsize=18)
    save_both(fig, "pm25_combined_publication_figure")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    df = load_dataset(args.csv)
    validate_dataset(df)

    country = df["country"].dropna().iloc[0]

    df, country_ts, monthly_ts = engineer_features(df)

    plot_time_series(monthly_ts, country)
    plot_distribution(country_ts)
    plot_rolling(monthly_ts)
    plot_seasonality(df)
    plot_station_comparison(df)
    plot_availability(df)
    plot_outliers(country_ts)
    plot_combined(
        df,
        country_ts,
        monthly_ts,
        country)

    print("\nEDA completed successfully.")

if __name__ == "__main__":
    main()
