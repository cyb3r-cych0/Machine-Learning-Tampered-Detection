import pandas as pd

from preprocessing.config import FLATLINE_WINDOW
from preprocessing.integrity import (
    detect_clock_duplicates,
    detect_constant_sequences,
    detect_flatlines,
    detect_spikes,
    detect_time_gaps,
    integrity_assessment,
)


def sensor_dataset(timestamps, values):
    return pd.DataFrame(
        {
            "country": ["ET"] * len(values),
            "location_id": [101] * len(values),
            "sensor_id": [201] * len(values),
            "timestamp_utc": pd.to_datetime(timestamps, utc=True),
            "value": values,
        }
    )


def test_detect_time_gaps_flags_large_sensor_gap():
    df = sensor_dataset(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-02T02:00:00Z",
        ],
        [10.0, 12.0],
    )

    result = detect_time_gaps(df)

    assert result["is_gap"].tolist() == [False, True]
    assert result.loc[1, "time_gap_hours"] == 26


def test_detect_flatlines_flags_full_constant_window():
    timestamps = pd.date_range(
        "2026-01-01T00:00:00Z",
        periods=FLATLINE_WINDOW,
        freq="h",
    )
    df = sensor_dataset(timestamps, [10.0] * FLATLINE_WINDOW)

    result = detect_flatlines(df)

    assert result["is_flatline"].sum() == 1
    assert result["is_flatline"].iloc[-1] == True


def test_detect_spikes_flags_extreme_outlier():
    timestamps = pd.date_range(
        "2026-01-01T00:00:00Z",
        periods=30,
        freq="h",
    )
    df = sensor_dataset(timestamps, [10.0] * 29 + [100.0])

    result = detect_spikes(df)

    assert result["is_spike"].sum() == 1
    assert result["is_spike"].iloc[-1] == True


def test_detect_clock_duplicates_flags_duplicate_timestamps():
    df = sensor_dataset(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-01T00:00:00Z",
            "2026-01-01T01:00:00Z",
        ],
        [10.0, 11.0, 12.0],
    )

    result = detect_clock_duplicates(df)

    assert result["duplicate_timestamp"].tolist() == [True, True, False]


def test_detect_constant_sequences_flags_repeated_values_after_first_row():
    df = sensor_dataset(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-01T01:00:00Z",
            "2026-01-01T02:00:00Z",
        ],
        [10.0, 10.0, 12.0],
    )

    result = detect_constant_sequences(df)

    assert result["constant_sequence"].tolist() == [False, True, False]


def test_integrity_assessment_adds_flags_and_report_counts():
    timestamps = pd.date_range(
        "2026-01-01T00:00:00Z",
        periods=FLATLINE_WINDOW,
        freq="h",
    )
    df = sensor_dataset(timestamps, [10.0] * FLATLINE_WINDOW)

    result, report = integrity_assessment(df)

    assert {
        "time_gap_hours",
        "is_gap",
        "flatline_std",
        "is_flatline",
        "zscore",
        "is_spike",
        "duplicate_timestamp",
        "constant_sequence",
    }.issubset(result.columns)
    assert report["time_gaps"] == 0
    assert report["flatlines"] == 1
    assert report["spikes"] == 0
    assert report["duplicate_timestamps"] == 0
    assert report["constant_sequences"] == FLATLINE_WINDOW - 1
