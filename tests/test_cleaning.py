import pandas as pd

from preprocessing.cleaning import clean_dataset, remove_duplicates


def dirty_dataset():
    return pd.DataFrame(
        {
            "country": ["ET", "ET", "ET", "ET", "ET"],
            "location_id": [101, 101, 101, 101, 101],
            "sensor_id": [201, 201, 201, 201, 201],
            "timestamp_utc": [
                "2026-01-01T02:00:00Z",
                "2026-01-01T02:00:00Z",
                "2026-01-01T01:00:00Z",
                "2026-01-01T03:00:00Z",
                None,
            ],
            "value": [10.0, 10.0, -5.0, None, 15.0],
        }
    )


def test_remove_duplicates_removes_duplicate_sensor_timestamp_rows():
    df = dirty_dataset().iloc[:2].copy()

    cleaned, removed = remove_duplicates(df)

    assert removed == 1
    assert len(cleaned) == 1


def test_clean_dataset_removes_invalid_rows_and_reports_counts():
    cleaned, report = clean_dataset(dirty_dataset())

    assert report["duplicates_removed"] == 1
    assert report["missing_removed"] == 2
    assert report["negative_removed"] == 1
    assert report["rows_after_cleaning"] == 1
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["value"] == 10.0


def test_clean_dataset_normalizes_and_sorts_timestamps():
    df = pd.DataFrame(
        {
            "country": ["ET", "ET"],
            "location_id": [101, 101],
            "sensor_id": [201, 201],
            "timestamp_utc": [
                "2026-01-01T02:00:00Z",
                "2026-01-01T01:00:00Z",
            ],
            "value": [20.0, 10.0],
        }
    )

    cleaned, _ = clean_dataset(df)

    assert cleaned["timestamp_utc"].is_monotonic_increasing
    assert list(cleaned["value"]) == [10.0, 20.0]
