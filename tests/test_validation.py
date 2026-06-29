import pandas as pd
import pytest

from preprocessing.validation import validate_dataset


def valid_dataset():
    return pd.DataFrame(
        {
            "country": ["ET", "ET"],
            "location_id": [101, 101],
            "sensor_id": [201, 201],
            "timestamp_utc": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"],
                utc=True,
            ),
            "value": [12.5, 14.0],
        }
    )


def test_validate_dataset_returns_report_for_valid_data():
    report = validate_dataset(valid_dataset())

    assert report["input_rows"] == 2
    assert report["input_columns"] == 5
    assert report["countries"] == 1
    assert report["stations"] == 1
    assert report["sensors"] == 1
    assert report["duplicates"] == 0
    assert report["sorted_before_cleaning"] is True


def test_validate_dataset_rejects_missing_required_column():
    df = valid_dataset().drop(columns=["value"])

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataset(df)


def test_validate_dataset_rejects_non_datetime_timestamp():
    df = valid_dataset()
    df["timestamp_utc"] = df["timestamp_utc"].astype(str)

    with pytest.raises(TypeError, match="timestamp_utc must be datetime"):
        validate_dataset(df)


def test_validate_dataset_rejects_missing_pm25_value():
    df = valid_dataset()
    df.loc[0, "value"] = None

    with pytest.raises(ValueError, match="missing PM2.5 values"):
        validate_dataset(df)
