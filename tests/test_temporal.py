import numpy as np
import pandas as pd

from preprocessing.temporal import (
    calendar_features,
    cyclic_features,
    difference_features,
    ema_features,
    lag_features,
    rolling_features,
    temporal_features,
)


def sensor_dataset(values, start="2026-01-03T00:00:00Z"):
    return pd.DataFrame(
        {
            "country": ["ET"] * len(values),
            "location_id": [101] * len(values),
            "sensor_id": [201] * len(values),
            "timestamp_utc": pd.date_range(
                start,
                periods=len(values),
                freq="h",
            ),
            "value": values,
        }
    )


def test_calendar_features_adds_expected_timestamp_parts():
    df = sensor_dataset([10.0], start="2026-01-03T05:00:00Z")

    result = calendar_features(df)

    assert result.loc[0, "hour"] == 5
    assert result.loc[0, "day"] == 3
    assert result.loc[0, "weekday"] == 5
    assert result.loc[0, "week"] == 1
    assert result.loc[0, "month"] == 1
    assert result.loc[0, "quarter"] == 1
    assert result.loc[0, "year"] == 2026
    assert result.loc[0, "dayofyear"] == 3
    assert result.loc[0, "weekend"] == 1


def test_cyclic_features_adds_hour_and_month_encodings():
    df = calendar_features(
        sensor_dataset([10.0], start="2026-03-01T06:00:00Z")
    )

    result = cyclic_features(df)

    assert np.isclose(result.loc[0, "hour_sin"], 1.0)
    assert np.isclose(result.loc[0, "hour_cos"], 0.0)
    assert np.isclose(result.loc[0, "month_sin"], 1.0)
    assert np.isclose(result.loc[0, "month_cos"], 0.0)


def test_lag_features_are_calculated_per_sensor():
    df = sensor_dataset([10.0, 20.0, 30.0, 40.0])

    result = lag_features(df)

    assert pd.isna(result.loc[0, "lag_1"])
    assert result.loc[1, "lag_1"] == 10.0
    assert result.loc[3, "lag_3"] == 10.0
    assert pd.isna(result.loc[3, "lag_6"])
    assert pd.isna(result.loc[3, "lag_12"])


def test_rolling_features_adds_mean_std_and_median():
    df = sensor_dataset([10.0, 20.0, 30.0])

    result = rolling_features(df)

    assert result.loc[0, "rolling_mean"] == 10.0
    assert result.loc[1, "rolling_mean"] == 15.0
    assert result.loc[2, "rolling_mean"] == 20.0
    assert pd.isna(result.loc[0, "rolling_std"])
    assert np.isclose(result.loc[1, "rolling_std"], 7.0710678118654755)
    assert result.loc[2, "rolling_median"] == 20.0


def test_ema_features_adds_exponential_moving_average():
    df = sensor_dataset([10.0, 20.0, 30.0])

    result = ema_features(df)

    assert result.loc[0, "ema"] == 10.0
    assert result["ema"].between(10.0, 30.0).all()
    assert result["ema"].is_monotonic_increasing


def test_difference_features_adds_difference_and_rate_of_change():
    df = lag_features(sensor_dataset([10.0, 20.0, 30.0]))

    result = difference_features(df)

    assert pd.isna(result.loc[0, "difference"])
    assert result.loc[1, "difference"] == 10.0
    assert result.loc[2, "difference"] == 10.0
    assert result.loc[1, "rate_of_change"] == 1.0
    assert result.loc[2, "rate_of_change"] == 0.5


def test_difference_features_uses_nan_for_zero_lag_rate_of_change():
    df = lag_features(sensor_dataset([0.0, 10.0]))

    result = difference_features(df)

    assert pd.isna(result.loc[1, "rate_of_change"])


def test_temporal_features_adds_all_expected_columns():
    result = temporal_features(sensor_dataset([10.0, 20.0, 30.0]))

    expected_columns = {
        "hour",
        "day",
        "weekday",
        "week",
        "month",
        "quarter",
        "year",
        "dayofyear",
        "weekend",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "lag_1",
        "lag_3",
        "lag_6",
        "lag_12",
        "rolling_mean",
        "rolling_std",
        "rolling_median",
        "ema",
        "difference",
        "rate_of_change",
    }

    assert expected_columns.issubset(result.columns)
    assert len(result) == 3
