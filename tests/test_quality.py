import pandas as pd

from preprocessing.quality import (
    compute_quality_score,
    flag_duplicate,
    flag_flatline,
    flag_gap,
    flag_missing,
    flag_outlier,
    quality_assessment,
    quality_category,
    quality_report,
)


def quality_dataset():
    return pd.DataFrame(
        {
            "country": ["ET", "ET", "ET", "ET"],
            "location_id": [101, 101, 101, 101],
            "sensor_id": [201, 201, 201, 201],
            "timestamp_utc": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T01:00:00Z",
                    "2026-01-01T02:00:00Z",
                    "2026-01-01T03:00:00Z",
                ],
                utc=True,
            ),
            "value": [10.0, None, 30.0, 40.0],
            "is_gap": [False, True, False, False],
            "is_flatline": [False, False, True, False],
            "duplicate_timestamp": [False, False, False, True],
            "is_spike": [False, False, True, False],
            "constant_sequence": [False, False, False, True],
        }
    )


def test_flag_missing_marks_rows_with_missing_pm25_values():
    result = flag_missing(quality_dataset())

    assert result["is_missing"].tolist() == [False, True, False, False]


def test_flag_duplicate_uses_existing_duplicate_timestamp_flag():
    result = flag_duplicate(quality_dataset())

    assert result["is_duplicate"].tolist() == [False, False, False, True]


def test_flag_duplicate_defaults_to_false_when_source_column_is_missing():
    df = quality_dataset().drop(columns=["duplicate_timestamp"])

    result = flag_duplicate(df)

    assert result["duplicate_timestamp"].tolist() == [False] * 4
    assert result["is_duplicate"].tolist() == [False] * 4


def test_flag_outlier_uses_existing_spike_flag():
    result = flag_outlier(quality_dataset())

    assert result["is_outlier"].tolist() == [False, False, True, False]


def test_flag_outlier_defaults_to_false_when_spike_column_is_missing():
    df = quality_dataset().drop(columns=["is_spike"])

    result = flag_outlier(df)

    assert result["is_spike"].tolist() == [False] * 4
    assert result["is_outlier"].tolist() == [False] * 4


def test_gap_and_flatline_flags_default_to_false_when_missing():
    df = quality_dataset().drop(columns=["is_gap", "is_flatline"])

    result = flag_flatline(flag_gap(df))

    assert result["is_gap"].tolist() == [False] * 4
    assert result["is_flatline"].tolist() == [False] * 4


def test_compute_quality_score_applies_twenty_point_penalty_per_flag():
    df = flag_outlier(flag_duplicate(quality_dataset()))

    result = compute_quality_score(df)

    assert result["quality_score"].tolist() == [100, 80, 60, 60]


def test_compute_quality_score_clips_at_zero():
    df = pd.DataFrame(
        {
            "is_gap": [True],
            "is_flatline": [True],
            "is_duplicate": [True],
            "is_outlier": [True],
            "constant_sequence": [True],
        }
    )

    result = compute_quality_score(df)

    assert result.loc[0, "quality_score"] == 0


def test_quality_category_assigns_expected_classes():
    df = pd.DataFrame(
        {
            "quality_score": [100, 80, 60, 40],
        }
    )

    result = quality_category(df)

    assert result["quality_class"].tolist() == [
        "Excellent",
        "Good",
        "Fair",
        "Poor",
    ]


def test_quality_report_counts_classes_and_mean_score():
    df = pd.DataFrame(
        {
            "quality_score": [100, 80, 60, 40],
            "quality_class": ["Excellent", "Good", "Fair", "Poor"],
        }
    )

    report = quality_report(df)

    assert report["excellent"] == 1
    assert report["good"] == 1
    assert report["fair"] == 1
    assert report["poor"] == 1
    assert report["mean_quality_score"] == 70.0


def test_quality_assessment_returns_augmented_dataframe_and_report():
    result, report = quality_assessment(quality_dataset())

    assert {
        "is_missing",
        "is_duplicate",
        "is_outlier",
        "is_gap",
        "is_flatline",
        "constant_sequence",
        "quality_score",
        "quality_class",
    }.issubset(result.columns)
    assert result["quality_score"].tolist() == [100, 80, 60, 60]
    assert report["excellent"] == 1
    assert report["good"] == 1
    assert report["fair"] == 2
    assert report["poor"] == 0
    assert report["mean_quality_score"] == 75.0
