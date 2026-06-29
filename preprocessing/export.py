"""
Exports the canonical preprocessing dataset.
"""

import json


def export_dataset(
    df,
    output_dir,
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_file = output_dir / "baseline_dataset.csv"

    parquet_file = output_dir / "baseline_dataset.parquet"

    df.to_csv(
        csv_file,
        index=False
    )

    df.to_parquet(
        parquet_file,
        index=False,
        engine="pyarrow"
    )

    return csv_file, parquet_file


def export_report(
    validation_report,
    cleaning_report,
    integrity_report,
    quality_report,
    output_dir
):

    report = {
        "validation": validation_report,
        "cleaning": cleaning_report,
        "integrity": integrity_report,
        "quality": quality_report
    }

    with open(
        output_dir /
        "preprocessing_report.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            report,
            f,
            indent=4,
            default=str
        )