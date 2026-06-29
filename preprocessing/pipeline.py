"""
Pipeline
"""

from .validation import validate_dataset
from .cleaning import clean_dataset
from .integrity import integrity_assessment
from .temporal import temporal_features
from .quality import quality_assessment
from .export import (
    export_dataset,
    export_report
)


def preprocess(

    df,

    output_dir

):

    validation = validate_dataset(df)

    df, cleaning = clean_dataset(df)

    df, integrity = integrity_assessment(df)

    df = temporal_features(df)

    df, quality = quality_assessment(df)

    export_dataset(

        df,

        output_dir

    )

    export_report(

        validation,

        cleaning,

        integrity,

        quality,

        output_dir

    )

    from .manifest import (
        build_manifest,
        save_manifest
    )

    manifest = build_manifest(

        df,

        "EDA",

        validation,

        cleaning,

        integrity,

        quality

    )

    save_manifest(

        manifest,

        output_dir

    )

    return df