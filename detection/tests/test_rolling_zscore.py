"""
Rolling Z-Score tests.
"""

import numpy as np

from detection.loader import load_datasets

from detection.models import MODEL_REGISTRY

from detection.config import *


TEST_ROWS = 5000


def test_rolling_zscore():

    baseline, attacked, _, _, _ = load_datasets(

        BASELINE_DATASET,

        BASELINE_CSV,

        ATTACKED_DATASET,

        ATTACKED_CSV,

        GROUND_TRUTH

    )

    baseline = baseline.iloc[:TEST_ROWS].copy()

    attacked = attacked.iloc[:TEST_ROWS].copy()

    detector = MODEL_REGISTRY[

        "Rolling Z-Score"

    ]()

    detector.fit(

        baseline

    )

    result = detector.predict(

        attacked

    )

    assert len(

        result.prediction

    ) == len(

        attacked

    )

    assert len(

        result.score

    ) == len(

        attacked

    )

    assert np.isin(

        result.prediction,

        [0, 1]

    ).all()

    assert np.isfinite(

        result.score

    ).all()

    assert result.threshold > 0

    assert result.model_name == "Historical Rolling Z-Score"