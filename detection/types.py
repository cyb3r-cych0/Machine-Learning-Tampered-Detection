from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class DetectorResult:

    model_name: str

    prediction: np.ndarray

    score: np.ndarray

    threshold: float | None

    model: object
