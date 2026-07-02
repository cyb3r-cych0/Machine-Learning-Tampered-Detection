"""
detection/models/rolling_zscore.py

Historical Rolling Z-Score detector.
"""

import numpy as np
from detection.types import DetectorResult

from detection.config import (

    ZSCORE_THRESHOLD,

    ROLLING_WINDOW

)


class RollingZScoreDetector:

    def __init__(

        self,

        threshold=ZSCORE_THRESHOLD,

        window=ROLLING_WINDOW

    ):

        self.threshold = threshold

        self.window = window

        self.reference = {}

    def fit(

        self,

        baseline_df

    ):

        grouped = baseline_df.groupby(

            "sensor_id",

            sort=False

        )

        for sensor_id, sensor in grouped:

            values = sensor["value"]

            rolling = values.rolling(

                self.window,

                min_periods=1

            )

            mean = rolling.mean().iloc[-1]
            std = rolling.std().iloc[-1]

            if np.isnan(mean):
                mean = float(values.mean())

            if np.isnan(std) or std <= 0:
                std = 1e-6

            self.reference[sensor_id] = {

                "mean": float(mean),

                "std": float(std)

            }

        return self

    def predict(

            self,

            attacked_df

    ):

        prediction = np.zeros(

            len(attacked_df),

            dtype=int

        )

        score = np.zeros(

            len(attacked_df),

            dtype=float

        )

        for idx, row in attacked_df.iterrows():

            sensor = row["sensor_id"]

            if sensor not in self.reference:
                continue

            value = row["value"]

            mean = self.reference[sensor]["mean"]

            std = self.reference[sensor]["std"]

            z = abs(

                (value - mean) / std

            )

            if not np.isfinite(z):
                z = 0.0

            score[idx] = z

            prediction[idx] = int(

                z >= self.threshold

            )

        score = np.nan_to_num(

            score,

            nan=0.0,

            posinf=0.0,

            neginf=0.0

        )

        return DetectorResult(

            model_name="Historical Rolling Z-Score",

            prediction=prediction,

            score=score,

            threshold=self.threshold,

            model=self.reference

        )