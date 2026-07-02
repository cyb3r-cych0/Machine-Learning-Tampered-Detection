"""
detection/experiment.py
"""

from dataclasses import dataclass

from detection.config import *

@dataclass(frozen=True)
class Experiment:

    name: str

    contamination: float | str

    rolling_window: int

    random_seed: int

DEFAULT_EXPERIMENT = Experiment(

    name="Default",

    contamination=IF_CONTAMINATION,

    rolling_window=ROLLING_WINDOW,

    random_seed=RANDOM_SEED

)

SENSITIVITY_EXPERIMENTS = [

    Experiment(

        name=f"W{window}_C{cont}",

        contamination=cont,

        rolling_window=window,

        random_seed=RANDOM_SEED

    )

    for window in WINDOW_SIZES

    for cont in CONTAMINATION_LEVELS

]