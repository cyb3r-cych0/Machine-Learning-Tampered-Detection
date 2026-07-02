"""
Experiment tests.
"""

from detection.experiment import (

    DEFAULT_EXPERIMENT,

    SENSITIVITY_EXPERIMENTS

)


def test_default():

    assert DEFAULT_EXPERIMENT.random_seed == 42

    assert DEFAULT_EXPERIMENT.rolling_window > 0


def test_grid():

    assert len(

        SENSITIVITY_EXPERIMENTS

    ) == 9