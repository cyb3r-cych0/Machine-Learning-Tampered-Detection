"""
Statistics tests.
"""

import numpy as np

from detection.statistics import (

    bootstrap_metric,

    mcnemar_test,

    wilcoxon_test

)


def test_bootstrap():

    y = np.array(

        [0,0,1,1,1,0]

    )

    p = np.array(

        [0,0,1,0,1,0]

    )

    s = np.array(

        [

            0.1,

            0.2,

            0.9,

            0.4,

            0.8,

            0.2

        ]

    )

    ci = bootstrap_metric(

        y,

        p,

        s,

        "f1",

        n_bootstrap=100

    )

    assert "mean" in ci

    assert "lower" in ci

    assert "upper" in ci


def test_mcnemar():

    y = np.array(

        [0,0,1,1]

    )

    a = np.array(

        [0,0,1,0]

    )

    b = np.array(

        [0,1,1,1]

    )

    result = mcnemar_test(

        a,

        b,

        y

    )

    assert "p_value" in result


def test_wilcoxon():

    a = np.array(

        [0.1,0.2,0.3,0.4]

    )

    b = np.array(

        [0.2,0.3,0.4,0.5]

    )

    result = wilcoxon_test(

        a,

        b

    )

    assert "p_value" in result