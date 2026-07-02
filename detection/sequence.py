"""
detection/sequence.py

Sequence generation for temporal models.
"""

import numpy as np

from detection.config import (

    LSTM_SEQUENCE_LENGTH

)


def build_sequences(

    X,

    sequence_length=LSTM_SEQUENCE_LENGTH

):

    values = X.to_numpy(

        dtype=np.float32

    )

    sequences = []

    indices = []

    for i in range(

        sequence_length,

        len(values)

    ):

        sequences.append(

            values[

                i-sequence_length:i

            ]

        )

        indices.append(

            i

        )

    return (

        np.asarray(

            sequences,

            dtype=np.float32

        ),

        np.asarray(

            indices

        )

    )