"""
detection/models/lstm_autoencoder.py
"""

import numpy as np

from detection.sequence import build_sequences

from detection.types import DetectorResult

from tensorflow.keras.models import Model

from tensorflow.keras.layers import (

    Input,

    LSTM,

    RepeatVector,

    TimeDistributed,

    Dense

)

from tensorflow.keras.callbacks import EarlyStopping

from detection.config import (

    LSTM_LATENT_DIM,

    LSTM_BATCH_SIZE,

    LSTM_EPOCHS,

    LSTM_THRESHOLD_PERCENTILE

)


class LSTMAutoencoderDetector:

    def __init__(self):

        self.model = None

        self.threshold = None

    def build(

        self,

        sequence_length,

        n_features

    ):

        inputs = Input(

            shape=(

                sequence_length,

                n_features

            )

        )

        encoded = LSTM(

            LSTM_LATENT_DIM,

            activation="tanh"

        )(inputs)

        decoded = RepeatVector(

            sequence_length

        )(encoded)

        decoded = LSTM(

            LSTM_LATENT_DIM,

            activation="tanh",

            return_sequences=True

        )(decoded)

        outputs = TimeDistributed(

            Dense(

                n_features

            )

        )(decoded)

        self.model = Model(

            inputs,

            outputs

        )

        self.model.compile(

            optimizer="adam",

            loss="mse"

        )

    def fit(

        self,

        X_train,

        epochs=LSTM_EPOCHS,

    ):

        self.build(

            X_train.shape[1],

            X_train.shape[2]

        )

        early_stop = EarlyStopping(

            monitor="loss",

            patience=5,

            restore_best_weights=True

        )

        self.model.fit(

            X_train,

            X_train,

            epochs=epochs,

            batch_size=LSTM_BATCH_SIZE,

            shuffle=False,

            verbose=0,

            callbacks=[

                early_stop

            ]

        )

        reconstruction = self.model.predict(

            X_train,

            verbose=0

        )

        train_error = np.mean(

            np.square(

                X_train -

                reconstruction

            ),

            axis=(1, 2)

        )

        self.threshold = np.percentile(

            train_error,

            LSTM_THRESHOLD_PERCENTILE

        )

        return self

    def predict(

        self,

        X_test

    ):

        reconstruction = self.model.predict(

            X_test,

            verbose=0

        )

        score = np.mean(

            np.square(

                X_test -

                reconstruction

            ),

            axis=(1, 2)

        )

        prediction = (

            score >

            self.threshold

        ).astype(int)

        return DetectorResult(

            model_name="LSTM Autoencoder",

            prediction=prediction,

            score=score,

            threshold=self.threshold,

            model=self.model

        )

    def predict_from_features(

            self,

            X_test

    ):
        X_test, _ = build_sequences(

            X_test

        )

        return self.predict(

            X_test

        )