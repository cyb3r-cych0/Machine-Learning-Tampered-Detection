"""
Detection model registry.
"""

from .isolation_forest import IsolationForestDetector

from .rolling_zscore import RollingZScoreDetector

from .lstm_autoencoder import LSTMAutoencoderDetector


MODEL_REGISTRY = {

    "Isolation Forest":

        IsolationForestDetector,

    "Rolling Z-Score":

        RollingZScoreDetector,

    "LSTM Autoencoder":

        LSTMAutoencoderDetector

}
