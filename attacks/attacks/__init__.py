"""
Attack registry.
"""

from .constant_bias import apply_constant_bias
from .gradual_drift import apply_gradual_drift
from .spike_suppression import apply_spike_suppression
from .random_stealth import apply_random_stealth


ATTACK_REGISTRY = {

    "Constant Bias": apply_constant_bias,

    "Gradual Drift": apply_gradual_drift,

    "Spike Suppression": apply_spike_suppression,

    "Random Stealth": apply_random_stealth,

}