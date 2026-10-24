"""Causal native-event state-space denoising.

The package intentionally keeps event-state extraction separate from the tiny
learned classifier.  This mirrors the intended FPGA datapath: a fixed local
state engine followed by a quantizable multilayer perceptron.
"""

from .core import NativeStateBank, StateConfig, bidirectional_features, feature_names
from .analytic import analytic_contrast_score
from .model import TinyEventClassifier, load_checkpoint, save_checkpoint
from .advanced import AdvancedConfig, MotionConfig, advanced_block_features
from .advanced_model import AdvancedEventClassifier

__all__ = [
    "NativeStateBank",
    "AdvancedConfig",
    "AdvancedEventClassifier",
    "MotionConfig",
    "StateConfig",
    "TinyEventClassifier",
    "analytic_contrast_score",
    "bidirectional_features",
    "advanced_block_features",
    "feature_names",
    "load_checkpoint",
    "save_checkpoint",
]
