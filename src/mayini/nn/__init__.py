"""
Neural Network components for MAYINI Deep Learning Framework.
"""

# Base classes
from .modules import Module, Sequential

# Core layers
from .modules import (
    Linear,
    Conv2D,
    MaxPool2D,
    AvgPool2D,
    Dropout,
    BatchNorm1d,
    Flatten,
)

# Activation modules
from .activations import ReLU, Sigmoid, Tanh, Softmax, GELU, LeakyReLU

# RNN components
from .rnn import RNNCell, LSTMCell, GRUCell, RNN

# Loss functions
from .losses import MSELoss, MAELoss, CrossEntropyLoss, BCELoss, HuberLoss

__all__ = [
    # Base classes
    "Module",
    "Sequential",
    # Core layers
    "Linear",
    "Conv2D",
    "MaxPool2D",
    "AvgPool2D",
    "Dropout",
    "BatchNorm1d",
    "Flatten",
    # Activations
    "ReLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "GELU",
    "LeakyReLU",
    # RNN components
    "RNNCell",
    "LSTMCell",
    "GRUCell",
    "RNN",
    # Loss functions
    "MSELoss",
    "MAELoss",
    "CrossEntropyLoss",
    "BCELoss",
    "HuberLoss",
]

__all__ = []


# ============================================================================
# FILE 49: src/mayini/optim/__init__.py
# BLACK-FORMATTED VERSION
# ============================================================================

"""Optimizer utilities"""

# Note: This is a minimal implementation as the main optimizer
# functionality already exists in the mayini library

__all__ = []


# ============================================================================
# FILE 50: src/mayini/__init__.py (Main init - if needed for completeness)
# BLACK-FORMATTED VERSION
# ============================================================================

