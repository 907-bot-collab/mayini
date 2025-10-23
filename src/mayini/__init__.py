"""
MAYINI Deep Learning Framework
A comprehensive deep learning framework built from scratch in Python.
"""

__version__ = "0.1.4"
__author__ = "Abhishek Adari"
__email__ = "abhishekadari85@gmail.com"

# Core components
from .tensor import Tensor

# Neural network modules
from .nn import (
    Module,
    Sequential,
    Linear,
    Conv2D,
    MaxPool2D,
    AvgPool2D,
    Dropout,
    BatchNorm1d,
    Flatten,
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    GELU,
    LeakyReLU,
    RNNCell,
    LSTMCell,
    GRUCell,
    RNN,
    MSELoss,
    MAELoss,
    CrossEntropyLoss,
    BCELoss,
    HuberLoss,
)

# Optimizers
from .optim import SGD, Adam, AdamW, RMSprop

# Training utilities
from .training import DataLoader, Trainer, Metrics, EarlyStopping

# Activation functions (functional interface)
from .nn.activations import relu, sigmoid, tanh, softmax, gelu, leaky_relu

__all__ = [
    # Core
    "Tensor",
    # Base classes
    "Module",
    "Sequential",
    # Layers
    "Linear",
    "Conv2D",
    "MaxPool2D",
    "AvgPool2D",
    "Dropout",
    "BatchNorm1d",
    "Flatten",
    # Activations (modules)
    "ReLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "GELU",
    "LeakyReLU",
    # Activations (functions)
    "relu",
    "sigmoid",
    "tanh",
    "softmax",
    "gelu",
    "leaky_relu",
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
    # Optimizers
    "SGD",
    "Adam",
    "AdamW",
    "RMSprop",
    # Training
    "DataLoader",
    "Trainer",
    "Metrics",
    "EarlyStopping",
]
"""mayini/__init__.py - Updated"""
from . import nn
from . import ml
from . import neat
from . import preprocessing
from . import optim
from . import training
from .tensor import Tensor

__version__ = "0.2.0"

__all__ = [
    'Tensor',
    'nn',
    'ml',
    'neat',
    'preprocessing',
    'optim',
    'training'
]
from . import preprocessing

# Add to __all__ list:
__all__ = [
    'Tensor',
    'nn',
    'optim',
    'training',
    'ml',
    'neat',
    'preprocessing',  # ADD THIS
]
