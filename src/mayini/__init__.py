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
from .training import DataLoad

