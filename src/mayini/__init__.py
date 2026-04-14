"""
MAYINI Deep Learning Framework
A comprehensive deep learning framework built from scratch in Python.
Includes: DeepLearning, Machine Learning, NEAT, Automated Preprocessing
"""

__version__ = "0.8.1"
__author__ = "Abhishek Adari"
__email__ = "abhishekadari85@gmail.com"

# ============================================================================
# SAFE IMPORTS - NO CIRCULAR DEPENDENCIES
# ============================================================================

# Import the core Tensor class
from .tensor import Tensor
from .ops import concatenate, stack

# ============================================================================
# SUBMODULE REGISTRATION
# ============================================================================

from . import nn

# ============================================================================
# LAZY LOADING FOR OPTIONAL MODULES
# ============================================================================

def __getattr__(name):
    if name == 'ml':
        from . import ml
        return ml
    elif name == 'neat':
        from . import neat
        return neat
    elif name == 'preprocessing':
        from . import preprocessing
        return preprocessing
    elif name == 'optim':
        try:
            from . import optim
            return optim
        except ImportError:
            raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    elif name == 'training':
        try:
            from . import training
            return training
        except ImportError:
            raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# ============================================================================
# PUBLIC API - CLASSES & FUNCTIONS DIRECTLY ACCESSIBLE
# ============================================================================

try:
    from .nn import (
        Module,
        Linear,
        Conv2D,
        LSTMCell,
        GRUCell,
        RNNCell,
        ReLU,
        Sigmoid,
        Tanh,
        Softmax,
        LeakyReLU,
        BatchNorm1d,
        MaxPool2D,
        AvgPool2D,
        Dropout,
        MSELoss,
        CrossEntropyLoss,
        BCELoss,
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Some nn classes could not be imported: {e}", ImportWarning)

# ============================================================================
# __all__ DEFINITION
# ============================================================================

__all__ = [
    "__version__", "__author__", "__email__",
    "Tensor", "Module",
    "Linear", "Conv2D",
    "LSTMCell", "GRUCell", "RNNCell",
    "ReLU", "Sigmoid", "Tanh", "Softmax", "LeakyReLU",
    "BatchNorm1d", "MaxPool2D", "AvgPool2D", "Dropout",
    "MSELoss", "CrossEntropyLoss", "BCELoss",
    "nn", "ml", "neat", "preprocessing", "optim", "training",
]
