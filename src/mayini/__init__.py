"""
MAYINI Deep Learning Framework
A comprehensive deep learning framework built from scratch in Python.
Includes: DeepLearning, Machine Learning, NEAT, Automated Preprocessing
"""

__version__ = "0.6.3"
__author__ = "Abhishek Adari"
__email__ = "abhishekadari85@gmail.com"

# ============================================================================
# SAFE IMPORTS - NO CIRCULAR DEPENDENCIES
# ============================================================================

# IMPORTANT: We import submodules WITHOUT their internal imports yet
# This prevents circular imports during package initialization

# Import the core Tensor class - this is safe
from .tensor import Tensor

# ============================================================================
# SUBMODULE REGISTRATION
# ============================================================================

# Import submodules (these are now "lazy" from parent's perspective)
from . import nn  # DeepLearning module (nn = neural networks)

# ============================================================================
# LAZY LOADING FOR OPTIONAL MODULES
# ============================================================================

# Use __getattr__ to lazy-load optional modules
# This prevents circular imports while maintaining backward compatibility

def __getattr__(name):
    """
    Lazy load submodules to avoid circular imports.

    When a submodule is requested via mayini.ml, mayini.neat, etc.,
    it's only loaded at that time, not during package initialization.

    This prevents circular dependencies between modules.
    """

    if name == 'ml':
        # Machine Learning module
        from . import ml
        return ml

    elif name == 'neat':
        # NEAT (NeuroEvolution of Augmenting Topologies) module
        from . import neat
        return neat

    elif name == 'preprocessing':
        # Automated Preprocessing module
        from . import preprocessing
        return preprocessing

    elif name == 'optim':
        # Optimization module (if exists)
        try:
            from . import optim
            return optim
        except ImportError:
            raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    elif name == 'training':
        # Training utilities module (if exists)
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

# Export commonly used classes directly from nn module for convenience
# Users can do: from mayini import Linear, LSTM, ReLU
try:
    from .nn import (
        # Core Module class
        Module,

        # Tensor class (already imported above, but for clarity)
        Tensor,

        # Layer classes
        Linear,
        Conv1D,
        Conv2D,
        Conv3D,

        # Recurrent layers
        LSTMCell,
        GRUCell,
        RNNCell,

        # Activation functions
        ReLU,
        Sigmoid,
        Tanh,
        Softmax,
        LeakyReLU,
        ELU,

        # Normalization
        BatchNorm1d,
        BatchNorm2d,
        LayerNorm,

        # Pooling
        MaxPool1d,
        MaxPool2d,
        AvgPool1d,
        AvgPool2d,

        # Dropout
        Dropout,

        # Loss functions
        MSELoss,
        CrossEntropyLoss,
        BCELoss,
        L1Loss,
    )
except ImportError as e:
    # If some classes don't exist, that's okay - they'll be available via nn module
    import warnings
    warnings.warn(f"Some nn classes could not be imported: {e}", ImportWarning)


# ============================================================================
# __all__ DEFINITION
# ============================================================================

__all__ = [
    # Version and metadata
    "__version__",
    "__author__",
    "__email__",

    # Core classes
    "Tensor",
    "Module",

    # Layer classes
    "Linear",
    "Conv1D",
    "Conv2D",
    "Conv3D",

    # Recurrent layers
    "LSTMCell",
    "GRUCell",
    "RNNCell",

    # Activation functions
    "ReLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
    "LeakyReLU",
    "ELU",

    # Normalization
    "BatchNorm1d",
    "BatchNorm2d",
    "LayerNorm",

    # Pooling
    "MaxPool1d",
    "MaxPool2d",
    "AvgPool1d",
    "AvgPool2d",

    # Dropout
    "Dropout",

    # Loss functions
    "MSELoss",
    "CrossEntropyLoss",
    "BCELoss",
    "L1Loss",

    # Submodules (accessible via lazy loading)
    "nn",  # Explicitly list as available
    "ml",
    "neat",
    "preprocessing",
    "optim",
    "training",
]

